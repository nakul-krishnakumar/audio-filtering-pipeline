from ..utils.data_loader import StreamingAudioDataset
from ..utils.logger import Logger
from .filterer import HardFilterActor, soft_filter_task
from torch.utils.data import DataLoader
import os
import sys
import ray
from typing import Optional

from huggingface_hub import get_token as hf_get_token


def _resolve_hf_token() -> Optional[str]:
	# Prefer explicit env var, then fall back to token saved by `hf auth login`.
	return os.getenv("HF_TOKEN") or hf_get_token()

def collate_fn(batch):
    return batch


def _build_ray_init_kwargs() -> dict:
	# Force workers to use the already-active interpreter instead of spawning a
	# fresh uv-managed env under /tmp, which can exceed disk quotas.
	temp_dir = os.getenv("RAY_TEMP_DIR", os.path.expanduser("~/.ray_tmp"))
	os.makedirs(temp_dir, exist_ok=True)

	return {
		"ignore_reinit_error": True,
		"_temp_dir": temp_dir,
		"runtime_env": {"py_executable": sys.executable},
	}

def run_pipeline(
		manifest_path: str = "./data/manifests/test_manifest.jsonl",
		target_sr: int = 160000,
        batch_size: int = 8,
        num_data_loader_workers: int = 4,
		debug_workers: bool = False,
	):

	ray.init(address="auto")
	logger = Logger()
	logger.info("Pipeline started")
	if debug_workers:
		cluster = ray.cluster_resources()
		logger.info(f"Ray cluster resources: {cluster}")
	hf_token = _resolve_hf_token()
	if hf_token:
		logger.info("Hugging Face token resolved (env or local cache).")
	else:
		logger.warn("No Hugging Face token found. Hard filter will run without pyannote model.")

	dataset = StreamingAudioDataset(
		logger=logger,
		manifest_path=manifest_path,
		target_sr=target_sr,
	)

	logger.info("Data loader initialized")
	dataIterator = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_data_loader_workers,
            collate_fn=collate_fn,
	)

	hard_filter_actor = HardFilterActor.remote(hf_token=hf_token)
	if debug_workers:
		hard_identity = ray.get(hard_filter_actor.get_identity.remote())
		logger.info(f"Hard actor identity: {hard_identity}")

	try:
		for idx, batch in enumerate(dataIterator):
			logger.info(f"Batch processing start for batch {idx+1}")
			soft_futures = [soft_filter_task.remote(sample) for sample in batch]

			soft_outputs = [s for s in ray.get(soft_futures) if s is not None]
			if debug_workers:
				soft_pids = sorted({int(item["_ray"]["pid"]) for item in soft_outputs if "_ray" in item})
				logger.info(f"Soft workers used this batch: count={len(soft_pids)} pids={soft_pids}")

			if not soft_outputs:
				continue

			filtered_batch = [item["sample"] for item in soft_outputs]
			soft_results = [item["soft_metrics"] for item in soft_outputs]

			hard_future = hard_filter_actor.process_batch.remote(filtered_batch)
			hard_results = ray.get(hard_future)

			logger.info(f"soft_count={len(soft_results)} | hard_count={len(hard_results)}")

		return
	finally:
		ray.shutdown()
