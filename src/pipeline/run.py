from ..utils.data_loader import StreamingAudioDataset
from ..utils.logger import Logger
from .filterer import HardFilterActor, soft_filter_task
from torch.utils.data import DataLoader
import json
import os
import time
import ray
from typing import Optional

from huggingface_hub import get_token as hf_get_token

with open("thresholds.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
def _resolve_hf_token() -> Optional[str]:
	# Prefer explicit env var, then fall back to token saved by `hf auth login`.
	return os.getenv("HF_TOKEN") or hf_get_token()

def collate_fn(batch):
    return batch

def run_pipeline(
		manifest_path: str = "./data/manifests/test_manifest.jsonl",
		output_dir: str = "./output",
		target_sr: int = 16000,
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

	os.makedirs(output_dir, exist_ok=True)
	output_file = os.path.join(output_dir, "filtered_manifest.jsonl")
	logger.info(f"Writing filtered output to: {output_file}")

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

	#!TODO: Remove before submission
	if debug_workers:
		hard_identity = ray.get(hard_filter_actor.get_identity.remote())
		logger.info(f"Hard actor identity: {hard_identity}")

	try:
		with open(output_file, "w", encoding="utf-8") as output_f:
			init_time = time.time()
			total_rejects = 0
			total_accepts = 0
			rejects_due_to = [0] * 8
			for idx, batch in enumerate(dataIterator):
				start = time.time()
				soft_futures = [soft_filter_task.remote(sample) for sample in batch]

				soft_outputs = [s for s in ray.get(soft_futures) if s is not None]

				#!TODO: Remove before submission
				if debug_workers:
					soft_pids = sorted({int(item["_ray"]["pid"]) for item in soft_outputs if "_ray" in item})
					logger.info(f"Soft workers used this batch: count={len(soft_pids)} pids={soft_pids}")

				if not soft_outputs:
					continue

				filtered_batch = [item["sample"] for item in soft_outputs]
				soft_results = [item["soft_metrics"] for item in soft_outputs]

				hard_future = hard_filter_actor.process_batch.remote(filtered_batch)
				hard_results = ray.get(hard_future)

				for soft_result, hard_result in zip(soft_results, hard_results):
					result = {**soft_result, **hard_result}
					
					reasons = []

					if not (cfg["min_duration"] < result["duration"] <= cfg["max_duration"]):
						reasons.append(0)

					if result["c50"] < cfg["min_c50"]:
						reasons.append(1)

					if result["snr"] < cfg["min_snr"]:
						reasons.append(2)

					if result["silence_ratio"] > cfg["max_silence_ratio"]:
						reasons.append(3)

					if result["clipping_ratio"] >= cfg["max_clipping_ratio"]:
						reasons.append(4)

					if result["vad_ratio"] < cfg["min_vad_ratio"]:
						reasons.append(5)

					if result["asr"] < cfg["min_asr_conf"]:
						reasons.append(6)

					pass_nisqa = sum([
						result["mos"] >= cfg["min_mos"],
						result["noisiness"] >= cfg["min_noisiness"],
						result["discontinuity"] >= cfg["min_discontinuity"],
						result["coloration"] >= cfg["min_coloration"],
						result["loudness"] >= cfg["min_loudness"],
					]) >= 4

					if not pass_nisqa:
						reasons.append(7)

					if reasons:
						result["status"] = "Reject"
						for r in reasons:
							rejects_due_to[r] += 1
					
						total_rejects += 1
					else:
						result["status"] = "Accept"
						total_accepts += 1
					output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
				
				end = time.time()
				logger.info(f"Batch {idx+1} processed in {end - start} seconds")
			logger.info(f"{'='*10} Final Report {'='*60}")
			logger.info(f"Total time taken: {end - init_time} (Including around 15 seconds of ray initialization time)")
			logger.info(f"Total Samples: {total_accepts + total_rejects}")
			logger.info(f"Total Rejects: {total_rejects}")
			logger.info(f"Total Accepts: {total_accepts}")
			logger.info(f"Rejected due to Duration: {rejects_due_to[0]}")
			logger.info(f"Rejected due to C50: {rejects_due_to[1]}")
			logger.info(f"Rejected due to Signal-to-Noise Ratio: {rejects_due_to[2]}")
			logger.info(f"Rejected due to Silence Ratio: {rejects_due_to[3]}")
			logger.info(f"Rejected due to Clipping Ratio: {rejects_due_to[4]}")
			logger.info(f"Rejected due to VAD Ratio: {rejects_due_to[5]}")
			logger.info(f"Rejected due to ASR Confidence: {rejects_due_to[6]}")
			logger.info(f"Rejected due to NISQA Metrics: {rejects_due_to[7]}")
			logger.debug("Note that a sample can be rejected due to one or more reasons!")
			logger.info('='*85)
			logger.debug(cfg)

		return
	finally:
		ray.shutdown()
