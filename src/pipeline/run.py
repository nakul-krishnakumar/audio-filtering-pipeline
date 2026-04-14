from ..utils.logger import Logger
from ..utils.data_loader import StreamingAudioDataset
from .filterer import HardFilterActor, soft_filter_task

import json
import os
import time
import ray
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from huggingface_hub import get_token as hf_get_token

load_dotenv()

# Set of unsupported languages by whisper to do ASR and LID
UNSUPPORTED_LANGS = [
	"brx", #bodo
	"doi", #dogri
	"ks", # kashmiri
	"kok", # konkani
	"mai", # maithili
	"mni", # manipuri
	"sat", # santali
	"or", # odia or oria
]

with open("thresholds.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

# Why a custom collate function?
# - Each audio sample has different shaped tensors, [1, num_samples]
# - where the num_samples depend on sampling rate (16kHz fixed here) and the 
# 	size of the audio which may vary
# - DataLoader will internally try to stack each tensor [2, 1, x]
# - But in out case the `x` is varying across audio samples and this stacking will break
# - So we make a custom collate function to directly pass the batch instead of stacking it
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
		logger.info("Ray cluster resources: %s", cluster)

	hf_token = os.getenv("HF_TOKEN") or hf_get_token()
	if hf_token:
		logger.info("Hugging Face token resolved (env or local cache).")
	else:
		logger.warn("No Hugging Face token found. Hard filter will run without pyannote model.")

	os.makedirs(output_dir, exist_ok=True)
	output_file = os.path.join(output_dir, "filtered_manifest.jsonl")
	logger.info("Writing filtered output to: %s", output_file)

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
		logger.info("Hard actor identity: %s", hard_identity)

	try:
		with open(output_file, "w", encoding="utf-8") as output_f:
			init_time = time.time()
			total_rejects = 0
			total_accepts = 0
			rejects_due_to = [0] * 9
			for idx, batch in enumerate(dataIterator):
				start = time.time()
				soft_futures = [soft_filter_task.remote(sample) for sample in batch]
				# Futures are ray object references, which will produce a result later

				soft_outputs = [s for s in ray.get(soft_futures) if s is not None]

				if debug_workers:
					soft_pids = sorted({int(item["_ray"]["pid"]) for item in soft_outputs if "_ray" in item})
					logger.info("Soft workers used this batch: count=%d pids=%s\n", len(soft_pids), soft_pids)

				if not soft_outputs:
					continue

				filtered_batch = [item["sample"] for item in soft_outputs]
				soft_results = [item["soft_metrics"] for item in soft_outputs]

				hard_future = hard_filter_actor.process_batch.remote(filtered_batch)
				hard_results = ray.get(hard_future)

				for soft_result, hard_result in zip(soft_results, hard_results):
					result = {**soft_result, **hard_result}
					
					reasons = []
					reason_names = []

					if not (cfg["min_duration"] < result["duration"] <= cfg["max_duration"]):
						reasons.append(0)
						reason_names.append("Duration")

					if result["c50"] < cfg["min_c50"]:
						reasons.append(1)
						reason_names.append("C50")

					if result["snr"] < cfg["min_snr"]:
						reasons.append(2)
						reason_names.append("SNR")

					if result["silence_ratio"] > cfg["max_silence_ratio"]:
						reasons.append(3)
						reason_names.append("Silence Ratio")

					if result["clipping_ratio"] >= cfg["max_clipping_ratio"]:
						reasons.append(4)
						reason_names.append("Clipping Ratio")

					if result["vad_ratio"] < cfg["min_vad_ratio"]:
						reasons.append(5)
						reason_names.append("VAD Ratio")

					if result["expected_lang"] not in UNSUPPORTED_LANGS:
						if result["asr"] < cfg["min_asr_conf"]:
							reasons.append(6)
							reason_names.append("Low ASR Confidence")
						
						if result["pred_lang"] != result["expected_lang"]:
							logger.warn(f"Expected lang: {result['expected_lang']} | Predicted lang: {result['pred_lang']}")
							reasons.append(7)
							reason_names.append("Incorrect Language")

					pass_nisqa = sum([
						result["mos"] >= cfg["min_mos"],
						result["noisiness"] >= cfg["min_noisiness"],
						result["discontinuity"] >= cfg["min_discontinuity"],
						result["coloration"] >= cfg["min_coloration"],
						result["loudness"] >= cfg["min_loudness"],
					]) >= 4

					if not pass_nisqa:
						reasons.append(8)
						reason_names.append("Low NISQA-MOS Score")

					if reasons:
						result["status"] = "Reject"
						for r in reasons:
							rejects_due_to[r] += 1
						total_rejects += 1
					else:
						result["status"] = "Accept"
						total_accepts += 1
					
					result["reject_due_to"] = reason_names
					output_f.write(json.dumps(result, ensure_ascii=False) + "\n")
				
				end = time.time()
				logger.info("Batch %d processed in %.3f seconds", idx + 1, end - start)

			final_end = time.time()
			logger.info("%s Final Report %s", '='*20, '='*50)
			logger.info("Total time taken: %.3f (Including around 15 seconds of ray initialization time)", final_end - init_time)
			logger.info("Total Samples: %d", total_accepts + total_rejects)
			logger.info("Total Rejects: %d", total_rejects)
			logger.info("Total Accepts: %d", total_accepts)
			logger.info("Rejected due to Duration: %d", rejects_due_to[0])
			logger.info("Rejected due to C50: %d", rejects_due_to[1])
			logger.info("Rejected due to Signal-to-Noise Ratio: %d", rejects_due_to[2])
			logger.info("Rejected due to Silence Ratio: %d", rejects_due_to[3])
			logger.info("Rejected due to Clipping Ratio: %d", rejects_due_to[4])
			logger.info("Rejected due to VAD Ratio: %d", rejects_due_to[5])
			logger.info("Rejected due to ASR Confidence: %d", rejects_due_to[6])
			logger.info("Rejected due to Unidentified Language: %d", rejects_due_to[7])
			logger.info("Rejected due to NISQA Metrics: %d", rejects_due_to[8])
			logger.debug("Note that a sample can be rejected due to one or more reasons!")
			logger.info('='*85)

		return
	finally:
		ray.shutdown()
