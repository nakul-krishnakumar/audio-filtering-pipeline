from utils import StreamingAudioDataset, Logger
from


def run(
		manifest_path: str = "./data/manifests/combined_manifest.jsonl",
		target_sr: int = 160000,
		max_samples: int = 1000
	):

	logger = Logger()
	dataset = StreamingAudioDataset(
		logger=logger,
		manifest_path=manifest_path,
		target_sr=target_sr,
		max_samples=max_samples
	)

	loader = Dat

	return
