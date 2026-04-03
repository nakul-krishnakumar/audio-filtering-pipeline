from ..utils import StreamingAudioDataset, Logger
from torch.utils.data import DataLoader

def collate_fn(batch):
    return batch

def run_pipeline(
		manifest_path: str = "./data/manifests/test_manifest.jsonl",
		target_sr: int = 160000,
        batch_size: int = 8,
        num_data_loader_workers: int = 4,
	):

	logger = Logger()
	dataset = StreamingAudioDataset(
		logger=logger,
		manifest_path=manifest_path,
		target_sr=target_sr,
	)

	dataIterator = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_data_loader_workers,
            collate_fn=collate_fn,
	)
      
	for batch in dataIterator:
		logger.info(f"{type(batch)} | size: {len(batch)} : sample | {batch[0].audio}")

	return
