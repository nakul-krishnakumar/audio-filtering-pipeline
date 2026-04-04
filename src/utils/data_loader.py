from torch.utils.data import IterableDataset, get_worker_info
import torch
import torchaudio
import soundfile as sf
import json
import os
from typing import Any
from pydantic import BaseModel

from .logger import Logger

class AudioSample(BaseModel):
	audio_filepath: str
	audio: Any
	sr: Any
	text: str
	duration: float
	lang: str

class StreamingAudioDataset(IterableDataset):
	def __init__(self, logger: Logger, manifest_path: str, target_sr: int = 16000):
		self.logger = logger
		self.manifest_path = manifest_path
		self.target_sr = target_sr
		self._manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

	def _resolve_audio_path(self, path: str) -> str:
		if os.path.isabs(path):
			return path
		# Resolve relative entries against current cwd first (common case), then
		# against manifest location for portability.
		cwd_candidate = os.path.abspath(path)
		if os.path.exists(cwd_candidate):
			return cwd_candidate
		return os.path.abspath(os.path.join(self._manifest_dir, path))

	def _line_iterator(self):
		worker_info = get_worker_info()

		if worker_info is None:
			worker_id = 0
			num_workers = 1
		else:
			worker_id = worker_info.id
			num_workers = worker_info.num_workers

		with open(self.manifest_path, "r") as f:
			for idx, line in enumerate(f):
				if idx % num_workers == worker_id:
					yield line
	
	def _convert_to_mono(self, waveform):
		"""
		Convert multi-channel audio to mono-channel
		"""
		return waveform.mean(dim=0, keepdim=True)

	def _load_waveform(self, path: str):
		audio, sr = sf.read(path, always_2d=True)
		waveform = torch.from_numpy(audio).float().transpose(0, 1)
		return waveform, sr

	def _resample(self, waveform, sr):
		"""
		Resample audio waveform to expected target sample rate
		"""
		self.logger.warn(f"Audio SR: {sr} | Expected SR: {self.target_sr} | Converting...")
		waveform = torchaudio.functional.resample(
						waveform, sr, self.target_sr
					)
		return waveform, self.target_sr

	def _generate_audio_sample(self, item, waveform, sr) -> AudioSample:
		audio_path = self._resolve_audio_path(item["audio_filepath"])
		return AudioSample(
			audio_filepath=audio_path,
			audio=waveform,
			sr=sr,
			text=item.get("text", ""),
			duration=item.get("duration", 0.0),
			lang=item.get("lang", ""),
		)

	def __iter__(self):
		for line in self._line_iterator():
			try:
				item = json.loads(line)
				audio_path = self._resolve_audio_path(item["audio_filepath"])
				waveform, sr = self._load_waveform(audio_path)
				item["audio_filepath"] = audio_path

				if waveform.shape[0] > 1:
					waveform = self._convert_to_mono(waveform)

				if sr != self.target_sr:
					waveform, sr = self._resample(waveform, sr)

				sample = self._generate_audio_sample(item, waveform, sr)
				yield sample

			except Exception as e:
				self.logger.warn(str(e))
				continue