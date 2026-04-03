from torch.utils.data import IterableDataset, get_worker_info
import ast
import torch
import torchaudio
import soundfile as sf
import json
from typing import Any
from pydantic import BaseModel

from .logger import Logger

class VerificationReport(BaseModel):
	decision: str = ""
	low_volume: bool = False
	noise_intermittent: bool = False
	chatter_intermittent: bool = False
	noise_persistent: bool = False
	chatter_persistent: bool = False
	unclear_audio: bool = False
	off_topic: bool = False
	repeating_content: bool = False
	long_pauses: bool = False
	mispronunciation: bool = False
	reading_prompt: bool = False
	book_read: bool = False
	sst: bool = False
	stretching: bool = False
	bad_extempore_quality: bool = False
	comments: str = ""
	objectionable_content: bool = False
	skipping_words: bool = False
	incorrect_text_prompt: bool = False
	factual_inaccuracy: bool = False

class AudioSample(BaseModel):
	audio_filepath: str
	audio: Any
	sr: Any
	text: str
	duration: float
	lang: str
	samples: int
	verbatim: str
	normalized: str
	speaker_id: str
	scenario: str
	task_name: str
	gender: str
	age_group: str
	job_type: str
	qualification: str
	area: str
	district: str
	state: str
	occupation: str
	verification_report: VerificationReport
	unsanitized_verbatim: str
	unsanitized_normalized: str

class StreamingAudioDataset(IterableDataset):
	def __init__(self, logger: Logger, manifest_path: str, target_sr: int = 16000):
		self.logger = logger
		self.manifest_path = manifest_path
		self.target_sr = target_sr

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
		vr = self._parse_verification_report(item.get("verification_report", {}))
		verification_report = VerificationReport(**vr)

		return AudioSample(
			audio_filepath=item["audio_filepath"],
			audio=waveform,
			sr=sr,
			text=item.get("text", ""),
			duration=item.get("duration", 0.0),
			lang=item.get("lang", ""),
			samples=waveform.shape[-1],
			verbatim=item.get("verbatim", ""),
			normalized=item.get("normalized", ""),
			speaker_id=item.get("speaker_id", ""),
			scenario=item.get("scenario", ""),
			task_name=item.get("task_name", ""),
			gender=item.get("gender", ""),
			age_group=item.get("age_group", ""),
			job_type=item.get("job_type", ""),
			qualification=item.get("qualification", ""),
			area=item.get("area", ""),
			district=item.get("district", ""),
			state=item.get("state", ""),
			occupation=item.get("occupation", ""),
			verification_report=verification_report,
			unsanitized_verbatim=item.get("unsanitized_verbatim", ""),
			unsanitized_normalized=item.get("unsanitized_normalized", ""),
		)

	def _parse_verification_report(self, raw_vr: Any) -> dict[str, Any]:
		"""Parse verification_report from dict or serialized string formats."""
		if isinstance(raw_vr, dict):
			return raw_vr

		if isinstance(raw_vr, str):
			text = raw_vr.strip()
			if not text:
				return {}

			try:
				parsed = json.loads(text)
				if isinstance(parsed, dict):
					return parsed
			except Exception:
				pass

			# Dataset stores python-dict-like strings with single quotes.
			try:
				parsed = ast.literal_eval(text)
				if isinstance(parsed, dict):
					return parsed
			except Exception:
				self.logger.warn(f"Could not parse verification_report: {text[:120]}...")

		return {}

	def __iter__(self):
		for line in self._line_iterator():
			try:
				item = json.loads(line)
				waveform, sr = self._load_waveform(item["audio_filepath"])

				if waveform.shape[0] > 1:
					waveform = self._convert_to_mono(waveform)

				if sr != self.target_sr:
					waveform, sr = self._resample(waveform, sr)

				sample = self._generate_audio_sample(item, waveform, sr)
				yield sample

			except Exception as e:
				self.logger.warn(str(e))
				continue