from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import os

import numpy as np
import ray
import torch
import whisper
from pyannote.audio import Inference, Model

from ..utils.logger import Logger
from ..utils.data_loader import AudioSample

from enum import Enum

class status(Enum):
	REJECT = 1
	ACCEPT = 2


def _runtime_identity() -> dict[str, str | int]:
	ctx = ray.get_runtime_context()
	node_id = str(ctx.get_node_id()) if hasattr(ctx, "get_node_id") else "unknown"
	task_id = str(ctx.get_task_id()) if hasattr(ctx, "get_task_id") else "unknown"
	actor_id = str(ctx.get_actor_id()) if hasattr(ctx, "get_actor_id") else "none"
	return {
		"pid": os.getpid(),
		"node_id": node_id,
		"task_id": task_id,
		"actor_id": actor_id,
	}

@dataclass(frozen=True)
class MetricConfig:
	clip_threshold: float = 0.98
	silence_threshold_db: float = -40.0
	frame_length: int = 1024
	vad_threshold: float = 0.5
	max_snr_db: float = 60.0

class AudioFilterer:
	def __init__(
		self,
		logger: Logger,
		hf_token: Optional[str] = None,
		device: Optional[str | torch.device] = None,
		config: MetricConfig = MetricConfig(),
		use_hard_filters: bool = False,
	):
		self.logger = logger
		self.config = config
		if device is None:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = device if isinstance(device, torch.device) else torch.device(device)

		self.brouhaha_inference: Optional[Inference] = None
		if use_hard_filters:
			if hf_token:
				model = Model.from_pretrained(
					"pyannote/brouhaha",
					use_auth_token=hf_token,
				)
				self.brouhaha_inference = Inference(model, device=self.device)
			
			self.asr_model = whisper.load_model("tiny")

	def calc_norm(self, waveform: torch.Tensor) -> torch.Tensor:
		return waveform / (waveform.abs().max() + 1e-10)

	def calc_clipping_ratio(self, waveform: torch.Tensor) -> float:
		clipped = torch.abs(waveform) >= self.config.clip_threshold
		return clipped.float().mean().item()
	
	def calc_silence_ratio(self, norm_waveform: torch.Tensor) -> float:
		# Flatten first so framing logic works for both [T] and [C, T] tensors.
		flat_waveform = norm_waveform.reshape(-1)
		num_frames = flat_waveform.numel() // self.config.frame_length
		if num_frames == 0:
			return 1.0
		else:
			frames = flat_waveform[: num_frames * self.config.frame_length].reshape(
				num_frames, self.config.frame_length
			)
			frame_power = (frames ** 2).mean(dim=1)
			threshold = 10 ** (self.config.silence_threshold_db / 10.0)
			return (frame_power < threshold).float().mean().item()
	
	def calc_asr_confidence(self, audio_path) -> float:
		result = self.asr_model.transcribe(audio_path, temperature=0)

		tokens = result["segments"]
		if len(tokens) == 0:
			return 0.0 

		log_probs = []
		for seg in tokens:
			if "avg_logprob" in seg:
				log_probs.append(seg["avg_logprob"])

		prob = np.exp(log_probs)

		return float(np.mean(prob))

	def compute_soft(self, sample: AudioSample) -> dict[str, Any]:
		waveform = sample.audio
		duration = sample.duration
		norm_waveform = self.calc_norm(waveform)
		clipping_ratio = self.calc_clipping_ratio(waveform)
		silence_ratio = self.calc_silence_ratio(norm_waveform)

		return {
			"audio_filepath": sample.audio_filepath,
			"duration": duration,
			"clipping_ratio": clipping_ratio,
			"silence_ratio": silence_ratio,
		}

	def compute_hard(self, sample: AudioSample) -> dict[str, Any]:
		if self.brouhaha_inference is None:
			return {"vad_ratio": 0.0, "snr": 0.0, "c50": 0.0}

		output = self.brouhaha_inference(sample.audio_filepath)
		asr_conf = self.calc_asr_confidence(sample.audio_filepath)

		vad_vals, snr_vals, c50_vals = [], [], []

		for _, (vad, snr, c50) in output:
			vad_vals.append(vad)
			snr_vals.append(snr)
			c50_vals.append(c50)

		if not vad_vals:
			return {"vad_ratio": 0.0, "snr": 0.0, "c50": 0.0, "asr": asr_conf}

		return {
			"vad_ratio": float((np.array(vad_vals) > self.config.vad_threshold).mean()),
			"snr": float(np.median(snr_vals)),
			"c50": float(np.median(c50_vals)),
			"asr": asr_conf
		}
	
@ray.remote(num_cpus=1)
def soft_filter_task(sample: AudioSample):
	logger = Logger("soft_filter")
	filterer = AudioFilterer(logger=logger, hf_token=None)
	identity = _runtime_identity()
	# Note that we initialised AudioFilterer object without brouhaha model, so
	# no need to use ray actors

	result = filterer.compute_soft(sample)
	result["status"] = status.ACCEPT.name

	# TODO
	if result["duration"] < 1.0:
		result["status"] = status.REJECT.name
	if result["silence_ratio"] > 0.6:
		result["status"] = status.REJECT.name
	if result["clipping_ratio"] > 0.2:
		result["status"] = status.REJECT.name

	if result["status"] == status.REJECT.name:
		return None

	return {"sample": sample, "soft_metrics": result, "_ray": identity}

"""
Why ray actor for HardFilters but not for SoftFilters?
- For HardFilters, we need stateful workers so as to retain the same model object across workers
- It is not possible if we do not use Ray Actor
- So SoftFilters is just a simple Ray Task
"""

@ray.remote(num_gpus=1)
class HardFilterActor:
	def __init__(self, hf_token: Optional[str] = None):
		logger = Logger("hard_filter")
		self.filterer = AudioFilterer(logger=logger, hf_token=hf_token, use_hard_filters=True)
		self.identity = _runtime_identity()

	def get_identity(self):
		return self.identity

	def process_batch(self, batch: list[AudioSample]):
		results = []

		for sample in batch:
			hard_metrics = self.filterer.compute_hard(sample)

			results.append({
				"audio_filepath": sample.audio_filepath,
				**hard_metrics
			})

		return results