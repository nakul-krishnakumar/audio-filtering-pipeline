from torch.utils.data import IterableDataset, get_worker_info
import torchaudio
import json
from pydantic import BaseModel

from .logger import Logger

class VerificationReport(BaseModel):
	decision: str
	low_volume: bool
	noise_intermittent: bool
	chatter_intermittent: bool
	noise_persistent: bool
	chatter_persistent: bool
	unclear_audio: bool
	off_topi: bool
	repeating_content: bool
	long_pause: bool
	mispronunciation: bool
	reading_promp: bool
	book_read: bool
	sst: bool
	stretchin: bool
	bad_extempore_quality: bool
	comments: str
	objectionable_content: bool
	skipping_word: bool
	incorrect_text_prompt: bool
	factual_inaccuracy: bool

class AudioSample(BaseModel):
	audio_filepath: str
	audio: any
	sr: any
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
	def __init__(self, logger: Logger, manifest_path: str, target_sr: int = 16000, max_samples: int = None):
		self.logger = logger
		self.manifest_path = manifest_path
		self.target_sr = target_sr
		self.max_samples = max_samples

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

	def _resample(self, waveform, sr):
		"""
		Resample audio waveform to expected target sample rate
		"""
		waveform = torchaudio.functional.resample(
						waveform, sr, self.target_sr
					)
		return waveform, self.target_sr

	def _generate_audio_sample(self, item, waveform, sr) -> AudioSample:
		vr = item.get("verification_report", {})
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


	def __iter__(self):
		count = 0

		for line in self._line_iterator():
			if self.max_samples is not None and count >= self.max_samples:
				break
			try:
				item = json.loads(line)
				waveform, sr = torchaudio.load(item["audio_filepath"])

				if waveform.shape[0] > 1:
					waveform = self._convert_to_mono(waveform)

				if sr != self.target_sr:
					waveform, sr = self._resample(waveform, sr)

				sample = self._generate_audio_sample(item, waveform, sr)
				yield sample

				count += 1

			except Exception as e:
				self.logger.warn(e)				
				continue