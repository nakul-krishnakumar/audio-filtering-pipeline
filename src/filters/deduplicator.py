from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

import librosa
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class FingerprintConfig:
	sr: int = 16000
	n_mels: int = 64
	hop_length: int = 512
	n_fft: int = 2048
	top_k: int = 8
	shingle_size: int = 3


def _extract_fingerprint(path: str, cfg: FingerprintConfig) -> np.ndarray:
	y, _ = librosa.load(path, sr=cfg.sr, mono=True)

	spec = librosa.feature.melspectrogram(
		y=y,
		sr=cfg.sr,
		n_mels=cfg.n_mels,
		n_fft=cfg.n_fft,
		hop_length=cfg.hop_length,
	)
	log_spec = librosa.power_to_db(spec, ref=np.max)

	# Frame-wise normalization keeps each time step comparable across loudness changes.
	denom = log_spec.std(axis=0, keepdims=True) + 1e-6
	return (log_spec - log_spec.mean(axis=0, keepdims=True)) / denom


def _fingerprint_to_tokens(log_spec: np.ndarray, top_k: int) -> list[tuple[int, ...]]:
	tokens: list[tuple[int, ...]] = []
	if log_spec.shape[0] < top_k:
		top_k = log_spec.shape[0]

	for frame in log_spec.T:
		idx = np.argpartition(frame, -top_k)[-top_k:]
		tokens.append(tuple(sorted(int(i) for i in idx)))
	return tokens


def _build_shingles(tokens: list[tuple[int, ...]], k: int) -> set[tuple[tuple[int, ...], ...]]:
	shingles: set[tuple[tuple[int, ...], ...]] = set()
	if k <= 0 or len(tokens) < k:
		return shingles

	for i in range(len(tokens) - k + 1):
		shingles.add(tuple(tokens[i : i + k]))
	return shingles


def _jaccard_similarity(
	a: set[tuple[tuple[int, ...], ...]],
	b: set[tuple[tuple[int, ...], ...]],
) -> float:
	if not a or not b:
		return 0.0
	return len(a & b) / len(a | b)


def _process_file(path: str, cfg: FingerprintConfig) -> tuple[str, set[tuple[tuple[int, ...], ...]] | None]:
	try:
		log_spec = _extract_fingerprint(path, cfg)
		tokens = _fingerprint_to_tokens(log_spec, top_k=cfg.top_k)
		shingles = _build_shingles(tokens, k=cfg.shingle_size)
		return path, shingles
	except Exception:
		return path, None


class AudioDeduplicator:
	"""
	Audio Deduplicator
	"""
	def __init__(
		self,
		*,
		threshold: float = 0.88,
		max_workers: int = 8,
		config: FingerprintConfig | None = None,
	) -> None:
		self.threshold = threshold
		self.max_workers = max_workers
		self.config = config or FingerprintConfig()

	def build_fingerprint_index(
		self,
		paths: Iterable[str],
		show_progress: bool = True,
	) -> dict[str, set[tuple[tuple[int, ...], ...]]]:
		paths_list = list(paths)
		fingerprints: dict[str, set[tuple[tuple[int, ...], ...]]] = {}

		with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
			futures = [executor.submit(_process_file, p, self.config) for p in paths_list]

			iterator = as_completed(futures)
			if show_progress:
				iterator = tqdm(iterator, total=len(paths_list), desc="Fingerprinting")

			for future in iterator:
				path, shingles = future.result()
				if shingles is not None:
					fingerprints[path] = shingles

		return fingerprints

	def deduplicate(
		self,
		paths: Iterable[str],
		*,
		threshold: float | None = None,
		show_progress: bool = True,
	) -> tuple[list[str], set[str]]:
		active_threshold = threshold if threshold is not None else self.threshold
		fingerprints = self.build_fingerprint_index(paths, show_progress=show_progress)

		keep: list[str] = []
		removed: set[str] = set()
		indexed_paths = list(fingerprints.keys())

		iterator = enumerate(indexed_paths)
		if show_progress:
			iterator = enumerate(tqdm(indexed_paths, desc="Deduplicating"))

		for i, p1 in iterator:
			if p1 in removed:
				continue

			keep.append(p1)
			fp1 = fingerprints[p1]

			print(f"p1: {p1} | fingerprint: {fp1}\n")

			for p2 in indexed_paths[i + 1 :]:
				if p2 in removed:
					continue

				fp2 = fingerprints[p2]

				print(f"p2: {p1} | fingerprint: {fp2}\n")

				similarity = _jaccard_similarity(fp1, fp2)
				if similarity >= active_threshold:
					removed.add(p2)
				
				print(f"Similarity: {similarity}\n")

		return keep, removed
