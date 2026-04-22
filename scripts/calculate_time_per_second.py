#!/usr/bin/env python3
"""
Calculate processing time per second of audio.

This is the inverse of RTF (Real-Time Factor):
  Time per second = Processing Time / Total Audio Duration
                  = 1 / RTF

Example: If RTF = 16.99, then time per second = 0.059 seconds (59ms)
         This means 1 second of audio takes 59ms to process.
"""

import json
import sys
from pathlib import Path


def calculate_time_per_second(
    manifest_path: str, processing_time_seconds: float
) -> dict:
    """
    Calculate processing time per second of audio.

    Args:
        manifest_path: Path to the filtered_manifest.jsonl file
        processing_time_seconds: Total wall clock time for processing

    Returns:
        Dictionary with timing metrics
    """
    total_duration = 0.0
    num_samples = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                duration = entry.get("duration", 0.0)
                total_duration += duration
                num_samples += 1
            except json.JSONDecodeError:
                continue

    # Calculate metrics
    time_per_second = (
        processing_time_seconds / total_duration if total_duration > 0 else 0
    )
    rtf = total_duration / processing_time_seconds if processing_time_seconds > 0 else 0

    # Calculate per-sample metrics
    time_per_sample = processing_time_seconds / num_samples if num_samples > 0 else 0
    samples_per_second = (
        num_samples / processing_time_seconds if processing_time_seconds > 0 else 0
    )

    return {
        "num_samples": num_samples,
        "total_audio_seconds": total_duration,
        "total_audio_minutes": total_duration / 60,
        "processing_time_seconds": processing_time_seconds,
        "time_per_audio_second": time_per_second,  # seconds of processing per 1 second of audio
        "time_per_audio_second_ms": time_per_second * 1000,  # in milliseconds
        "rtf": rtf,
        "time_per_sample": time_per_sample,
        "samples_per_second": samples_per_second,
    }


def print_timing_report(metrics: dict, model_name: str = ""):
    """Print a formatted timing report."""
    header = f"{'=' * 65}"
    title = f" Timing Report {model_name} ".center(65, "=")

    print(f"\n{header}")
    print(title)
    print(f"{header}")

    print(f"\n Sample Count: {metrics['num_samples']} samples")
    print(
        f" Total Audio:    {metrics['total_audio_seconds']:.2f} seconds "
        f"({metrics['total_audio_minutes']:.2f} minutes)"
    )
    print(f" Wall Time:      {metrics['processing_time_seconds']:.2f} seconds")

    print(f"\n{'=' * 65}")
    print(" KEY METRIC: Processing Time per 1 Second of Audio ".center(65, "="))
    print(f"{'=' * 65}")

    print(f"\n  {metrics['time_per_audio_second']:.4f} seconds")
    print(f"  {metrics['time_per_audio_second_ms']:.2f} milliseconds")
    print(f"\n  Interpretation:")
    print(
        f"  -> To process 1 second of audio, it takes {metrics['time_per_audio_second']:.4f}s"
    )
    print(
        f"  -> To process 1 minute of audio, it takes {metrics['time_per_audio_second'] * 60:.2f}s"
    )
    print(
        f"  -> To process 1 hour of audio, it takes {metrics['time_per_audio_second'] * 3600 / 60:.2f} minutes"
    )

    print(f"\n Additional Metrics:")
    print(f"   RTF (Real-Time Factor):     {metrics['rtf']:.2f}x")
    print(f"   Time per sample:            {metrics['time_per_sample']:.3f}s")
    print(
        f"   Throughput:                 {metrics['samples_per_second']:.2f} samples/second"
    )

    print(f"\n{header}\n")


def main():
    manifest_path = Path("output/filtered_manifest.jsonl")

    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(" PROCESSING TIME PER SECOND OF AUDIO ".center(65, "="))
    print("=" * 65)

    # Calculate for Whisper Tiny (60 seconds processing time)
    metrics_tiny = calculate_time_per_second(
        str(manifest_path), processing_time_seconds=60
    )
    print_timing_report(metrics_tiny, model_name="[Whisper Tiny - 60s]")

    # Calculate for Whisper Medium (80 seconds processing time)
    metrics_medium = calculate_time_per_second(
        str(manifest_path), processing_time_seconds=80
    )
    print_timing_report(metrics_medium, model_name="[Whisper Medium - 80s]")

    # Summary comparison
    print("\n" + "=" * 65)
    print(" COMPARISON SUMMARY ".center(65, "="))
    print("=" * 65)

    print(f"\n Processing 1 Second of Audio:")
    print(f"   Whisper Tiny:   {metrics_tiny['time_per_audio_second_ms']:.2f} ms")
    print(f"   Whisper Medium: {metrics_medium['time_per_audio_second_ms']:.2f} ms")

    print(f"\n Processing 1 Minute of Audio:")
    print(
        f"   Whisper Tiny:   {metrics_tiny['time_per_audio_second'] * 60:.2f} seconds"
    )
    print(
        f"   Whisper Medium: {metrics_medium['time_per_audio_second'] * 60:.2f} seconds"
    )

    speed_diff = (
        (
            metrics_tiny["time_per_audio_second"]
            / metrics_medium["time_per_audio_second"]
        )
        - 1
    ) * 100
    if speed_diff < 0:
        print(f"\n Tiny is {abs(speed_diff):.1f}% faster than Medium")
    else:
        print(f"\n Medium is {abs(speed_diff):.1f}% faster than Tiny")

    print(f"\n{'=' * 65}\n")


if __name__ == "__main__":
    main()
