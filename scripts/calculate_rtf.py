#!/usr/bin/env python3
"""
Calculate Real-Time Factor (RTF) for the audio filtering pipeline.

RTF = Total Audio Duration / Processing Time

An RTF of 1.0 means processing takes exactly real-time.
An RTF > 1.0 means processing faster than real-time (good).
An RTF < 1.0 means processing slower than real-time.

Example: RTF = 2.0 means processing 2 seconds of audio in 1 second of wall clock time.
"""

import json
import sys
from pathlib import Path


def calculate_rtf(manifest_path: str, processing_time_seconds: float) -> dict:
    """
    Calculate RTF from a filtered manifest JSONL file.

    Args:
        manifest_path: Path to the filtered_manifest.jsonl file
        processing_time_seconds: Total wall clock time for processing

    Returns:
        Dictionary with RTF metrics
    """
    total_duration = 0.0
    num_samples = 0
    accepted_duration = 0.0
    rejected_duration = 0.0
    num_accepted = 0
    num_rejected = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                duration = entry.get("duration", 0.0)
                status = entry.get("status", "Unknown")

                total_duration += duration
                num_samples += 1

                if status == "Accept":
                    accepted_duration += duration
                    num_accepted += 1
                else:
                    rejected_duration += duration
                    num_rejected += 1

            except json.JSONDecodeError:
                continue

    # Calculate RTF
    rtf_overall = (
        total_duration / processing_time_seconds if processing_time_seconds > 0 else 0
    )
    rtf_accepted = (
        accepted_duration / processing_time_seconds
        if processing_time_seconds > 0
        else 0
    )

    # Calculate throughput (audio seconds per wall clock second)
    throughput = (
        total_duration / processing_time_seconds if processing_time_seconds > 0 else 0
    )

    return {
        "num_samples": num_samples,
        "num_accepted": num_accepted,
        "num_rejected": num_rejected,
        "total_duration_seconds": total_duration,
        "total_duration_minutes": total_duration / 60,
        "total_duration_hours": total_duration / 3600,
        "accepted_duration_seconds": accepted_duration,
        "rejected_duration_seconds": rejected_duration,
        "processing_time_seconds": processing_time_seconds,
        "processing_time_minutes": processing_time_seconds / 60,
        "rtf_overall": rtf_overall,
        "rtf_accepted": rtf_accepted,
        "throughput_audio_seconds_per_wall_second": throughput,
    }


def print_rtf_report(metrics: dict, model_name: str = ""):
    """Print a formatted RTF report."""
    header = f"{'=' * 60}"
    title = f" RTF Report {model_name} ".center(60, "=")

    print(f"\n{header}")
    print(title)
    print(f"{header}")

    print(f"\n Sample Statistics:")
    print(f"   Total Samples Processed:    {metrics['num_samples']}")
    print(
        f"   Accepted:                   {metrics['num_accepted']} ({metrics['num_accepted'] / metrics['num_samples'] * 100:.1f}%)"
    )
    print(
        f"   Rejected:                   {metrics['num_rejected']} ({metrics['num_rejected'] / metrics['num_samples'] * 100:.1f}%)"
    )

    print(f"\n Duration Statistics:")
    print(
        f"   Total Audio Duration:       {metrics['total_duration_seconds']:.2f} seconds"
    )
    print(
        f"                               {metrics['total_duration_minutes']:.2f} minutes"
    )
    print(f"                               {metrics['total_duration_hours']:.4f} hours")
    print(
        f"   Accepted Duration:          {metrics['accepted_duration_seconds']:.2f} seconds"
    )
    print(
        f"   Rejected Duration:          {metrics['rejected_duration_seconds']:.2f} seconds"
    )

    print(f"\n Processing Time:")
    print(
        f"   Wall Clock Time:            {metrics['processing_time_seconds']:.2f} seconds"
    )
    print(
        f"                               {metrics['processing_time_minutes']:.2f} minutes"
    )

    print(f"\n Real-Time Factor (RTF):")
    print(f"   RTF (Overall):              {metrics['rtf_overall']:.4f}")
    print(f"   RTF (Accepted only):        {metrics['rtf_accepted']:.4f}")
    print(f"\n   Interpretation:")
    print(f"   - Processed {metrics['rtf_overall']:.2f}x real-time")
    print(
        f"   - Every 1 second of wall clock time processes {metrics['rtf_overall']:.2f} seconds of audio"
    )

    if metrics["rtf_overall"] > 1.0:
        speedup = metrics["rtf_overall"]
        print(f"\n   Processing is {speedup:.2f}x faster than real-time")
    else:
        slowdown = (
            1.0 / metrics["rtf_overall"] if metrics["rtf_overall"] > 0 else float("inf")
        )
        print(f"\n   Processing is {slowdown:.2f}x slower than real-time")

    print(f"\n{header}\n")


def main():
    manifest_path = Path("output/filtered_manifest.jsonl")

    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}")
        print("Please ensure the pipeline has been run and output exists.")
        sys.exit(1)

    # Calculate RTF for Whisper Tiny (60 seconds processing time)
    print("\n" + "=" * 60)
    print(" CALCULATING RTF ".center(60, "="))
    print("=" * 60)

    # Whisper Tiny - 60 seconds
    metrics_tiny = calculate_rtf(str(manifest_path), processing_time_seconds=60)
    print_rtf_report(metrics_tiny, model_name="[Whisper Tiny - 60s processing]")

    # Whisper Medium - 80 seconds
    metrics_medium = calculate_rtf(str(manifest_path), processing_time_seconds=80)
    print_rtf_report(metrics_medium, model_name="[Whisper Medium - 80s processing]")

    # Summary comparison
    print("\n" + "=" * 60)
    print(" COMPARISON SUMMARY ".center(60, "="))
    print("=" * 60)
    print(f"\nWhisper Tiny (60s):")
    print(
        f"   RTF: {metrics_tiny['rtf_overall']:.4f} | Throughput: {metrics_tiny['total_duration_minutes']:.2f} min audio / min wall time"
    )
    print(f"\nWhisper Medium (80s):")
    print(
        f"   RTF: {metrics_medium['rtf_overall']:.4f} | Throughput: {metrics_medium['total_duration_minutes']:.2f} min audio / min wall time"
    )

    speed_diff = (metrics_tiny["rtf_overall"] / metrics_medium["rtf_overall"] - 1) * 100
    if speed_diff > 0:
        print(f"\nTiny is {abs(speed_diff):.1f}% faster than Medium in terms of RTF")
    else:
        print(f"\nMedium is {abs(speed_diff):.1f}% faster than Tiny in terms of RTF")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
