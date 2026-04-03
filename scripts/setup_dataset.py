#!/usr/bin/env python3
"""
IndicVoices Dataset Processing Script

This script downloads and processes the IndicVoices dataset from HuggingFace,
extracting audio files and generating manifest files for TTS training.

Created for Sarvam TTS Team Hiring Assessment

Usage:
    python setup_dataset.py <save_directory>

Requirements:
    - polars
    - huggingface_hub

Important:
    This dataset is gated. You must:
    1. Request access at https://huggingface.co/datasets/ai4bharat/IndicVoices
    2. Run 'huggingface-cli login' before executing this script

Note: 
    This script will download the dataset to the specified save directory.
    It will take ~14GB of disk space.
    Final file structure will be:
    <save_directory>
    ├── hf
    ├── audios
    ├── manifests
    └── setup.log

    The hf directory will contain the downloaded dataset from HuggingFace.
    The audios directory will contain the extracted audio files.
    The manifests directory will contain the generated manifest files.
    The setup.log will contain the logs of the script.
"""

import os
import logging
import argparse
import polars as pl

from tqdm import tqdm
from huggingface_hub import snapshot_download
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


n_procs = max(1, os.cpu_count() - 1)
n_cpus = os.cpu_count()
n_threads = max(1, n_cpus // n_procs)


def setup_logger(log_file=None, log_level=logging.INFO):
    """Setup logger with optional file output."""
    logger = logging.getLogger("IndicvoicesSetup")
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}\n")

    return logger





def process_row(row, dest_dir):
    try:
        audio_data = row['audio_filepath']['bytes']
        audio_name = row['audio_filepath']['path']

        del row['audio_filepath']

        save_path = os.path.join(dest_dir, audio_name)
        with open(save_path, "wb") as f:
            f.write(audio_data)

        entry = {
            'audio_filepath': save_path,
            **row
        }

        return True, entry
    except Exception as e:
        message = f"Error processing sample: {e}"
        return False, message



def process_parquet(parquet_file, lang, audio_save_dir, manifest_save_dir, logger):
    index = os.path.basename(parquet_file).split('.')[0]
    dest_dir = os.path.join(audio_save_dir, lang, index)
    os.makedirs(dest_dir, exist_ok=True)

    def log(message):
        logger.info(f"{lang} - {os.path.basename(parquet_file)} - {message}")
    def log_error(message):
        logger.error(f"{lang} - {os.path.basename(parquet_file)} - {message}")

    log(f"Processing...")
    df = pl.read_parquet(parquet_file)

    manifest = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [
            executor.submit(process_row, row, dest_dir)
            for row in df.iter_rows(named=True)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {lang}/{os.path.basename(parquet_file)} results"):
            result = future.result()
            if result[0]:
                manifest.append(result[1])
            else:
                log_error(result[1])
    
    log(f"Processed {len(manifest)} rows!")

    manifest_df = pl.DataFrame(manifest)
    save_path = os.path.join(manifest_save_dir, f'{lang}_manifests', f"{index}.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    manifest_df.write_ndjson(save_path)

    log(f"Saved manifest to {save_path}")

    return manifest


def main(save_dir: str):
    HF_SAVE_DIR = os.path.join(save_dir, 'hf')
    AUDIO_SAVE_DIR = os.path.join(save_dir, 'audios')
    MANIFEST_SAVE_DIR = os.path.join(save_dir, 'manifests')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(HF_SAVE_DIR, exist_ok=True)
    os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)
    os.makedirs(MANIFEST_SAVE_DIR, exist_ok=True)

    log_file = os.path.join(save_dir, 'setup.log')
    if os.path.exists(log_file):
        os.remove(log_file)
    logger = setup_logger(log_file=log_file)
    logger.info(f"Created directories: {save_dir}, {HF_SAVE_DIR}, {AUDIO_SAVE_DIR}, {MANIFEST_SAVE_DIR}\n")

    logger.info(f"Downloading IndicVoices valid set from Hugging Face")
    local_path = snapshot_download(
        repo_id="ai4bharat/IndicVoices",
        repo_type="dataset",
        local_dir=HF_SAVE_DIR,
        local_dir_use_symlinks=False,
        max_workers=32,
        resume_download=True,
        allow_patterns=["*/valid*.parquet"] # Removed downloading train parquet
        # It is mentioned in the doctest above that it will only download ~14gb of data,
        # but instead it downloaded more than 50gb, so I removed the train parquets
    )
    logger.info(f"Downloaded to {HF_SAVE_DIR}\n")

    logger.info(f"Processing IndicVoices valid set")
    manifest = []
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        for lang in os.listdir(HF_SAVE_DIR):
            # look for "*.parquet files within lang dir"
            lang_dir = os.path.join(HF_SAVE_DIR, lang)
            if not os.path.isdir(lang_dir): continue

            parquet_files = [
                os.path.join(lang_dir, f)
                for f in os.listdir(lang_dir)
                if f.endswith(".parquet")
            ]
            if not parquet_files: continue

            logger.info(f"Processing {lang} with {len(parquet_files)} parquet files")
            futures = [
                executor.submit(process_parquet, parquet_file, lang, AUDIO_SAVE_DIR, MANIFEST_SAVE_DIR, logger)
                for parquet_file in parquet_files
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing results"):
                manifest.extend(future.result())

    manifest_df = pl.DataFrame(manifest)
    save_path = os.path.join(MANIFEST_SAVE_DIR, f"combined_manifest.jsonl")
    manifest_df.write_ndjson(save_path)

    logger.info(f"Saved combined manifest to {save_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', type=str)
    args = parser.parse_args()

    main(save_dir=args.save_dir)