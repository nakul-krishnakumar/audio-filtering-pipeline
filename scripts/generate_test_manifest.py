import json
import shutil
from pathlib import Path


def generate_manifest(
    root_dir: str,
    output_file: str,
    audio_output_dir: str = "./input/audios",
    max_samples_per_file: int = 500,
) -> None:
    root = Path(root_dir)
    output_path = Path(output_file)
    audio_output_path = Path(audio_output_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_output_path.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_moved = 0

    def _target_audio_path(source_audio: Path) -> Path:
        parts = source_audio.parts
        if "audios" in parts:
            audios_idx = parts.index("audios")
            relative_audio = Path(*parts[audios_idx + 1 :])
        else:
            relative_audio = Path(source_audio.name)
        return audio_output_path / relative_audio

    with output_path.open("w", encoding="utf-8") as out_f:
        for subdir in root.iterdir():
            if not subdir.is_dir():
                continue

            if not subdir.name.endswith("_manifests"):
                continue

            print(f"Processing: {subdir}")
            json_files = list(subdir.glob("*.json")) + list(subdir.glob("*.jsonl"))

            if not json_files:
                print(f"No JSON file found in {subdir}")
                continue

            json_file = json_files[0]
            with json_file.open("r", encoding="utf-8") as f:
                written_for_file = 0
                for i, line in enumerate(f):
                    if written_for_file >= max_samples_per_file:
                        break

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Skipping malformed JSON line in {json_file}: {i + 1}")
                        continue

                    audio_rel = entry.get("audio_filepath")
                    if not audio_rel:
                        print(f"Skipping entry without audio_filepath in {json_file}: {i + 1}")
                        continue

                    source_audio = Path(audio_rel)
                    if not source_audio.is_absolute():
                        source_audio = (Path.cwd() / source_audio).resolve()

                    target_audio = _target_audio_path(source_audio)
                    target_audio.parent.mkdir(parents=True, exist_ok=True)

                    if source_audio.exists() and not target_audio.exists():
                        shutil.move(str(source_audio), str(target_audio))
                        total_moved += 1
                    elif not source_audio.exists() and not target_audio.exists():
                        print(f"Audio missing, skipping entry: {audio_rel}")
                        continue

                    entry["audio_filepath"] = f"./{target_audio.as_posix().lstrip('./')}"
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_written += 1
                    written_for_file += 1

    print(f"\nDone. Total lines written: {total_written}")
    print(f"Audio files moved to {audio_output_path}: {total_moved}")


if __name__ == "__main__":
    generate_manifest(
        root_dir="./data/manifests",
        output_file="./input/test_manifest.jsonl",
        audio_output_dir="./input/audios",
        max_samples_per_file=10,
    )
