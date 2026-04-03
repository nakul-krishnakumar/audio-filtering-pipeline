from pathlib import Path


def generate_manifest(
    root_dir: str,
    output_file: str,
    max_lines_per_file: int = 500,
) -> None:
    root = Path(root_dir)
    output_path = Path(output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0

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
                for i, line in enumerate(f):
                    if i >= max_lines_per_file:
                        break
                    out_f.write(line)
                    total_written += 1

    print(f"\nDone. Total lines written: {total_written}")


if __name__ == "__main__":
    generate_manifest(
        root_dir="./data/manifests",
        output_file="./data/manifests/test_manifest.jsonl",
        max_lines_per_file=250,
    )
