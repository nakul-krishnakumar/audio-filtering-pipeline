from src.pipeline import run_pipeline


def main():
    run_pipeline(
        manifest_path = "./data/manifests/test_manifest.jsonl",
		target_sr = 16000,
        batch_size = 32,
        num_data_loader_workers = 8,
	)

    return


if __name__ == "__main__":
    main()