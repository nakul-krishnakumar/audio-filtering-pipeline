from src.pipeline.run import run_pipeline


def main():
    run_pipeline(
        manifest_path = "./data/manifests/test_manifest.jsonl",
        output_dir= "./output",
		target_sr = 16000,
        batch_size = 5,
        num_data_loader_workers = 8,
        debug_workers=True
	)

    return


if __name__ == "__main__":
    main()