from src import Logger, LANGUAGE_CODE_MAP
from src import LIDPredictor


def main():
    logger = Logger()
    logger.info(LANGUAGE_CODE_MAP["malayalam"]["iso2"])
    lid = LIDPredictor(device="cuda")

    batch_paths = [
        "audio1.flac",
        "audio2.flac",
        "audio3.flac",
    ]

    results = lid.predict_batch(
        batch_paths,
        expected_language="hin",   # example: Hindi
        min_keep_conf=0.8,
    )

    for r in results:
        print(r["audio_path"], r["predicted_language"], r["top_confidence"], r["keep"])

    return


if __name__ == "__main__":
    main()