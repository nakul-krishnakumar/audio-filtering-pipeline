import io
import json
import os

import soundfile as sf
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "output", "filtered_manifest.jsonl")
THRESHOLDS_PATH = os.path.join(BASE_DIR, "thresholds.json")
PER_PAGE = 20

THRESHOLDS = {}
MARGINS = {
    "duration": 0.5,
    "c50": 3.0,
    "snr": 3.0,
    "silence_ratio": 0.05,
    "vad_ratio": 0.05,
    "asr": 0.05,
    "clipping_ratio": 0.02,
    "mos": 0.2,
    "noisiness": 0.2,
    "discontinuity": 0.2,
    "coloration": 0.2,
    "loudness": 0.2,
}

FILTER_MAP = {
    "duration_min": ("min_duration", "duration"),
    "duration_max": ("max_duration", "duration"),
    "c50": ("min_c50", "c50"),
    "snr": ("min_snr", "snr"),
    "silence_ratio": ("max_silence_ratio", "silence_ratio"),
    "vad_ratio": ("min_vad_ratio", "vad_ratio"),
    "asr": ("min_asr_conf", "asr"),
    "clipping_ratio": ("max_clipping_ratio", "clipping_ratio"),
    "mos": ("min_mos", "mos"),
    "noisiness": ("min_noisiness", "noisiness"),
    "discontinuity": ("min_discontinuity", "discontinuity"),
    "coloration": ("min_coloration", "coloration"),
    "loudness": ("min_loudness", "loudness"),
}

SORTABLE_COLUMNS = [
    "duration",
    "asr",
    "mos",
    "snr",
    "c50",
    "vad_ratio",
    "silence_ratio",
    "clipping_ratio",
    "noisiness",
    "discontinuity",
    "coloration",
    "loudness",
]


def load_thresholds():
    global THRESHOLDS
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH, "r") as f:
            THRESHOLDS = json.load(f)


def load_manifest():
    samples = []
    with open(MANIFEST_PATH, "r") as f:
        for idx, line in enumerate(f):
            if line.strip():
                sample = json.loads(line)
                sample["_idx"] = idx
                samples.append(sample)
    return samples


def save_manifest(samples):
    samples_sorted = sorted(samples, key=lambda x: x["_idx"])
    with open(MANIFEST_PATH, "w") as f:
        for sample in samples_sorted:
            sample_copy = {k: v for k, v in sample.items() if k != "_idx"}
            f.write(json.dumps(sample_copy) + "\n")


def filter_by_boundary(samples, filter_name):
    """
    	Filter samples near the threshold boundary for a given metric.
    """
    if filter_name not in FILTER_MAP:
        return samples

    threshold_key, field = FILTER_MAP[filter_name]
    if threshold_key not in THRESHOLDS:
        return samples

    threshold = THRESHOLDS[threshold_key]
    margin_key = field.replace("_ratio", "_ratio") if "ratio" in field else field
    margin = MARGINS.get(margin_key, MARGINS.get(field, 0.2))

    lower = threshold - margin
    upper = threshold + margin

    return [s for s in samples if lower <= s.get(field, 0) <= upper]


def sort_samples(samples, sort_by, order):
    """
    	Sort samples by a given column.
    """
    if sort_by not in SORTABLE_COLUMNS:
        return samples

    reverse = order == "desc"
    return sorted(samples, key=lambda x: x.get(sort_by, 0), reverse=reverse)


@app.route("/")
def index():
    samples = load_manifest()

    page = request.args.get("page", 1, type=int)
    sort_by = request.args.get("sort", "")
    order = request.args.get("order", "asc")
    filter_name = request.args.get("filter", "")

    if filter_name:
        samples = filter_by_boundary(samples, filter_name)

    if sort_by:
        samples = sort_samples(samples, sort_by, order)

    # Pagination
    total_samples = len(samples)
    total_pages = max(1, (total_samples + PER_PAGE - 1) // PER_PAGE)
    page = min(page, total_pages)

    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    page_samples = samples[start:end]

    for sample in page_samples:
        sample["_filename"] = os.path.basename(sample["audio_filepath"])

    return render_template(
        "index.html",
        samples=page_samples,
        page=page,
        total_pages=total_pages,
        total_samples=total_samples,
        sort_by=sort_by,
        order=order,
        filter_name=filter_name,
        sortable_columns=SORTABLE_COLUMNS,
        filters=list(FILTER_MAP.keys()),
    )


@app.route("/audio/<int:idx>")
def serve_audio(idx):
    samples = load_manifest()
    if not (0 <= idx < len(samples)):
        return "Audio not found", 404

    filepath = samples[idx]["audio_filepath"]
    if not os.path.exists(filepath):
        return "Audio not found", 404

    data, samplerate = sf.read(filepath)
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, data, samplerate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)

    # Why convert FLAC to WAV?
    # - Because I noticed a significant degradation in audio quality in the web audio player
    # - Might be due to low support for FLAC codec in the web player
    # - WAV plays really well (atleast for me)

    return send_file(
        wav_buffer,
        mimetype="audio/wav",
        as_attachment=False,
        download_name=f"audio_{idx}.wav",
    )


@app.route("/update", methods=["POST"])
def update_statuses():
    """
    	Update statuses from the submitted form data.
    """
    data = request.get_json()
    updates = data.get("updates", {})

    samples = load_manifest()
    for idx_str, new_status in updates.items():
        idx = int(idx_str)
        if 0 <= idx < len(samples):
            samples[idx]["status"] = new_status

    save_manifest(samples)
    return jsonify({"success": True, "updated": len(updates)})

load_thresholds()

if __name__ == "__main__":
    print(f"Loading manifest from: {MANIFEST_PATH}")
    print("Starting dashboard at http://localhost:5000")
    app.run(debug=True, port=5000)
