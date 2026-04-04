import io
import json
import os

import soundfile as sf
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

MANIFEST_PATH = "output/filtered_manifest.jsonl"
PER_PAGE = 20

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


@app.route("/")
def index():
    samples = load_manifest()
    page = request.args.get("page", 1, type=int)
    total_pages = (len(samples) + PER_PAGE - 1) // PER_PAGE

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
        total_samples=len(samples),
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


if __name__ == "__main__":
    print(f"Loading manifest from: {MANIFEST_PATH}")
    print("Starting dashboard at http://localhost:5000")
    app.run(debug=True, port=5000)
