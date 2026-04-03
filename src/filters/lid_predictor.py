import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from typing import List, Optional, Dict, Any

class LIDPredictor:
    def __init__(self, model_id: str = "facebook/mms-lid-256", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id).to(device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model.eval()

        # model.config.id2label maps class index -> ISO 639-3 language code
        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

    def _load_audio(self, path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)  # shape: [channels, time]

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        return waveform.squeeze(0)  # shape: [time]

    @torch.no_grad()
    def predict_batch(
        self,
        audio_paths: List[str],
        expected_language: Optional[str] = None,
        min_keep_conf: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """
        expected_language: ISO 639-3 code, e.g.:
            hin, tam, tel, mal, ben, mar, kan, urd, asm, etc.
        """
        waves = [self._load_audio(p).cpu().numpy() for p in audio_paths]

        inputs = self.feature_extractor(
            waves,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        results = []
        for i, path in enumerate(audio_paths):
            top_prob, top_idx = torch.max(probs[i], dim=-1)
            pred_lang = self.id2label[top_idx.item()]

            expected_prob = None
            matches_expected = None
            keep = top_prob.item() >= min_keep_conf

            if expected_language is not None:
                exp_id = self.label2id.get(expected_language)
                if exp_id is not None:
                    expected_prob = probs[i, exp_id].item()
                    matches_expected = (pred_lang == expected_language)
                    keep = keep and matches_expected and (expected_prob >= min_keep_conf)
                else:
                    matches_expected = False
                    keep = False

            results.append({
                "audio_path": path,
                "predicted_language": pred_lang,
                "top_confidence": float(top_prob.item()),
                "expected_language": expected_language,
                "expected_confidence": expected_prob,
                "matches_expected": matches_expected,
                "keep": keep,
            })

        return results