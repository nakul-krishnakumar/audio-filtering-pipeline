LANGUAGE_CODE_MAP = {
    "assamese":  {"iso2": "as", "iso3": "asm"},
    "bengali":   {"iso2": "bn", "iso3": "ben"},
    "gujarati":  {"iso2": "gu", "iso3": "guj"},
    "hindi":     {"iso2": "hi", "iso3": "hin"},
    "kannada":   {"iso2": "kn", "iso3": "kan"},
    "kashmiri":  {"iso2": "ks", "iso3": "kas"},
    "konkani":   {"iso2": "kok", "iso3": "gom"},   # special case
    "maithili":  {"iso2": "mai", "iso3": "mai"},   # no ISO2 → use same
    "malayalam": {"iso2": "ml", "iso3": "mal"},
    "marathi":   {"iso2": "mr", "iso3": "mar"},
    "nepali":    {"iso2": "ne", "iso3": "npi"},
    "odia":      {"iso2": "or", "iso3": "ory"},
    "punjabi":   {"iso2": "pa", "iso3": "pan"},
    "sanskrit":  {"iso2": "sa", "iso3": "san"},
    "santali":   {"iso2": "sat", "iso3": "sat"},   # no ISO2
    "sindhi":    {"iso2": "sd", "iso3": "snd"},
    "tamil":     {"iso2": "ta", "iso3": "tam"},
    "telugu":    {"iso2": "te", "iso3": "tel"},
    "urdu":      {"iso2": "ur", "iso3": "urd"},
}

ISO2_TO_ISO3 = {v["iso2"]: v["iso3"] for v in LANGUAGE_CODE_MAP.values()}

ISO3_TO_ISO2 = {v["iso3"]: v["iso2"] for v in LANGUAGE_CODE_MAP.values()}