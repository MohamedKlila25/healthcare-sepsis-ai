"""Validated extraction from synthetic text reports; OCR/VLM is a separate future module."""
import re

from pydantic import BaseModel, Field


class ExtractedMeasurement(BaseModel):
    field: str
    value: float
    unit: str
    source_text: str


class DocumentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    synthetic: bool = True


PATTERNS = {
    "Lactate": (r"lactate\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mmol/L)", (0.1, 40)),
    "WBC": (r"WBC\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(10\^9/L)", (0.1, 200)),
    "Temp": (r"(?:temperature|temp)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(C)", (25, 45)),
}


def extract_document(request):
    if not request.synthetic:
        raise ValueError("Only synthetic demonstration text is accepted")
    measurements, rejected = [], []
    for field, (pattern, limits) in PATTERNS.items():
        for match in re.finditer(pattern, request.text, re.IGNORECASE):
            value = float(match.group(1))
            if not limits[0] <= value <= limits[1]:
                rejected.append({"field": field, "reason": "outside parser plausibility bounds"})
            else:
                measurements.append(ExtractedMeasurement(field=field, value=value,
                    unit=match.group(2), source_text=match.group(0)).model_dump())
    return {"method": "regex_text_extraction", "synthetic": True, "measurements": measurements,
            "rejected": rejected, "requires_human_review": True,
            "limitations": "Text only; no image OCR or VLM. Missing units are never inferred. Not linked to patient records."}

