from dataclasses import dataclass
from typing import Optional

@dataclass
class LatencyMetrics:
    feature_extraction_ms: float = 0.0
    classifier_ms: float = 0.0
    end_to_end_ms: float = 0.0
    note: Optional[str] = None
    
    def to_dict(self):
        return {
            "feature_extraction_ms_per_sample": round(self.feature_extraction_ms, 4),
            "classifier_ms_per_sample": round(self.classifier_ms, 4),
            "end_to_end_ms_per_sample": round(self.end_to_end_ms, 4),
            "latency_note": self.note
        }
