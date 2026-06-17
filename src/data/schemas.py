from dataclasses import dataclass
from typing import Optional, Any
import numpy as np

@dataclass
class AudioSample:
    """Standardized schema for audio samples across all pipelines."""
    path: str
    label: Optional[int] = None
    audio: Optional[np.ndarray] = None
    sr: Optional[int] = None
    status: str = "success"  # "success" or "error"
    error_reason: Optional[str] = None
    feature: Optional[Any] = None
