from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .single_detectors import DetectorInfo


@dataclass
class DetectorRegistry:
    detectors: List[DetectorInfo] = field(default_factory=list)
    by_quasi: Dict[str, List[int]] = field(default_factory=dict)
    by_product: Dict[str, List[int]] = field(default_factory=dict)

    def add(self, info: DetectorInfo) -> None:
        det_id = info.id
        self.detectors.append(info)
        if info.kind == 'quasi':
            self.by_quasi.setdefault(info.label, []).append(det_id)
        elif info.kind == 'product':
            self.by_product.setdefault(info.label, []).append(det_id)

    def get_detector(self, det_id: int) -> Optional[DetectorInfo]:
        if 0 <= det_id < len(self.detectors):
            return self.detectors[det_id]
        return None

    def get_detectors_for_quasi(self, label: str) -> List[DetectorInfo]:
        ids = self.by_quasi.get(label, [])
        return [self.detectors[i] for i in ids if 0 <= i < len(self.detectors)]

    def get_detectors_for_product(self, label: str) -> List[DetectorInfo]:
        ids = self.by_product.get(label, [])
        return [self.detectors[i] for i in ids if 0 <= i < len(self.detectors)]
