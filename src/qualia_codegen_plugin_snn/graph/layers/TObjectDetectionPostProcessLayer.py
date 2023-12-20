
from __future__ import annotations

import sys
from dataclasses import dataclass

from qualia_codegen_core.graph.layers.TBaseLayer import TBaseLayer
from qualia_codegen_core.typing import TYPE_CHECKING, NDArrayFloatOrInt

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: TCH003

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@dataclass
class TObjectDetectionPostProcessLayer(TBaseLayer):
    num_classes: int
    num_fms: int
    image_shape: tuple[int, ...]
    score_threshold: float
    nms_threshold: float
    topk_candidates: int
    detections_per_image: int
    box_coder_weights: list[float]
    anchors: NDArrayFloatOrInt

    @property
    @override
    def weights(self) -> OrderedDict[str, NDArrayFloatOrInt]:
        w = super().weights
        w['anchors'] = self.anchors
        return w
