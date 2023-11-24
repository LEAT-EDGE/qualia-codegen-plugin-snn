from dataclasses import dataclass

from qualia_codegen_core.graph.layers.TBaseLayer import TBaseLayer
from qualia_codegen_core.typing import NDArrayFloatOrInt


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
    def weights(self) -> dict[str, NDArrayFloatOrInt]:
        return {'anchors': self.anchors}
