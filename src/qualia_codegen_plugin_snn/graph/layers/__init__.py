from qualia_codegen_core.graph.layers import layers_t as core_layers_t

# SNN layers
from .TIfLayer import TIfLayer
from .TLifLayer import TLifLayer

# SNN OD layers
from .TObjectDetectionPostProcessLayer import TObjectDetectionPostProcessLayer

layers_t = {**core_layers_t,
    # SNN Layers
    'TIfLayer': TIfLayer,
    'TLifLayer': TLifLayer,

    # SNN OD Layers
    'TObjectDetectionPostProcessLayer': TObjectDetectionPostProcessLayer,
}

__all__ = ['TIfLayer', 'TLifLayer', 'TObjectDetectionPostProcessLayer']
