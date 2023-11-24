from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Protocol, cast

import numpy as np
import qualia_codegen_core.graph
from qualia_codegen_core.graph.layers import TBaseLayer, TInputLayer
from qualia_codegen_core.typing import DTypes, Shape, Shapes
from spikingjelly.activation_based.layer import SeqToANNContainer, torch  # type: ignore[import]
from spikingjelly.activation_based.neuron import IFNode, LIFNode, ParametricLIFNode  # type: ignore[import]

from .layers import (
    TIfLayer,
    TLifLayer,
)

if TYPE_CHECKING:
    from qualia_codegen_core.graph import ModelGraph
    from torch.nn import Module

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

class SequentialForward(Protocol):
    """Type Sequential.forward with torch.Tensor as the original Sequential.forward is untyped and makes mypy unhappy."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

logger = logging.getLogger(__name__)

class TorchModelGraph(qualia_codegen_core.graph.TorchModelGraph):
    MODULE_MAPPING: ClassVar[dict[type[Module], Callable[[Module, TBaseLayer], tuple[type[TBaseLayer], list[Any]]]]] = {
        # SNN layers
        IFNode: lambda module, _: (TIfLayer, [np.array(cast(IFNode, module).v_threshold, dtype=np.float32),
                                              np.array(cast(IFNode, module).v_reset, dtype=np.float32)
                                                if module.v_reset is not None else None,
                                              int(cast(IFNode, module).v_reset is None)]),
        LIFNode: lambda module, _: (TLifLayer, [np.array(cast(LIFNode, module).v_threshold, dtype=np.float32),
                                                np.array(cast(LIFNode, module).v_reset, dtype=np.float32)
                                                  if module.v_reset is not None else None,
                                                int(cast(LIFNode, module).v_reset is None),
                                                np.array(1 / cast(LIFNode, module).tau, dtype=np.float32),
                                                int(cast(LIFNode, module).decay_input)]),
        ParametricLIFNode: lambda module, _: (TLifLayer,
                                              [np.array(cast(ParametricLIFNode, module).v_threshold, dtype=np.float32),
                                               np.array(cast(ParametricLIFNode, module).v_reset, dtype=np.float32)
                                                 if module.v_reset is not None else None,
                                               int(cast(ParametricLIFNode, module).v_reset is None),
                                               np.array(float(cast(ParametricLIFNode, module).w.sigmoid()), dtype=np.float32),
                                               int(cast(ParametricLIFNode, module).decay_input)]),
        **qualia_codegen_core.graph.TorchModelGraph.MODULE_MAPPING,
    }

    @override
    def convert(self,
                custom_layers: dict[type[Module],
                                             Callable[[Module, TBaseLayer],
                                                      tuple[type[TBaseLayer], list[Any]]]] | None = None) -> ModelGraph | None:
        custom_layers = custom_layers if custom_layers is not None else {}
        custom_layers = {**TorchModelGraph.MODULE_MAPPING, **custom_layers}

        # Monkey-patch SeqToANNContainer forward() to be able to trace enclosed module properly
        seqtoanncontainer_forward = SeqToANNContainer.forward
        SeqToANNContainer.forward = lambda self, x_seq: cast(SequentialForward, super(type(self), self)).forward(x_seq)

        ret = super().convert(custom_layers)

        # Restore original method
        SeqToANNContainer.forward = seqtoanncontainer_forward

        return ret

    @override
    def _convert_placeholder(self) -> TBaseLayer | None:
        if not hasattr(self._model, 'input_shape') or not isinstance(self._model.input_shape, tuple):
            logger.error('Model must have input_shape attribute')
            return None

        shp: Shape = Shape(self._model.input_shape)
        if not getattr(self._model, 'is_snn', False):
            # Prepend dummy dimension instead of timestep if not SNN model
            shp = Shape((1, *shp))

        # Assume input is single-precision floating-point # WIP it could change
        return TInputLayer(Shapes((shp,)), Shapes((shp,)), DTypes((np.float32,)), 'input')
