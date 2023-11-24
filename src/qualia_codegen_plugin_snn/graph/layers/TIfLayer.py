from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass

from qualia_codegen_core.graph.layers.TBaseLayer import TBaseLayer
from qualia_codegen_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qualia_codegen_core.typing import NDArrayFloatOrInt

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@dataclass
class TIfLayer(TBaseLayer):
    v_threshold: NDArrayFloatOrInt
    v_reset: NDArrayFloatOrInt | None
    soft_reset: bool

    @property
    @override
    def weights(self) -> OrderedDict[str, NDArrayFloatOrInt]:
        w: OrderedDict[str, NDArrayFloatOrInt] = OrderedDict([('v_threshold', self.v_threshold)])
        if not self.soft_reset and self.v_reset is not None:
            w['v_reset'] = self.v_reset
        return w
