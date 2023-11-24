from __future__ import annotations

import sys
from dataclasses import dataclass

from qualia_codegen_core.typing import TYPE_CHECKING

from .TIfLayer import TIfLayer

# We are inside a TYPE_CHECKING block but our custom TYPE_CHECKING constant triggers TCH001-TCH003 so ignore them
if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: TCH003

    from qualia_codegen_core.typing import NDArrayFloatOrInt

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@dataclass
class TLifLayer(TIfLayer):
    reciprocal_tau: NDArrayFloatOrInt
    decay_input: bool

    @property
    @override
    def weights(self) -> OrderedDict[str, NDArrayFloatOrInt]:
        w = super().weights
        w['reciprocal_tau'] = self.reciprocal_tau
        return w
