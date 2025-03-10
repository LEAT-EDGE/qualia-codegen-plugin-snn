# Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

from __future__ import annotations

import logging
import sys
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import qualia_codegen_core

from .graph import layers

if TYPE_CHECKING:
    from qualia_codegen_core.graph.layers import TBaseLayer
    from qualia_codegen_core.graph.ModelGraph import ModelGraph

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class Converter(qualia_codegen_core.Converter):
    layer_template_files: ClassVar[dict[type[TBaseLayer], str | None]] = {**qualia_codegen_core.Converter.layer_template_files,
        # SNN layers
        layers.TIfLayer: 'if',
        layers.TLifLayer: 'lif',

        # SNN OD layers
        layers.TObjectDetectionPostProcessLayer: 'od_postprocess',
    }

    TEMPLATE_PATH = files('qualia_codegen_plugin_snn.assets')

    def __init__(self,
                 output_path: Path | None = None,
                 dump_featuremaps: bool = False,  # noqa: FBT001, FBT002
                 timestep_mode: Literal['duplicate', 'iterate'] = 'duplicate') -> None:
        super().__init__(output_path=output_path, dump_featuremaps=dump_featuremaps)

        # Super failed to popuate template_path
        if self._template_path is None:
            return

        template_path: Path | None = None
        # Prepend to template_path so that our files have higher priority over qualia_codegen_core files
        if isinstance(Converter.TEMPLATE_PATH, Path): # Already Path objected, no need for hackery
            template_path = Converter.TEMPLATE_PATH
        elif sys.version_info >= (3, 10): # Python 3.10 may return MultiplexedPath
            from importlib.readers import MultiplexedPath
            if isinstance(Converter.TEMPLATE_PATH, MultiplexedPath):
                template_path = Converter.TEMPLATE_PATH / '' # / operator applies to underlying Path

        if template_path is not None:
            self._template_path.insert(0, template_path)
        else: # If we failed, also clear _template_path to fail conversion altogether instead of having incorrect search path
            self._template_path = None

        self.__timestep_mode = timestep_mode

    @override
    def write_model_header(self, modelgraph: ModelGraph) -> str:
        return self.render_template('include/model.hh',
                                    self.output_path_header / 'model.h',
                                    nodes=modelgraph.nodes,
                                    qtype2ctype=self.dataconverter.qtype2ctype,
                                    timestep_mode=self.__timestep_mode)
