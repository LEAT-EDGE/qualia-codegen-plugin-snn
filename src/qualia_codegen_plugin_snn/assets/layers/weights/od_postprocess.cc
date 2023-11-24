/**
  ******************************************************************************
  * @file    weights/od_postprocess.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    3 november 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define ANCHORS {{ node.layer.anchors.shape[0] }}

const {{ weights.anchors.dtype }} {{ node.layer.name }}_anchors[ANCHORS][4] = {{ weights.anchors.data }};

#undef ANCHORS
