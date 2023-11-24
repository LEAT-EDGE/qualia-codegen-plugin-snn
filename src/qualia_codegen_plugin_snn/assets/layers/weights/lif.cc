/**
  ******************************************************************************
  * @file    weights/lif.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    9 october 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const {{ weights.v_threshold.dtype }} {{ node.layer.name }}_v_threshold = {{ weights.v_threshold.data }};
{% if not node.layer.soft_reset %}
const {{ weights.v_reset.dtype }} {{ node.layer.name }}_v_reset = {{ weights.v_reset.data }};
{% endif %}
const {{ weights.reciprocal_tau.dtype }} {{ node.layer.name}}_reciprocal_tau = {{ node.layer.reciprocal_tau }};
