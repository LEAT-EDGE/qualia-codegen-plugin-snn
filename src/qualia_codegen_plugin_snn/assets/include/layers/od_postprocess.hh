/**
  ******************************************************************************
  * @file    od_postprocess.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    14 december 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _{{ node.layer.name | upper }}_H_
#define _{{ node.layer.name | upper }}_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define NUM_ANCHORS {{ node.layer.anchors.shape[0] }}

#define DETECTIONS_PER_IMAGE {{ node.layer.detections_per_image }}

typedef struct {
  {{ qtype2ctype(node.q.number_type, node.q.width) }} boxes[DETECTIONS_PER_IMAGE][4];
  {{ qtype2ctype(node.q.number_type, node.q.width) }} scores[DETECTIONS_PER_IMAGE];
  {{ qtype2ctype(node.q.number_type, node.q.width) }} labels[DETECTIONS_PER_IMAGE];
  size_t count;
} *{{ node.layer.name }}_output_type;

#if 0
void {{ node.layer.name }}(
  // Regression heads
  {% for s in node.layer.input_shape[:node.layer.num_fms] -%}
  const number_t input_regression_{{ loop.index }}{% for dim in s[1:] %}[{{ dim }}]{% endfor %}, // IN
  {% endfor %}
  // Classification heads
  {% for s in node.layer.input_shape[node.layer.num_fms:] -%}
  const number_t input_classification_{{ loop.index }}{% for dim in s[1:] %}[{{ dim }}]{% endfor %}, // IN
  {% endfor %}
  const number_t anchors[NUM_ANCHORS][4], // IN
  {{ node.layer.name }}_output_type *output);			          // OUT
#endif

#undef NUM_ANCHORS
#undef DETECTIONS_PER_IMAGE

#endif//_{{ node.layer.name | upper }}_H_
