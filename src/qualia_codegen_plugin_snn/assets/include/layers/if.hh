/**
  ******************************************************************************
  * @file    if.hh
  * @author  Jonathan Courtois, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2022
  * @brief   Template generating plain C code for the implementation of Leaky Integrate and Fire neuron
  */

#ifndef _{{ node.layer.name | upper }}_H_
#define _{{ node.layer.name | upper }}_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  {{ node.input_shape[0][-1] }}
// Input_Shape : {{ node.input_shape[0] }}
{% if node.input_shape[0] | length == 4 %}
#define INPUT_SAMPLESH   {{ node.input_shape[0][-3] }}
#define INPUT_SAMPLESW   {{ node.input_shape[0][-2] }}
{% elif node.input_shape[0] | length == 3 %}
#define INPUT_SAMPLES   {{ node.input_shape[0][-2] }}
{% elif node.input_shape[0] | length == 2 %}
#define INPUT_SAMPLES   1
{% endif %}

{% if node.input_shape[0] | length == 4 %}  // 2D
typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type[INPUT_SAMPLESH][INPUT_SAMPLESW][INPUT_CHANNELS];

#if 0
void {{ node.layer.name }}(
  const number_t input[INPUT_SAMPLESH][INPUT_SAMPLESW][INPUT_CHANNELS], 	    // IN
  number_t  output[INPUT_SAMPLESH][INPUT_SAMPLESW][INPUT_CHANNELS]);	        // OUT
#endif
{% else %} // 1D
typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type[INPUT_SAMPLES][INPUT_CHANNELS];

#if 0
void {{ node.layer.name }} (
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t  output[INPUT_SAMPLES][INPUT_CHANNELS]);	        // OUT
#endif

{% endif %}

#undef INPUT_CHANNELS  
{% if node.input_shape[0] | length == 4 %}
#undef INPUT_SAMPLESH
#undef INPUT_SAMPLESW
{% else %}
#undef INPUT_SAMPLES   
{% endif %}

#endif//_{{ node.layer.name | upper }}_H_
