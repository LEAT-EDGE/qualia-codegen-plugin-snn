/**
  ******************************************************************************
  * @file    if.cc
  * @author  Jonathan Courtois, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    02 september 2022
  * @brief   Template generating plain C code for the implementation of Integrate and Fire neuron
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  {{ node.input_shape[0][-1] }}

#define SOFTRESET       {{ node.layer.soft_reset}}

// For fixed point quantization
#define INPUT_SCALE_FACTOR {{ node.innodes[0].q.output_scale_factor }}
#define WEIGHTS_SCALE_FACTOR {{ node.q.weights_scale_factor }}
#define OUTPUT_SCALE_FACTOR {{ node.q.output_scale_factor }}
#define TMP_SCALE_FACTOR {{ [node.innodes[0].q.output_scale_factor, node.q.weights_scale_factor] | max }}
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}
#define LONG_NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.long_width) }}


{% if node.input_shape[0] | length == 4 %}  // 2D
#define INPUT_SAMPLESH   {{ node.input_shape[0][-3] }}
#define INPUT_SAMPLESW   {{ node.input_shape[0][-2] }}

static NUMBER_T {{ node.layer.name }}_potent[INPUT_SAMPLESH][INPUT_SAMPLESW][INPUT_CHANNELS];	        // INOUT

static inline void {{ node.layer.name }}(
  const NUMBER_T input[INPUT_SAMPLESH][INPUT_SAMPLESW][INPUT_CHANNELS], 	    // IN
  const NUMBER_T v_threshold,
#if !SOFTRESET
  const NUMBER_T v_reset,
#endif
  NUMBER_T output[INPUT_SAMPLESH][INPUT_SAMPLESW][INPUT_CHANNELS]) {	        // OUT

  LONG_NUMBER_T v_threshold_scaled_to_tmp = scale(NUMBER_T, (LONG_NUMBER_T)v_threshold, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR);
  LONG_NUMBER_T tmp;
  unsigned short j, h, w;

  for (h = 0; h < INPUT_SAMPLESH; h++) {
    for (w = 0; w < INPUT_SAMPLESW; w++) {
      for (j = 0; j < INPUT_CHANNELS; j++) {
        // Scale potential and inputs to same factor, max between input and weights(potential) scale factor for the addition and put result into long variable to avoid overflow
        tmp = scale(NUMBER_T, (LONG_NUMBER_T){{ node.layer.name }}_potent[h][w][j], WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR);
        tmp = tmp + scale(NUMBER_T, (LONG_NUMBER_T)input[h][w][j], INPUT_SCALE_FACTOR - TMP_SCALE_FACTOR);

        // fire
        if (tmp - v_threshold_scaled_to_tmp >= 0) {
          output[h][w][j] = clamp_to(NUMBER_T, scale(NUMBER_T, 1, -OUTPUT_SCALE_FACTOR));

          // Reset
#if SOFTRESET
          tmp = tmp - v_threshold_scaled_to_tmp;

          // Scale back to WEIGHTS_SCALE_FACTOR used by potent
          tmp = scale(NUMBER_T, tmp, TMP_SCALE_FACTOR - WEIGHTS_SCALE_FACTOR);
          {{ node.layer.name }}_potent[i][j] = clamp_to(NUMBER_T, tmp);
#else
          // v_reset already scaled to WEIGHTS_SCALE_FACTOR
          {{ node.layer.name }}_potent[h][w][j] = v_reset;
#endif
        } else {
          output[h][w][j] = 0;

          // Scale back to WEIGHTS_SCALE_FACTOR used by potent
          tmp = scale(NUMBER_T, tmp, TMP_SCALE_FACTOR - WEIGHTS_SCALE_FACTOR);
          {{ node.layer.name }}_potent[h][w][j] = clamp_to(NUMBER_T, tmp);
        }

      }
    }
  }
}

static inline void {{ node.layer.name }}_reset_pot(){
  unsigned short j, h, w;
  for (h = 0;  h < INPUT_SAMPLESH; h++) {
    for (w = 0; w < INPUT_SAMPLESW; w++) {
      for (j = 0; j < INPUT_CHANNELS; j++) {
        {{ node.layer.name }}_potent[h][w][j] = 0;
      }
    }
  }
}

#undef INPUT_SAMPLESH
#undef INPUT_SAMPLESW

{% else %} // 1D
{% if node.input_shape[0] | length == 3 %}
#define INPUT_SAMPLES   {{ node.input_shape[0][-2] }}
{% elif node.input_shape[0] | length == 2 %}
#define INPUT_SAMPLES   1
{% endif %}

static NUMBER_T {{ node.layer.name }}_potent[INPUT_SAMPLES][INPUT_CHANNELS];	        // INOUT

static inline void {{ node.layer.name }}(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  const NUMBER_T v_threshold,
#if !SOFTRESET
  const NUMBER_T v_reset,
#endif
  NUMBER_T output[INPUT_SAMPLES][INPUT_CHANNELS]) {	        // OUT

  LONG_NUMBER_T v_threshold_scaled_to_tmp = scale(NUMBER_T, (LONG_NUMBER_T)v_threshold, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR);
  LONG_NUMBER_T tmp;
  unsigned short i, j; 	                                    // loop indexes for output volume

  for (i = 0; i < INPUT_SAMPLES; i++) {
    for (j = 0; j < INPUT_CHANNELS; j++) {
      // Scale potential and inputs to same factor, max between input and weights(potential) scale factor for the addition and put result into long variable to avoid overflow
      tmp = scale(NUMBER_T, (LONG_NUMBER_T){{ node.layer.name }}_potent[i][j], WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR);
      tmp = tmp + scale(NUMBER_T, (LONG_NUMBER_T)input[i][j], INPUT_SCALE_FACTOR - TMP_SCALE_FACTOR);

      // fire
      if (tmp - v_threshold_scaled_to_tmp >= 0) {
        output[i][j] = clamp_to(NUMBER_T, scale(NUMBER_T, 1, -OUTPUT_SCALE_FACTOR));

        // Reset
#if SOFTRESET
        tmp = tmp - v_threshold_scaled_to_tmp;

        // Scale back to WEIGHTS_SCALE_FACTOR used by potent
        tmp = scale(NUMBER_T, tmp, TMP_SCALE_FACTOR - WEIGHTS_SCALE_FACTOR);
        {{ node.layer.name }}_potent[i][j] = clamp_to(NUMBER_T, tmp);
#else
        // v_reset already scaled to WEIGHTS_SCALE_FACTOR
        {{ node.layer.name }}_potent[i][j] = v_reset;
#endif
      } else {
        output[i][j] = 0;

        // Scale back to WEIGHTS_SCALE_FACTOR used by potent
        tmp = scale(NUMBER_T, tmp, TMP_SCALE_FACTOR - WEIGHTS_SCALE_FACTOR);
        {{ node.layer.name }}_potent[i][j] = clamp_to(NUMBER_T, tmp);
      }

    }
  }
}

static inline void {{ node.layer.name }}_reset_pot(){
    unsigned short j, i; 
    for (i = 0; i < INPUT_SAMPLES; i++) {
        for (j = 0; j < INPUT_CHANNELS; j++) {
            {{ node.layer.name }}_potent[i][j] = 0;
        }
    }
}

#undef INPUT_SAMPLES
{% endif %}

#undef INPUT_CHANNELS

#undef SOFTRESET

#undef INPUT_SCALE_FACTOR
#undef WEIGHTS_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
