/**
  ******************************************************************************
  * @file    od_postprocess.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    4 november 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif
{% if node.q.number_type.__name__ == 'int' %}
#include <math.h>
{% endif %}

// qsort()
#include <stdlib.h>

//memcpy()
#include <string.h>

#define NUM_CLASSES {{ node.layer.num_classes }}
#define NUM_ANCHORS {{ node.layer.anchors.shape[0] }}

#define NUM_BOXES {% for s in node.layer.input_shape[:node.layer.num_fms] %}{{ s[1:] | join('*') }} / 4{{ ' + ' if not loop.last }}{% endfor %}
#define NUM_LOGITS {% for s in node.layer.input_shape[node.layer.num_fms:] %}{{ s[1:] | join('*') }} / NUM_CLASSES{{ ' + ' if not loop.last }}{% endfor %}

//FIXME: fixed-point
#define BOX_CODER_WEIGHT_X {{ node.layer.box_coder_weights[0] }}
#define BOX_CODER_WEIGHT_Y {{ node.layer.box_coder_weights[1] }}
#define BOX_CODER_WEIGHT_W {{ node.layer.box_coder_weights[2] }}
#define BOX_CODER_WEIGHT_H {{ node.layer.box_coder_weights[3] }}

#define IMAGE_HEIGHT {{ node.layer.image_shape[0] }}
#define IMAGE_WIDTH {{ node.layer.image_shape[1] }}

#define TOPK_CANDIDATES {{ node.layer.topk_candidates }}

//FIXME: fixed-point
#define SCORE_THRESHOLD {{ node.layer.score_threshold }}

#define DETECTIONS_PER_IMAGE {{ node.layer.detections_per_image }}

//FIXME: fixed-point
#define NMS_THRESHOLD {{ node.layer.nms_threshold }}

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR {{ node.q.weights_scale_factor }}
#define INPUT_SCALE_FACTOR {{ node.innodes[0].q.output_scale_factor }}
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}
#define LONG_NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.long_width) }}


static inline NUMBER_T sigmoid(const NUMBER_T x) {
{% if node.q.number_type.__name__ == 'int' %}
#error
{% else %}
  return 1 / (1 + expf(-x));
{% endif %}
}

{% for s in node.layer.input_shape[:node.layer.num_fms] %}
static inline void reshape_regression_head_outputs_{{ loop.index }}(
  const NUMBER_T head_output{% for d in s[1:] %}[{{ d }}]{% endfor %}, // IN (H, W, A * NUM_CLASSES)
  NUMBER_T out[{{ s[1:] | join ('*') }} / 4][4]) { // OUT (H * W * A, NUM_CLASSES)
  memcpy(out, head_output, {{ s[1:] | join('*') }} * sizeof(NUMBER_T));
}
{% endfor %}

{% for s in node.layer.input_shape[node.layer.num_fms:] %}
static inline void reshape_classification_head_outputs_{{ loop.index }}(
  const NUMBER_T head_output{% for d in s[1:] %}[{{ d }}]{% endfor %}, // IN (H, W, A * 4)
  NUMBER_T out[{{ s[1:] | join ('*') }} / NUM_CLASSES][NUM_CLASSES]) { // OUT (H * W * A, 4)
  memcpy(out, head_output, {{ s[1:] | join('*') }} * sizeof(NUMBER_T));
}
{% endfor %}

static inline void decode_single(
  NUMBER_T boxes[NUM_BOXES][4], // IN-OUT
  const NUMBER_T anchors[NUM_ANCHORS][4]) { // IN

  for (size_t i = 0, j = 0; i < NUM_BOXES && j < NUM_ANCHORS; i++, j++) {
    NUMBER_T width = anchors[j][2] - anchors[j][0];
    NUMBER_T height = anchors[j][3] - anchors[j][1];
    NUMBER_T ctr_x = anchors[j][0] + width / 2;
    NUMBER_T ctr_y = anchors[j][1] + height / 2;
    NUMBER_T dx = boxes[i][0] / BOX_CODER_WEIGHT_X;
    NUMBER_T dy = boxes[i][1] / BOX_CODER_WEIGHT_Y;
    NUMBER_T dw = boxes[i][2] / BOX_CODER_WEIGHT_W;
    NUMBER_T dh = boxes[i][3] / BOX_CODER_WEIGHT_H;

    // clamp w?
    // clamp h?

    NUMBER_T pred_ctr_x = dx * width + ctr_x;
    NUMBER_T pred_ctr_y = dy * height + ctr_y;
{% if node.q.number_type.__name__ == 'int' %}
  #error
{% else %}
    NUMBER_T pred_w = expf(dw) * width;
    NUMBER_T pred_h = expf(dh) * height;
{% endif %}

    NUMBER_T c_to_c_h = pred_h / 2;
    NUMBER_T c_to_c_w = pred_w / 2;

    NUMBER_T pred_boxes1 = pred_ctr_x - c_to_c_w;
    NUMBER_T pred_boxes2 = pred_ctr_y - c_to_c_h;
    NUMBER_T pred_boxes3 = pred_ctr_x + c_to_c_w;
    NUMBER_T pred_boxes4 = pred_ctr_y + c_to_c_h;

    boxes[i][0] = pred_boxes1;
    boxes[i][1] = pred_boxes2;
    boxes[i][2] = pred_boxes3;
    boxes[i][3] = pred_boxes4;
  }
}

static inline void clip_boxes_to_images(
  NUMBER_T boxes[NUM_BOXES][4]) { // IN-OUT
  for (size_t i = 0; i < NUM_BOXES; i++) {
    NUMBER_T boxes_x1 = boxes[i][0];
    NUMBER_T boxes_y1 = boxes[i][1];
    NUMBER_T boxes_x2 = boxes[i][2];
    NUMBER_T boxes_y2 = boxes[i][3];

    if (boxes_x1 > IMAGE_WIDTH) {
      boxes_x1 = IMAGE_WIDTH;
    } else if (boxes_x1 < 0) {
      boxes_x1 = 0;
    }

    if (boxes_y1 > IMAGE_HEIGHT) {
      boxes_y1 = IMAGE_HEIGHT;
    } else if (boxes_y1 < 0) {
      boxes_y1 = 0;
    }

    if (boxes_x2 > IMAGE_WIDTH) {
      boxes_x2 = IMAGE_WIDTH;
    } else if (boxes_x2 < 0) {
      boxes_x2 = 0;
    }

    if (boxes_y2 > IMAGE_HEIGHT) {
      boxes_y2 = IMAGE_HEIGHT;
    } else if (boxes_y2 < 0) {
      boxes_y2 = 0;
    }

    boxes[i][0] = boxes_x1;
    boxes[i][1] = boxes_y1;
    boxes[i][2] = boxes_x2;
    boxes[i][3] = boxes_y2;
  }
}

typedef struct {
  size_t idx;
  NUMBER_T score;
} scores_idxs_t;

static int compare_scores_idxs(const void *a, const void *b) {
  float fa = (*(scores_idxs_t *)a).score;
  float fb = (*(scores_idxs_t *)b).score;
  return (fb > fa) - (fb < fa);
}

static inline size_t get_all_preds(
  const NUMBER_T boxes[NUM_BOXES][4], // IN
  const NUMBER_T logits[NUM_LOGITS][NUM_CLASSES], // IN

  NUMBER_T image_boxes[TOPK_CANDIDATES * NUM_CLASSES][4], // OUT
  NUMBER_T image_scores[TOPK_CANDIDATES * NUM_CLASSES], // OUT
  NUMBER_T image_labels[TOPK_CANDIDATES * NUM_CLASSES]) { // OUT

  size_t count = 0;
  static scores_idxs_t scores_idxs[NUM_LOGITS];

  for (size_t label = 0; label < NUM_CLASSES; label++) {
    for (size_t i = 0; i < NUM_LOGITS; i++) {
      scores_idxs[i].idx = i;
      scores_idxs[i].score = sigmoid(logits[i][label]);
    }

    qsort(scores_idxs, NUM_LOGITS, sizeof(scores_idxs_t), compare_scores_idxs);

    for (size_t i = 0; i < NUM_LOGITS && count < TOPK_CANDIDATES * (label + 1); i++, count++) {
      // Remove low scoring boxes
      if (scores_idxs[i].score < SCORE_THRESHOLD) {
        break;
      }

      for (size_t k = 0; k < 4; k++) {
        image_boxes[count][k] = boxes[scores_idxs[i].idx][k];
      }
      image_scores[count] = scores_idxs[i].score;
      image_labels[count] = label;
    }
  }

  return count;
}

static inline size_t get_nms_preds(
  const size_t count, // IN
  const NUMBER_T image_boxes[TOPK_CANDIDATES * NUM_CLASSES][4], // IN
  const NUMBER_T image_scores[TOPK_CANDIDATES * NUM_CLASSES], // IN
  const NUMBER_T image_labels[TOPK_CANDIDATES * NUM_CLASSES], // IN

  NUMBER_T filtered_boxes[DETECTIONS_PER_IMAGE][4],
  NUMBER_T filtered_scores[DETECTIONS_PER_IMAGE],
  NUMBER_T filtered_labels[DETECTIONS_PER_IMAGE]
  ) {
  static NUMBER_T areas[TOPK_CANDIDATES * NUM_CLASSES]; // Compute in inner loop to save on memory
  static size_t order[TOPK_CANDIDATES * NUM_CLASSES];
  size_t order_i = 0;
  static size_t new_order[TOPK_CANDIDATES * NUM_CLASSES];
  size_t new_order_i = 0;

  size_t results = 0;

  // Compute areas
  for (size_t i = 0; i < count; i++) {
    NUMBER_T x1 = image_boxes[i][0];
    NUMBER_T y1 = image_boxes[i][1];
    NUMBER_T x2 = image_boxes[i][2];
    NUMBER_T y2 = image_boxes[i][3];

    areas[i] = (x2 - x1) * (y2 - y1);
  }

  // 'order' keep the indices of bounding boxes with largest scores, in decreasing order
  // since 'boxes' and 'scores' are already sorted, it's just a range
  for (order_i = 0; order_i < count; order_i++) {
    order[order_i] = order_i;
  }

  while (order_i > 0 && results < DETECTIONS_PER_IMAGE) {
    // The index of the largest confidence score
    size_t idx = order[0];
    filtered_boxes[results][0] = image_boxes[idx][0];
    filtered_boxes[results][1] = image_boxes[idx][1];
    filtered_boxes[results][2] = image_boxes[idx][2];
    filtered_boxes[results][3] = image_boxes[idx][3];
    filtered_scores[results] = image_scores[idx];
    filtered_labels[results] = image_labels[idx];
    results++;

    for (size_t i = 1; i < order_i; i++) {
      // Compute ordinates for IoUlong_
      NUMBER_T xx1 = image_boxes[order[i]][0]; // x1[order[i]]
      if (xx1 < image_boxes[idx][0] /* x1[idx] */) {
        xx1 = image_boxes[idx][0]; // x1[idx]
      }
      NUMBER_T yy1 = image_boxes[order[i]][1];
      if (yy1 < image_boxes[idx][1] /* y1[idx] */) {
        yy1 = image_boxes[idx][1]; // y1[idx]
      }
      NUMBER_T xx2 = image_boxes[order[i]][2];
      if (xx2 > image_boxes[idx][2] /* x2[idx] */) {
        xx2 = image_boxes[idx][2]; // x2[idx]
      }
      NUMBER_T yy2 = image_boxes[order[i]][3];
      if (yy2 > image_boxes[idx][3] /* y2[idx] */) {
        yy2 = image_boxes[idx][3]; // y2[idx]
      }

      // Compute areas for IoU
      NUMBER_T w = (xx2 - xx1);
      if (w < 0) {
        w = 0;
      }
      NUMBER_T h = (yy2 - yy1);
      if (h < 0) {
        h = 0;
      }
      NUMBER_T intersection = w * h;

      // Compute the IoU
      NUMBER_T iou = intersection / (areas[idx] + areas[order[i]] - intersection);
      if (iou <= NMS_THRESHOLD) {
        new_order[new_order_i] = order[i];
        new_order_i++;
      }
    }
    
    for (size_t i = 0; i < new_order_i; i++) {
      order[i] = new_order[i];
    }
    order_i = new_order_i;
    new_order_i = 0;
  }

  return results;
}

static inline void {{ node.layer.name }}(
  // Regression heads
  {% for s in node.layer.input_shape[:node.layer.num_fms] -%}
  const NUMBER_T input_regression_{{ loop.index }}{% for dim in s[1:] %}[{{ dim }}]{% endfor %}, // IN
  {% endfor %}
  // Classification heads
  {% for s in node.layer.input_shape[node.layer.num_fms:] -%}
  const NUMBER_T input_classification_{{ loop.index }}{% for dim in s[1:] %}[{{ dim }}]{% endfor %}, // IN
  {% endfor %}
  const NUMBER_T anchors[NUM_ANCHORS][4], // IN
  {{ node.layer.name }}_output_type output) {			          // OUT

  static size_t offset;
  static NUMBER_T boxes[NUM_BOXES][4];
  static NUMBER_T logits[NUM_LOGITS][NUM_CLASSES];

  static NUMBER_T image_boxes[TOPK_CANDIDATES * NUM_CLASSES][4];
  static NUMBER_T image_scores[TOPK_CANDIDATES * NUM_CLASSES];
  static NUMBER_T image_labels[TOPK_CANDIDATES * NUM_CLASSES];

  // Reshape and concat regression heads to boxes
  offset = 0;
  {% for s in node.layer.input_shape[:node.layer.num_fms] %}
  reshape_regression_head_outputs_{{ loop.index }}(
    input_regression_{{ loop.index }},
    &boxes[offset]
  );
  offset += {{ s[1:] | join('*') }} / 4;
  {% endfor %}

  // Reshape and concat classification heads to logits
  offset = 0;
  {% for s in node.layer.input_shape[node.layer.num_fms:] %}
  reshape_classification_head_outputs_{{ loop.index }}(
    input_classification_{{ loop.index }},
    &logits[offset]
  );
  offset += {{ s[1:] | join('*') }} / NUM_CLASSES;
  {% endfor %}

  decode_single(
    boxes, // IN-OUT
    anchors // IN
  );

  clip_boxes_to_images(
    boxes // IN-OUT
  );

  (*output).count = get_all_preds(
    boxes, // IN
    logits, // IN
    image_boxes, // OUT
    image_scores, // OUT
    image_labels // OUT
  );

  (*output).count = get_nms_preds(
    (*output).count, // IN
    image_boxes, // IN
    image_scores, // IN
    image_labels, // IN
    (*output).boxes, // OUT
    (*output).scores, // OUT
    (*output).labels // OUT
  );
}

#undef NUM_CLASSES
#undef NUM_ANCHORS
#undef NUM_BOXES
#undef NUM_LOGITS
#undef BOX_CODER_WEIGHT_X
#undef BOX_CODER_WEIGHT_Y
#undef BOX_CODER_WEIGHT_W
#undef BOX_CODER_WEIGHT_H
#undef IMAGE_HEIGHT
#undef IMAGE_WIDTH
#undef TOPK_CANDIDATES
#undef SCORE_THRESHOLD
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
