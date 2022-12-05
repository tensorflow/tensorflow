/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_NON_MAX_SUPPRESSION_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_NON_MAX_SUPPRESSION_H_

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <queue>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {
namespace {

// Helper functions
template <typename T>
inline T Truncate(int32_t input, T range_min = std::numeric_limits<T>::min(),
                  T range_max = std::numeric_limits<T>::max()) {
  return static_cast<T>(
      std::min<int32_t>(range_max, std::max<int32_t>(range_min, input)));
}

template <typename T>
T Rescale(T orig_value, int32_t orig_zero_point, int32_t rescale_multiplier,
          int32_t rescale_shift, int32_t rescale_zero_point) {
  const int32_t rescaled_value =
      MultiplyByQuantizedMultiplier(
          ((static_cast<int32_t>(orig_value) - orig_zero_point)),
          rescale_multiplier, rescale_shift) +
      rescale_zero_point;
  return static_cast<T>(rescaled_value);
}

// A pair of diagonal corners of the box.
template <typename T>
struct BoxCornerEncoding {
  T y1;
  T x1;
  T y2;
  T x2;
};

template <typename T>
inline std::pair<T, T> ComputeIntersectionAndUnion(
    const BoxCornerEncoding<T>& box_1, const BoxCornerEncoding<T>& box_2) {
  const T box_1_y_min = std::min<T>(box_1.y1, box_1.y2);
  const T box_1_y_max = std::max<T>(box_1.y1, box_1.y2);
  const T box_1_x_min = std::min<T>(box_1.x1, box_1.x2);
  const T box_1_x_max = std::max<T>(box_1.x1, box_1.x2);

  const T box_2_y_min = std::min<T>(box_2.y1, box_2.y2);
  const T box_2_y_max = std::max<T>(box_2.y1, box_2.y2);
  const T box_2_x_min = std::min<T>(box_2.x1, box_2.x2);
  const T box_2_x_max = std::max<T>(box_2.x1, box_2.x2);

  const T area_1 = (box_1_y_max - box_1_y_min) * (box_1_x_max - box_1_x_min);
  const T area_2 = (box_2_y_max - box_2_y_min) * (box_2_x_max - box_2_x_min);

  constexpr T k_zero = static_cast<T>(0);
  if (area_1 <= k_zero || area_2 <= k_zero) {
    return {0.f, 0.f};
  }

  const T intersection_ymax = std::min<T>(box_1_y_max, box_2_y_max);
  const T intersection_xmax = std::min<T>(box_1_x_max, box_2_x_max);
  const T intersection_ymin = std::max<T>(box_1_y_min, box_2_y_min);
  const T intersection_xmin = std::max<T>(box_1_x_min, box_2_x_min);

  const T intersection_area =
      std::max<T>(intersection_ymax - intersection_ymin, k_zero) *
      std::max<T>(intersection_xmax - intersection_xmin, k_zero);

  const T union_area = area_1 + area_2 - intersection_area;

  return std::make_pair(intersection_area, union_area);
}

// Float IOU implementation.
inline float ComputeIou(const float* boxes, int i, int j) {
  const auto& box_i =
      reinterpret_cast<const BoxCornerEncoding<float>*>(boxes)[i];
  const auto& box_j =
      reinterpret_cast<const BoxCornerEncoding<float>*>(boxes)[j];

  std::pair<float, float> iou_operands =
      ComputeIntersectionAndUnion(box_i, box_j);

  const float intersection_area = iou_operands.first;
  const float union_area = iou_operands.second;

  if (union_area == 0.f) {
    return 0.f;
  }

  return intersection_area / union_area;
}

// Quantized IOU implementation.
template <typename T>
inline T ComputeIou(const T* boxes, int i, int j,
                    const NonMaxSuppressionParams& params) {
  auto& box_i_orig = reinterpret_cast<const BoxCornerEncoding<T>*>(boxes)[i];
  auto& box_j_orig = reinterpret_cast<const BoxCornerEncoding<T>*>(boxes)[j];

  using CalcT = int32_t;

  BoxCornerEncoding<CalcT> box_i;
  box_i.y1 = static_cast<CalcT>(box_i_orig.y1);
  box_i.x1 = static_cast<CalcT>(box_i_orig.x1);
  box_i.y2 = static_cast<CalcT>(box_i_orig.y2);
  box_i.x2 = static_cast<CalcT>(box_i_orig.x2);

  BoxCornerEncoding<CalcT> box_j;
  box_j.y1 = static_cast<CalcT>(box_j_orig.y1);
  box_j.x1 = static_cast<CalcT>(box_j_orig.x1);
  box_j.y2 = static_cast<CalcT>(box_j_orig.y2);
  box_j.x2 = static_cast<CalcT>(box_j_orig.x2);

  const auto iou_operands = ComputeIntersectionAndUnion(box_i, box_j);

  auto intersection_area = iou_operands.first;
  auto union_area = iou_operands.second;

  CalcT iou_sym = 0;  // If union_area == 0, we return quantized 0.
  if (union_area != 0) {
    // Union and intersection area might need to be rescaled in order to avoid
    // overflow during the final computation (16 bit only).
    int leading_zeros = CountLeadingZeros(
        static_cast<std::make_unsigned<CalcT>::type>(union_area));
    int shift = std::max(17 - leading_zeros, 0);

    union_area = union_area >> shift;
    intersection_area = intersection_area >> shift;

    iou_sym = (intersection_area * params.iou_inverse_scale) / union_area;
  }

  return Truncate<T>(iou_sym + params.iou_zero_point);
}

template <typename T>
struct Candidate {
  int index;
  T score;
  int suppress_begin_index;

  constexpr bool operator<(const Candidate<T>& that) const {
    return this->score < that.score;
  }
};

auto GetCandidatePriorityQueue(const float* scores, const int num_boxes,
                               const float score_threshold) {
  std::priority_queue<Candidate<float>, std::deque<Candidate<float>>>
      candidate_priority_queue;

  // Populate queue with candidates above the score threshold.
  for (int i = 0; i < num_boxes; ++i) {
    if (scores[i] > score_threshold) {
      candidate_priority_queue.emplace(Candidate<float>({i, scores[i], 0}));
    }
  }

  return candidate_priority_queue;
}

template <typename T>
auto GetCandidatePriorityQueue(const NonMaxSuppressionParams& params,
                               const T* scores, const int num_boxes,
                               const T score_threshold) {
  std::priority_queue<Candidate<T>, std::deque<Candidate<T>>>
      candidate_priority_queue;

  // Populate queue with candidates above the score threshold.
  for (int i = 0; i < num_boxes; ++i) {
    const T score = Rescale<T>(
        scores[i], params.scores_zero_point, params.scores_rescale_multiplier,
        params.scores_rescale_shift, params.scores_rescale_zero_point);
    if (score > score_threshold) {
      candidate_priority_queue.emplace(Candidate<T>({i, score, 0}));
    }
  }

  return candidate_priority_queue;
}

}  // namespace

// Implements (Single-Class) Soft-NMS (with Gaussian weighting) with
// FLOAT32 inputs.
// Supports functionality of TensorFlow ops NonMaxSuppressionV4 & V5.
// Reference: "Soft-NMS - Improving Object Detection With One Line of Code"
//            [Bodla et al, https://arxiv.org/abs/1704.04503]
// Implementation adapted from the TensorFlow NMS code at
// tensorflow/core/kernels/non_max_suppression_op.cc.
//
// Arguments:
//  params: quantisation parameters (not used in the float implementation)
//  boxes: box encodings in format [y1, x1, y2, x2], shape: [num_boxes, 4]
//  num_boxes: number of candidates
//  scores: scores for candidate boxes, in the same order. shape: [num_boxes]
//  max_output_size: the maximum number of selections.
//  iou_threshold: Intersection-over-Union (IoU) threshold for NMS
//  score_threshold: All candidate scores below this value are rejected
//  soft_nms_sigma: Soft-NMS parameter, used for decaying scores
//
// Outputs:
//  selected_indices: all the selected indices. Underlying array must have
//    length >= max_output_size. Cannot be null.
//  selected_scores: scores of selected indices. Defer from original value for
//    Soft-NMS. If not null, array must have length >= max_output_size.
//  num_selected_indices: Number of selections. Only these many elements are
//    set in selected_indices, selected_scores. Cannot be null.
//
// Assumes inputs are valid (for eg, iou_threshold must be >= 0).
inline void NonMaxSuppression(const NonMaxSuppressionParams& params,
                              const float* boxes, const int num_boxes,
                              const float* scores, const int max_output_size,
                              const float iou_threshold,
                              const float score_threshold,
                              const float soft_nms_sigma, int* selected_indices,
                              float* selected_scores,
                              int* num_selected_indices) {
  auto candidate_priority_queue =
      GetCandidatePriorityQueue(scores, num_boxes, score_threshold);

  *num_selected_indices = 0;
  int num_outputs = std::min(static_cast<int>(candidate_priority_queue.size()),
                             max_output_size);
  if (num_outputs == 0) return;

  // NMS loop.
  const float scale = soft_nms_sigma > 0.f ? (-0.5f / soft_nms_sigma) : 0.f;
  while (*num_selected_indices < num_outputs &&
         !candidate_priority_queue.empty()) {
    Candidate<float> next_candidate = candidate_priority_queue.top();
    const float original_score = next_candidate.score;
    candidate_priority_queue.pop();

    // Overlapping boxes are likely to have similar scores, therefore we
    // iterate through the previously selected boxes backwards in order to
    // see if `next_candidate` should be suppressed. We also enforce a property
    // that a candidate can be suppressed by another candidate no more than
    // once via `suppress_begin_index` which tracks which previously selected
    // boxes have already been compared against next_candidate prior to a given
    // iteration.  These previous selected boxes are then skipped over in the
    // following loop.
    bool should_hard_suppress = false;
    for (int j = *num_selected_indices - 1;
         j >= next_candidate.suppress_begin_index; --j) {
      const float iou =
          ComputeIou(boxes, next_candidate.index, selected_indices[j]);

      // First decide whether to perform hard suppression.
      if (iou >= iou_threshold) {
        should_hard_suppress = true;
        break;
      }

      // Suppress score if NMS sigma > 0.
      if (soft_nms_sigma > 0.0) {
        next_candidate.score =
            next_candidate.score * std::exp(scale * iou * iou);
      }

      // If score has fallen below score_threshold, it won't be pushed back into
      // the queue.
      if (next_candidate.score <= score_threshold) break;
    }
    // If `next_candidate.score` has not dropped below `score_threshold`
    // by this point, then we know that we went through all of the previous
    // selections and can safely update `suppress_begin_index` to
    // `selected.size()`. If on the other hand `next_candidate.score`
    // *has* dropped below the score threshold, then since `suppress_weight`
    // always returns values in [0, 1], further suppression by items that were
    // not covered in the above for loop would not have caused the algorithm
    // to select this item. We thus do the same update to
    // `suppress_begin_index`, but really, this element will not be added back
    // into the priority queue.
    next_candidate.suppress_begin_index = *num_selected_indices;

    if (!should_hard_suppress) {
      if (next_candidate.score == original_score) {
        // Suppression has not occurred, so select next_candidate.
        selected_indices[*num_selected_indices] = next_candidate.index;
        if (selected_scores) {
          selected_scores[*num_selected_indices] = next_candidate.score;
        }
        ++*num_selected_indices;
      }
      if (next_candidate.score > score_threshold) {
        // Soft suppression might have occurred and current score is still
        // greater than score_threshold; add next_candidate back onto priority
        // queue.
        candidate_priority_queue.push(next_candidate);
      }
    }
  }
}

// Implements (Single-Class) Soft-NMS (with Gaussian weighting) with
// INT8 and INT16 quantised inputs.
// Supports functionality of TensorFlow ops NonMaxSuppressionV4 & V5.
// Reference: "Soft-NMS - Improving Object Detection With One Line of Code"
//            [Bodla et al, https://arxiv.org/abs/1704.04503]
// Implementation adapted from the TensorFlow NMS code at
// tensorflow/core/kernels/non_max_suppression_op.cc.
//
// Arguments:
//  params: quantisation parameters
//  boxes: box encodings in format [y1, x1, y2, x2], shape: [num_boxes, 4]
//  num_boxes: number of candidates
//  scores: scores for candidate boxes, in the same order. shape: [num_boxes]
//  max_output_size: the maximum number of selections.
//  iou_threshold: Intersection-over-Union (IoU) threshold for NMS
//  score_threshold: All candidate scores below this value are rejected
//  soft_nms_sigma: Soft-NMS parameter, used for decaying scores
//
// Outputs:
//  selected_indices: all the selected indices. Underlying array must have
//    length >= max_output_size. Cannot be null.
//  selected_scores: scores of selected indices. Defer from original value for
//    Soft-NMS. If not null, array must have length >= max_output_size.
//  num_selected_indices: Number of selections. Only these many elements are
//    set in selected_indices, selected_scores. Cannot be null.
//
// Assumes inputs are valid and have the same quantisation parameters.
template <typename T>
inline void NonMaxSuppression(const NonMaxSuppressionParams& params,
                              const T* boxes, const int num_boxes,
                              const T* scores, const int max_output_size,
                              const T iou_threshold, const T score_threshold,
                              const T soft_nms_sigma, int* selected_indices,
                              T* selected_scores, int* num_selected_indices) {
  // Rescale quantized IOU threshold to restricted scale.
  const T iou_threshold_rescaled =
      Rescale<T>(iou_threshold, params.iou_threshold_zero_point,
                 params.iou_threshold_rescale_multiplier,
                 params.iou_threshold_rescale_shift,
                 params.iou_threshold_rescale_zero_point);

  // Rescale quantized score threshold to restricted scale.
  const T score_threshold_rescaled =
      Rescale<T>(score_threshold, params.score_threshold_zero_point,
                 params.score_threshold_rescale_multiplier,
                 params.score_threshold_rescale_shift,
                 params.score_threshold_rescale_zero_point);

  auto candidate_priority_queue = GetCandidatePriorityQueue(
      params, scores, num_boxes, score_threshold_rescaled);

  *num_selected_indices = 0;
  int num_outputs = std::min(static_cast<int>(candidate_priority_queue.size()),
                             max_output_size);
  if (num_outputs == 0) return;

  // Obtain pointer to Soft-NMS LUT.
  const T* soft_nms_lut = static_cast<const T*>(params.soft_nms_lut);

  while (*num_selected_indices < num_outputs &&
         !candidate_priority_queue.empty()) {
    Candidate<T> next_candidate = candidate_priority_queue.top();

    const T original_score = next_candidate.score;
    candidate_priority_queue.pop();

    bool should_hard_suppress = false;
    for (int j = *num_selected_indices - 1;
         j >= next_candidate.suppress_begin_index; --j) {
      const T iou =
          ComputeIou(boxes, next_candidate.index, selected_indices[j], params);

      // First decide whether to perform hard suppression.
      if (iou >= iou_threshold_rescaled) {
        should_hard_suppress = true;
        break;
      }

      // Suppress score if Soft-NMS.
      if (soft_nms_lut != nullptr) {
        // Calculate new score using the similarity metric from the LUT.
        // Note thate we need to subtract zero_point in order to correctly
        // handle asymmetrically quantized values. This will be added back to
        // the final result at the end.
        const T similarity_metric = LUTLookup(iou, soft_nms_lut);

        const int32_t new_score = (static_cast<int32_t>(next_candidate.score) -
                                   params.scores_rescale_zero_point) *
                                  (static_cast<int32_t>(similarity_metric) -
                                   params.scores_rescale_zero_point);

        // Rescale new score to fit into quantized output range.
        const int32_t new_score_rescaled = MultiplyByQuantizedMultiplier(
            new_score, params.selected_scores_rescale_multiplier,
            params.selected_scores_rescale_shift);

        // Update score.
        next_candidate.score = Truncate<T>(
            new_score_rescaled + params.selected_scores_rescale_zero_point);
      }

      // If score has fallen below score_threshold, it won't be pushed back
      // into the queue.
      if (next_candidate.score <= score_threshold_rescaled) break;
    }

    next_candidate.suppress_begin_index = *num_selected_indices;

    if (!should_hard_suppress) {
      if (next_candidate.score == original_score) {
        // Suppression has not occurred, so select next_candidate.
        selected_indices[*num_selected_indices] = next_candidate.index;
        if (selected_scores) {
          selected_scores[*num_selected_indices] = next_candidate.score;
        }
        ++*num_selected_indices;
      }
      if (next_candidate.score > score_threshold_rescaled) {
        // Soft suppression might have occurred and current score is still
        // greater than score_threshold; add next_candidate back onto priority
        // queue.
        candidate_priority_queue.push(next_candidate);
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_NON_MAX_SUPPRESSION_H_
