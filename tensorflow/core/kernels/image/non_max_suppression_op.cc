/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/image_ops.cc

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/image/non_max_suppression_op.h"

#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

typedef Eigen::ThreadPoolDevice CPUDevice;

static inline void CheckScoreSizes(OpKernelContext* context, int num_boxes,
                                   const Tensor& scores) {
  // The shape of 'scores' is [num_boxes]
  OP_REQUIRES(context, scores.dims() == 1,
              errors::InvalidArgument(
                  "scores must be 1-D", scores.shape().DebugString(),
                  " (Shape must be rank 1 but is rank ", scores.dims(), ")"));
  OP_REQUIRES(
      context, scores.dim_size(0) == num_boxes,
      errors::InvalidArgument("scores has incompatible shape (Dimensions must "
                              "be equal, but are ",
                              num_boxes, " and ", scores.dim_size(0), ")"));
}

static inline void ParseAndCheckOverlapSizes(OpKernelContext* context,
                                             const Tensor& overlaps,
                                             int* num_boxes) {
  // the shape of 'overlaps' is [num_boxes, num_boxes]
  OP_REQUIRES(context, overlaps.dims() == 2,
              errors::InvalidArgument("overlaps must be 2-D",
                                      overlaps.shape().DebugString()));

  *num_boxes = overlaps.dim_size(0);
  OP_REQUIRES(context, overlaps.dim_size(1) == *num_boxes,
              errors::InvalidArgument("overlaps must be square",
                                      overlaps.shape().DebugString()));
}

static inline void ParseAndCheckBoxSizes(OpKernelContext* context,
                                         const Tensor& boxes, int* num_boxes) {
  // The shape of 'boxes' is [num_boxes, 4]
  OP_REQUIRES(context, boxes.dims() == 2,
              errors::InvalidArgument(
                  "boxes must be 2-D", boxes.shape().DebugString(),
                  " (Shape must be rank 2 but is rank ", boxes.dims(), ")"));
  *num_boxes = boxes.dim_size(0);
  OP_REQUIRES(context, boxes.dim_size(1) == 4,
              errors::InvalidArgument("boxes must have 4 columns (Dimension "
                                      "must be 4 but is ",
                                      boxes.dim_size(1), ")"));
}

static inline void CheckCombinedNMSScoreSizes(OpKernelContext* context,
                                              int num_boxes,
                                              const Tensor& scores) {
  // The shape of 'scores' is [batch_size, num_boxes, num_classes]
  OP_REQUIRES(context, scores.dims() == 3,
              errors::InvalidArgument("scores must be 3-D",
                                      scores.shape().DebugString()));
  OP_REQUIRES(context, scores.dim_size(1) == num_boxes,
              errors::InvalidArgument("scores has incompatible shape"));
}

static inline void ParseAndCheckCombinedNMSBoxSizes(OpKernelContext* context,
                                                    const Tensor& boxes,
                                                    int* num_boxes,
                                                    const int num_classes) {
  // The shape of 'boxes' is [batch_size, num_boxes, q, 4]
  OP_REQUIRES(context, boxes.dims() == 4,
              errors::InvalidArgument("boxes must be 4-D",
                                      boxes.shape().DebugString()));

  bool box_check = boxes.dim_size(2) == 1 || boxes.dim_size(2) == num_classes;
  OP_REQUIRES(context, box_check,
              errors::InvalidArgument(
                  "third dimension of boxes must be either 1 or num classes"));
  *num_boxes = boxes.dim_size(1);
  OP_REQUIRES(context, boxes.dim_size(3) == 4,
              errors::InvalidArgument("boxes must have 4 columns"));
}
// Return intersection-over-union overlap between boxes i and j
template <typename T>
static inline float IOU(typename TTypes<T, 2>::ConstTensor boxes, int i,
                        int j) {
  const float ymin_i = Eigen::numext::mini<float>(
      static_cast<float>(boxes(i, 0)), static_cast<float>(boxes(i, 2)));
  const float xmin_i = Eigen::numext::mini<float>(
      static_cast<float>(boxes(i, 1)), static_cast<float>(boxes(i, 3)));
  const float ymax_i = Eigen::numext::maxi<float>(
      static_cast<float>(boxes(i, 0)), static_cast<float>(boxes(i, 2)));
  const float xmax_i = Eigen::numext::maxi<float>(
      static_cast<float>(boxes(i, 1)), static_cast<float>(boxes(i, 3)));
  const float ymin_j = Eigen::numext::mini<float>(
      static_cast<float>(boxes(j, 0)), static_cast<float>(boxes(j, 2)));
  const float xmin_j = Eigen::numext::mini<float>(
      static_cast<float>(boxes(j, 1)), static_cast<float>(boxes(j, 3)));
  const float ymax_j = Eigen::numext::maxi<float>(
      static_cast<float>(boxes(j, 0)), static_cast<float>(boxes(j, 2)));
  const float xmax_j = Eigen::numext::maxi<float>(
      static_cast<float>(boxes(j, 1)), static_cast<float>(boxes(j, 3)));
  const float area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
  const float area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);
  if (area_i <= 0 || area_j <= 0) {
    return 0.0;
  }
  const float intersection_ymin = Eigen::numext::maxi<float>(ymin_i, ymin_j);
  const float intersection_xmin = Eigen::numext::maxi<float>(xmin_i, xmin_j);
  const float intersection_ymax = Eigen::numext::mini<float>(ymax_i, ymax_j);
  const float intersection_xmax = Eigen::numext::mini<float>(xmax_i, xmax_j);
  const float intersection_area =
      Eigen::numext::maxi<float>(intersection_ymax - intersection_ymin, 0.0) *
      Eigen::numext::maxi<float>(intersection_xmax - intersection_xmin, 0.0);
  return intersection_area / (area_i + area_j - intersection_area);
}

static inline float IOU(const float* boxes, int i, int j) {
  const float ymin_i = Eigen::numext::mini<float>(boxes[i], boxes[i + 2]);
  const float xmin_i = Eigen::numext::mini<float>(boxes[i + 1], boxes[i + 3]);
  const float ymax_i = Eigen::numext::maxi<float>(boxes[i], boxes[i + 2]);
  const float xmax_i = Eigen::numext::maxi<float>(boxes[i + 1], boxes[i + 3]);
  const float ymin_j = Eigen::numext::mini<float>(boxes[j], boxes[j + 2]);
  const float xmin_j = Eigen::numext::mini<float>(boxes[j + 1], boxes[j + 3]);
  const float ymax_j = Eigen::numext::maxi<float>(boxes[j], boxes[j + 2]);
  const float xmax_j = Eigen::numext::maxi<float>(boxes[j + 1], boxes[j + 3]);
  const float area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
  const float area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);
  if (area_i <= 0 || area_j <= 0) {
    return 0.0;
  }
  const float intersection_ymin = Eigen::numext::maxi<float>(ymin_i, ymin_j);
  const float intersection_xmin = Eigen::numext::maxi<float>(xmin_i, xmin_j);
  const float intersection_ymax = Eigen::numext::mini<float>(ymax_i, ymax_j);
  const float intersection_xmax = Eigen::numext::mini<float>(xmax_i, xmax_j);
  const float intersection_area =
      Eigen::numext::maxi<float>(intersection_ymax - intersection_ymin, 0.0) *
      Eigen::numext::maxi<float>(intersection_xmax - intersection_xmin, 0.0);
  return intersection_area / (area_i + area_j - intersection_area);
}

template <typename T>
static inline T Overlap(typename TTypes<T, 2>::ConstTensor overlaps, int i,
                        int j) {
  return overlaps(i, j);
}

template <typename T>
static inline std::function<float(int, int)> CreateIOUSimilarityFn(
    const Tensor& boxes) {
  typename TTypes<T, 2>::ConstTensor boxes_data = boxes.tensor<T, 2>();
  return std::bind(&IOU<T>, boxes_data, std::placeholders::_1,
                   std::placeholders::_2);
}

template <typename T>
static inline std::function<T(int, int)> CreateOverlapSimilarityFn(
    const Tensor& overlaps) {
  typename TTypes<T, 2>::ConstTensor overlaps_data =
      overlaps.tensor<float, 2>();
  return std::bind(&Overlap<T>, overlaps_data, std::placeholders::_1,
                   std::placeholders::_2);
}

template <typename T>
void DoNonMaxSuppressionOp(OpKernelContext* context, const Tensor& scores,
                           int num_boxes, const Tensor& max_output_size,
                           const T similarity_threshold,
                           const T score_threshold, const T soft_nms_sigma,
                           const std::function<float(int, int)>& similarity_fn,
                           bool return_scores_tensor = false,
                           bool pad_to_max_output_size = false,
                           int* ptr_num_valid_outputs = nullptr) {
  const int output_size = max_output_size.scalar<int>()();
  OP_REQUIRES(context, output_size >= 0,
              errors::InvalidArgument("output size must be non-negative"));

  std::vector<T> scores_data(num_boxes);
  std::copy_n(scores.flat<T>().data(), num_boxes, scores_data.begin());

  // Data structure for a selection candidate in NMS.
  struct Candidate {
    int box_index;
    T score;
    int suppress_begin_index;
  };

  auto cmp = [](const Candidate bs_i, const Candidate bs_j) {
    return ((bs_i.score == bs_j.score) && (bs_i.box_index > bs_j.box_index)) ||
           bs_i.score < bs_j.score;
  };
  std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)>
      candidate_priority_queue(cmp);
  for (int i = 0; i < scores_data.size(); ++i) {
    if (scores_data[i] > score_threshold) {
      candidate_priority_queue.emplace(Candidate({i, scores_data[i], 0}));
    }
  }

  T scale = static_cast<T>(0.0);
  bool is_soft_nms = soft_nms_sigma > static_cast<T>(0.0);
  if (is_soft_nms) {
    scale = static_cast<T>(-0.5) / soft_nms_sigma;
  }

  auto suppress_weight = [similarity_threshold, scale,
                          is_soft_nms](const T sim) {
    const T weight = Eigen::numext::exp<T>(scale * sim * sim);
    return is_soft_nms || sim <= similarity_threshold ? weight
                                                      : static_cast<T>(0.0);
  };

  std::vector<int> selected;
  std::vector<T> selected_scores;
  float similarity;
  T original_score;
  Candidate next_candidate;

  while (selected.size() < output_size && !candidate_priority_queue.empty()) {
    next_candidate = candidate_priority_queue.top();
    original_score = next_candidate.score;
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
    for (int j = static_cast<int>(selected.size()) - 1;
         j >= next_candidate.suppress_begin_index; --j) {
      similarity = similarity_fn(next_candidate.box_index, selected[j]);

      next_candidate.score *= suppress_weight(static_cast<T>(similarity));

      // First decide whether to perform hard suppression
      if (!is_soft_nms && static_cast<T>(similarity) > similarity_threshold) {
        should_hard_suppress = true;
        break;
      }

      // If next_candidate survives hard suppression, apply soft suppression
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
    // into the priority queue in the following.
    next_candidate.suppress_begin_index = selected.size();

    if (!should_hard_suppress) {
      if (next_candidate.score == original_score) {
        // Suppression has not occurred, so select next_candidate
        selected.push_back(next_candidate.box_index);
        selected_scores.push_back(next_candidate.score);
        continue;
      }
      if (next_candidate.score > score_threshold) {
        // Soft suppression has occurred and current score is still greater than
        // score_threshold; add next_candidate back onto priority queue.
        candidate_priority_queue.push(next_candidate);
      }
    }
  }

  int num_valid_outputs = selected.size();
  if (pad_to_max_output_size) {
    selected.resize(output_size, 0);
    selected_scores.resize(output_size, static_cast<T>(0));
  }
  if (ptr_num_valid_outputs) {
    *ptr_num_valid_outputs = num_valid_outputs;
  }

  // Allocate output tensors
  Tensor* output_indices = nullptr;
  TensorShape output_shape({static_cast<int>(selected.size())});
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, output_shape, &output_indices));
  TTypes<int, 1>::Tensor output_indices_data = output_indices->tensor<int, 1>();
  std::copy_n(selected.begin(), selected.size(), output_indices_data.data());

  if (return_scores_tensor) {
    Tensor* output_scores = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, output_shape, &output_scores));
    typename TTypes<T, 1>::Tensor output_scores_data =
        output_scores->tensor<T, 1>();
    std::copy_n(selected_scores.begin(), selected_scores.size(),
                output_scores_data.data());
  }
}

struct ResultCandidate {
  int box_index;
  float score;
  int class_idx;
  float box_coord[4];
};

void DoNMSPerClass(int batch_idx, int class_idx, const float* boxes_data,
                   const float* scores_data, int num_boxes, int q,
                   int num_classes, const int size_per_class,
                   const float score_threshold, const float iou_threshold,
                   std::vector<ResultCandidate>& result_candidate_vec) {
  // Do NMS, get the candidate indices of form vector<int>
  // Data structure for selection candidate in NMS.
  struct Candidate {
    int box_index;
    float score;
  };
  auto cmp = [](const Candidate bs_i, const Candidate bs_j) {
    return bs_i.score < bs_j.score;
  };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp)>
      candidate_priority_queue(cmp);
  float temp_score;
  for (int i = 0; i < num_boxes; ++i) {
    temp_score = scores_data[i * num_classes + class_idx];
    if (temp_score > score_threshold) {
      candidate_priority_queue.emplace(Candidate({i, temp_score}));
    }
  }

  std::vector<int> selected;
  Candidate next_candidate;

  int candidate_box_data_idx, selected_box_data_idx, class_box_idx;
  class_box_idx = (q > 1) ? class_idx : 0;

  float iou;
  while (selected.size() < size_per_class &&
         !candidate_priority_queue.empty()) {
    next_candidate = candidate_priority_queue.top();
    candidate_priority_queue.pop();

    candidate_box_data_idx = (next_candidate.box_index * q + class_box_idx) * 4;

    // Overlapping boxes are likely to have similar scores,
    // therefore we iterate through the previously selected boxes backwards
    // in order to see if `next_candidate` should be suppressed.
    bool should_select = true;
    for (int j = selected.size() - 1; j >= 0; --j) {
      selected_box_data_idx = (selected[j] * q + class_box_idx) * 4;
      iou = IOU(boxes_data, candidate_box_data_idx, selected_box_data_idx);
      if (iou > iou_threshold) {
        should_select = false;
        break;
      }
    }

    if (should_select) {
      // Add the selected box to the result candidate. Sorted by score
      result_candidate_vec[selected.size() + size_per_class * class_idx] = {
          next_candidate.box_index,
          next_candidate.score,
          class_idx,
          {boxes_data[candidate_box_data_idx],
           boxes_data[candidate_box_data_idx + 1],
           boxes_data[candidate_box_data_idx + 2],
           boxes_data[candidate_box_data_idx + 3]}};
      selected.push_back(next_candidate.box_index);
    }
  }
}

void SelectResultPerBatch(std::vector<float>& nmsed_boxes,
                          std::vector<float>& nmsed_scores,
                          std::vector<float>& nmsed_classes,
                          std::vector<ResultCandidate>& result_candidate_vec,
                          std::vector<int>& final_valid_detections,
                          const int batch_idx, int total_size_per_batch,
                          bool pad_per_class, int max_size_per_batch,
                          bool clip_boxes, int per_batch_size) {
  auto rc_cmp = [](const ResultCandidate rc_i, const ResultCandidate rc_j) {
    return rc_i.score > rc_j.score;
  };
  std::sort(result_candidate_vec.begin(), result_candidate_vec.end(), rc_cmp);

  int max_detections = 0;
  int result_candidate_size =
      std::count_if(result_candidate_vec.begin(), result_candidate_vec.end(),
                    [](ResultCandidate rc) { return rc.box_index > -1; });
  // If pad_per_class is false, we always pad to max_total_size
  if (!pad_per_class) {
    max_detections = std::min(result_candidate_size, total_size_per_batch);
  } else {
    max_detections = std::min(per_batch_size, result_candidate_size);
  }

  final_valid_detections[batch_idx] = max_detections;

  int curr_total_size = max_detections;
  int result_idx = 0;
  // Pick the top max_detections values
  while (curr_total_size > 0 && result_idx < result_candidate_vec.size()) {
    ResultCandidate next_candidate = result_candidate_vec[result_idx++];
    // Add to final output vectors
    if (clip_boxes) {
      const float box_min = 0.0;
      const float box_max = 1.0;
      nmsed_boxes.push_back(
          std::max(std::min(next_candidate.box_coord[0], box_max), box_min));
      nmsed_boxes.push_back(
          std::max(std::min(next_candidate.box_coord[1], box_max), box_min));
      nmsed_boxes.push_back(
          std::max(std::min(next_candidate.box_coord[2], box_max), box_min));
      nmsed_boxes.push_back(
          std::max(std::min(next_candidate.box_coord[3], box_max), box_min));
    } else {
      nmsed_boxes.push_back(next_candidate.box_coord[0]);
      nmsed_boxes.push_back(next_candidate.box_coord[1]);
      nmsed_boxes.push_back(next_candidate.box_coord[2]);
      nmsed_boxes.push_back(next_candidate.box_coord[3]);
    }
    nmsed_scores.push_back(next_candidate.score);
    nmsed_classes.push_back(next_candidate.class_idx);
    curr_total_size--;
  }

  nmsed_boxes.resize(per_batch_size * 4, 0);
  nmsed_scores.resize(per_batch_size, 0);
  nmsed_classes.resize(per_batch_size, 0);
}

void BatchedNonMaxSuppressionOp(
    OpKernelContext* context, const Tensor& inp_boxes, const Tensor& inp_scores,
    int num_boxes, const int max_size_per_class, const int total_size_per_batch,
    const float score_threshold, const float iou_threshold,
    bool pad_per_class = false, bool clip_boxes = true) {
  const int num_batches = inp_boxes.dim_size(0);
  int num_classes = inp_scores.dim_size(2);
  int q = inp_boxes.dim_size(2);

  const float* scores_data =
      const_cast<float*>(inp_scores.flat<float>().data());
  const float* boxes_data = const_cast<float*>(inp_boxes.flat<float>().data());

  int boxes_per_batch = num_boxes * q * 4;
  int scores_per_batch = num_boxes * num_classes;
  const int size_per_class = std::min(max_size_per_class, num_boxes);
  std::vector<std::vector<ResultCandidate>> result_candidate_vec(
      num_batches,
      std::vector<ResultCandidate>(size_per_class * num_classes,
                                   {-1, -1.0, -1, {0.0, 0.0, 0.0, 0.0}}));

  // [num_batches, per_batch_size * 4]
  std::vector<std::vector<float>> nmsed_boxes(num_batches);
  // [num_batches, per_batch_size]
  std::vector<std::vector<float>> nmsed_scores(num_batches);
  // [num_batches, per_batch_size]
  std::vector<std::vector<float>> nmsed_classes(num_batches);
  // [num_batches]
  std::vector<int> final_valid_detections(num_batches);

  auto shard_nms = [&](int begin, int end) {
    for (int idx = begin; idx < end; ++idx) {
      int batch_idx = idx / num_classes;
      int class_idx = idx % num_classes;
      DoNMSPerClass(batch_idx, class_idx,
                    boxes_data + boxes_per_batch * batch_idx,
                    scores_data + scores_per_batch * batch_idx, num_boxes, q,
                    num_classes, size_per_class, score_threshold, iou_threshold,
                    result_candidate_vec[batch_idx]);
    }
  };

  int length = num_batches * num_classes;
  // Input data boxes_data, scores_data
  int input_bytes = num_boxes * 10 * sizeof(float);
  int output_bytes = num_boxes * 10 * sizeof(float);
  int compute_cycles = Eigen::TensorOpCost::AddCost<int>() * num_boxes * 14 +
                       Eigen::TensorOpCost::MulCost<int>() * num_boxes * 9 +
                       Eigen::TensorOpCost::MulCost<float>() * num_boxes * 9 +
                       Eigen::TensorOpCost::AddCost<float>() * num_boxes * 8;
  // The cost here is not the actual number of cycles, but rather a set of
  // hand-tuned numbers that seem to work best.
  const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);
  const CPUDevice& d = context->eigen_device<CPUDevice>();
  d.parallelFor(length, cost, shard_nms);

  int per_batch_size = total_size_per_batch;
  // Avoid overflow.
  int max_total_size = static_cast<int>(
      std::min(static_cast<int64_t>(std::numeric_limits<int>::max()),
               static_cast<int64_t>(max_size_per_class) * num_classes));
  if (pad_per_class) {
    per_batch_size = std::min(total_size_per_batch, max_total_size);
  }

  Tensor* valid_detections_t = nullptr;
  TensorShape valid_detections_shape({num_batches});
  OP_REQUIRES_OK(context, context->allocate_output(3, valid_detections_shape,
                                                   &valid_detections_t));
  auto valid_detections_flat = valid_detections_t->template flat<int>();

  auto shard_result = [&](int begin, int end) {
    for (int batch_idx = begin; batch_idx < end; ++batch_idx) {
      SelectResultPerBatch(
          nmsed_boxes[batch_idx], nmsed_scores[batch_idx],
          nmsed_classes[batch_idx], result_candidate_vec[batch_idx],
          final_valid_detections, batch_idx, total_size_per_batch,
          pad_per_class, max_total_size, clip_boxes, per_batch_size);
      valid_detections_flat(batch_idx) = final_valid_detections[batch_idx];
    }
  };
  length = num_batches;
  // Input data boxes_data, scores_data
  input_bytes =
      num_boxes * 10 * sizeof(float) + per_batch_size * 6 * sizeof(float);
  output_bytes =
      num_boxes * 5 * sizeof(float) + per_batch_size * 6 * sizeof(float);
  compute_cycles = Eigen::TensorOpCost::AddCost<int>() * num_boxes * 5 +
                   Eigen::TensorOpCost::AddCost<float>() * num_boxes * 5;
  // The cost here is not the actual number of cycles, but rather a set of
  // hand-tuned numbers that seem to work best.
  const Eigen::TensorOpCost cost_result(input_bytes, output_bytes,
                                        compute_cycles);
  d.parallelFor(length, cost_result, shard_result);

  Tensor* nmsed_boxes_t = nullptr;
  TensorShape boxes_shape({num_batches, per_batch_size, 4});
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, boxes_shape, &nmsed_boxes_t));
  auto nmsed_boxes_flat = nmsed_boxes_t->template flat<float>();

  Tensor* nmsed_scores_t = nullptr;
  TensorShape scores_shape({num_batches, per_batch_size});
  OP_REQUIRES_OK(context,
                 context->allocate_output(1, scores_shape, &nmsed_scores_t));
  auto nmsed_scores_flat = nmsed_scores_t->template flat<float>();

  Tensor* nmsed_classes_t = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(2, scores_shape, &nmsed_classes_t));
  auto nmsed_classes_flat = nmsed_classes_t->template flat<float>();

  auto shard_copy_result = [&](int begin, int end) {
    for (int idx = begin; idx < end; ++idx) {
      int batch_idx = idx / per_batch_size;
      int j = idx % per_batch_size;
      nmsed_scores_flat(idx) = nmsed_scores[batch_idx][j];
      nmsed_classes_flat(idx) = nmsed_classes[batch_idx][j];
      for (int k = 0; k < 4; ++k) {
        nmsed_boxes_flat(idx * 4 + k) = nmsed_boxes[batch_idx][j * 4 + k];
      }
    }
  };
  length = num_batches * per_batch_size;
  // Input data boxes_data, scores_data
  input_bytes = 6 * sizeof(float);
  output_bytes = 6 * sizeof(float);
  compute_cycles = Eigen::TensorOpCost::AddCost<int>() * 2 +
                   Eigen::TensorOpCost::MulCost<int>() * 2 +
                   Eigen::TensorOpCost::DivCost<float>() * 2;
  const Eigen::TensorOpCost cost_copy_result(input_bytes, output_bytes,
                                             compute_cycles);
  d.parallelFor(length, cost_copy_result, shard_copy_result);
}

// Extract a scalar of type T from a tensor, with correct type checking.
// This is necessary because several of the kernels here assume
// T == T_threshold.
template <typename T>
T GetScalar(const Tensor& tensor) {
  switch (tensor.dtype()) {
    case DT_FLOAT:
      return static_cast<T>(tensor.scalar<float>()());
    case DT_DOUBLE:
      return static_cast<T>(tensor.scalar<double>()());
    case DT_BFLOAT16:
      return static_cast<T>(tensor.scalar<Eigen::bfloat16>()());
    case DT_HALF:
      return static_cast<T>(tensor.scalar<Eigen::half>()());
    default:
      DCHECK(false) << "Unsupported type " << tensor.dtype();
      break;
  }
  return static_cast<T>(0);
}

}  // namespace

template <typename Device>
class NonMaxSuppressionOp : public OpKernel {
 public:
  explicit NonMaxSuppressionOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("iou_threshold", &iou_threshold_));
  }

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));

    OP_REQUIRES(context, iou_threshold_ >= 0 && iou_threshold_ <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }
    auto similarity_fn = CreateIOUSimilarityFn<float>(boxes);

    const float score_threshold_val = std::numeric_limits<float>::lowest();
    const float dummy_soft_nms_sigma = static_cast<float>(0.0);
    DoNonMaxSuppressionOp<float>(context, scores, num_boxes, max_output_size,
                                 iou_threshold_, score_threshold_val,
                                 dummy_soft_nms_sigma, similarity_fn);
  }

 private:
  float iou_threshold_;
};

template <typename Device, typename T>
class NonMaxSuppressionV2Op : public OpKernel {
 public:
  explicit NonMaxSuppressionV2Op(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const T iou_threshold_val = GetScalar<T>(iou_threshold);

    OP_REQUIRES(context,
                iou_threshold_val >= static_cast<T>(0.0) &&
                    iou_threshold_val <= static_cast<T>(1.0),
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }
    auto similarity_fn = CreateIOUSimilarityFn<T>(boxes);

    const T score_threshold_val = std::numeric_limits<T>::lowest();
    const T dummy_soft_nms_sigma = static_cast<T>(0.0);
    DoNonMaxSuppressionOp<T>(context, scores, num_boxes, max_output_size,
                             iou_threshold_val, score_threshold_val,
                             dummy_soft_nms_sigma, similarity_fn);
  }
};

template <typename Device, typename T>
class NonMaxSuppressionV3Op : public OpKernel {
 public:
  explicit NonMaxSuppressionV3Op(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString(),
                                " (Shape must be rank 0 but is ", "rank ",
                                max_output_size.dims(), ")"));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString(),
                                        " (Shape must be rank 0 but is rank ",
                                        iou_threshold.dims(), ")"));
    const T iou_threshold_val = GetScalar<T>(iou_threshold);
    OP_REQUIRES(context,
                iou_threshold_val >= static_cast<T>(0.0) &&
                    iou_threshold_val <= static_cast<T>(1.0),
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const T score_threshold_val = GetScalar<T>(score_threshold);

    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }

    auto similarity_fn = CreateIOUSimilarityFn<T>(boxes);

    const T dummy_soft_nms_sigma = static_cast<T>(0.0);
    DoNonMaxSuppressionOp<T>(context, scores, num_boxes, max_output_size,
                             iou_threshold_val, score_threshold_val,
                             dummy_soft_nms_sigma, similarity_fn);
  }
};

template <typename Device, typename T>
class NonMaxSuppressionV4Op : public OpKernel {
 public:
  explicit NonMaxSuppressionV4Op(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pad_to_max_output_size",
                                             &pad_to_max_output_size_));
  }

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const T iou_threshold_val = GetScalar<T>(iou_threshold);
    OP_REQUIRES(context,
                iou_threshold_val >= static_cast<T>(0.0) &&
                    iou_threshold_val <= static_cast<T>(1.0),
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const T score_threshold_val = GetScalar<T>(score_threshold);

    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }

    auto similarity_fn = CreateIOUSimilarityFn<T>(boxes);
    int num_valid_outputs;

    bool return_scores_tensor_ = false;
    const T dummy_soft_nms_sigma = static_cast<T>(0.0);
    DoNonMaxSuppressionOp<T>(
        context, scores, num_boxes, max_output_size, iou_threshold_val,
        score_threshold_val, dummy_soft_nms_sigma, similarity_fn,
        return_scores_tensor_, pad_to_max_output_size_, &num_valid_outputs);
    if (!context->status().ok()) {
      return;
    }

    // Allocate scalar output tensor for number of indices computed.
    Tensor* num_outputs_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, tensorflow::TensorShape{}, &num_outputs_t));
    num_outputs_t->scalar<int32>().setConstant(num_valid_outputs);
  }

 private:
  bool pad_to_max_output_size_;
};

template <typename Device, typename T>
class NonMaxSuppressionV5Op : public OpKernel {
 public:
  explicit NonMaxSuppressionV5Op(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pad_to_max_output_size",
                                             &pad_to_max_output_size_));
  }

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 4]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const T iou_threshold_val = iou_threshold.scalar<T>()();
    OP_REQUIRES(context,
                iou_threshold_val >= static_cast<T>(0.0) &&
                    iou_threshold_val <= static_cast<T>(1.0),
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const T score_threshold_val = score_threshold.scalar<T>()();

    // soft_nms_sigma: scalar
    const Tensor& soft_nms_sigma = context->input(5);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(soft_nms_sigma.shape()),
        errors::InvalidArgument("soft_nms_sigma must be 0-D, got shape ",
                                soft_nms_sigma.shape().DebugString()));
    const T soft_nms_sigma_val = soft_nms_sigma.scalar<T>()();
    OP_REQUIRES(context, soft_nms_sigma_val >= static_cast<T>(0.0),
                errors::InvalidArgument("soft_nms_sigma_val must be >= 0"));

    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }

    auto similarity_fn = CreateIOUSimilarityFn<T>(boxes);
    int num_valid_outputs;

    // For NonMaxSuppressionV5Op, we always return a second output holding
    // corresponding scores, so `return_scores_tensor` should never be false.
    const bool return_scores_tensor_ = true;
    DoNonMaxSuppressionOp<T>(
        context, scores, num_boxes, max_output_size, iou_threshold_val,
        score_threshold_val, soft_nms_sigma_val, similarity_fn,
        return_scores_tensor_, pad_to_max_output_size_, &num_valid_outputs);
    if (!context->status().ok()) {
      return;
    }

    // Allocate scalar output tensor for number of indices computed.
    Tensor* num_outputs_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                2, tensorflow::TensorShape{}, &num_outputs_t));
    num_outputs_t->scalar<int32>().setConstant(num_valid_outputs);
  }

 private:
  bool pad_to_max_output_size_;
};

template <typename Device>
class NonMaxSuppressionWithOverlapsOp : public OpKernel {
 public:
  explicit NonMaxSuppressionWithOverlapsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // overlaps: [num_boxes, num_boxes]
    const Tensor& overlaps = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // overlap_threshold: scalar
    const Tensor& overlap_threshold = context->input(3);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(overlap_threshold.shape()),
        errors::InvalidArgument("overlap_threshold must be 0-D, got shape ",
                                overlap_threshold.shape().DebugString()));
    const float overlap_threshold_val = overlap_threshold.scalar<float>()();

    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    int num_boxes = 0;
    ParseAndCheckOverlapSizes(context, overlaps, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }
    auto similarity_fn = CreateOverlapSimilarityFn<float>(overlaps);

    const float dummy_soft_nms_sigma = static_cast<float>(0.0);
    DoNonMaxSuppressionOp<float>(context, scores, num_boxes, max_output_size,
                                 overlap_threshold_val, score_threshold_val,
                                 dummy_soft_nms_sigma, similarity_fn);
  }
};

template <typename Device>
class CombinedNonMaxSuppressionOp : public OpKernel {
 public:
  explicit CombinedNonMaxSuppressionOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("pad_per_class", &pad_per_class_));
    OP_REQUIRES_OK(context, context->GetAttr("clip_boxes", &clip_boxes_));
  }

  void Compute(OpKernelContext* context) override {
    // boxes: [batch_size, num_anchors, q, 4]
    const Tensor& boxes = context->input(0);
    // scores: [batch_size, num_anchors, num_classes]
    const Tensor& scores = context->input(1);
    OP_REQUIRES(
        context, (boxes.dim_size(0) == scores.dim_size(0)),
        errors::InvalidArgument("boxes and scores must have same batch size"));

    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_size_per_class must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    const int max_size_per_class = max_output_size.scalar<int>()();
    OP_REQUIRES(context, max_size_per_class > 0,
                errors::InvalidArgument("max_size_per_class must be positive"));
    // max_total_size: scalar
    const Tensor& max_total_size = context->input(3);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_total_size.shape()),
        errors::InvalidArgument("max_total_size must be 0-D, got shape ",
                                max_total_size.shape().DebugString()));
    const int max_total_size_per_batch = max_total_size.scalar<int>()();
    OP_REQUIRES(context, max_total_size_per_batch > 0,
                errors::InvalidArgument("max_total_size must be > 0"));
    // Throw warning when `max_total_size` is too large as it may cause OOM.
    if (max_total_size_per_batch > pow(10, 6)) {
      LOG(WARNING) << "Detected a large value for `max_total_size`. This may "
                   << "cause OOM error. (max_total_size: "
                   << max_total_size.scalar<int>()() << ")";
    }
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(4);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();

    // score_threshold: scalar
    const Tensor& score_threshold = context->input(5);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    int num_boxes = 0;
    const int num_classes = scores.dim_size(2);
    ParseAndCheckCombinedNMSBoxSizes(context, boxes, &num_boxes, num_classes);
    CheckCombinedNMSScoreSizes(context, num_boxes, scores);

    if (!context->status().ok()) {
      return;
    }
    BatchedNonMaxSuppressionOp(context, boxes, scores, num_boxes,
                               max_size_per_class, max_total_size_per_batch,
                               score_threshold_val, iou_threshold_val,
                               pad_per_class_, clip_boxes_);
  }

 private:
  bool pad_per_class_;
  bool clip_boxes_;
};

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppression").Device(DEVICE_CPU),
                        NonMaxSuppressionOp<CPUDevice>);

REGISTER_KERNEL_BUILDER(
    Name("NonMaxSuppressionV2").TypeConstraint<float>("T").Device(DEVICE_CPU),
    NonMaxSuppressionV2Op<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV2")
                            .TypeConstraint<Eigen::half>("T")
                            .Device(DEVICE_CPU),
                        NonMaxSuppressionV2Op<CPUDevice, Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("NonMaxSuppressionV3").TypeConstraint<float>("T").Device(DEVICE_CPU),
    NonMaxSuppressionV3Op<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV3")
                            .TypeConstraint<Eigen::half>("T")
                            .Device(DEVICE_CPU),
                        NonMaxSuppressionV3Op<CPUDevice, Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("NonMaxSuppressionV4").TypeConstraint<float>("T").Device(DEVICE_CPU),
    NonMaxSuppressionV4Op<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV4")
                            .TypeConstraint<Eigen::half>("T")
                            .Device(DEVICE_CPU),
                        NonMaxSuppressionV4Op<CPUDevice, Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("NonMaxSuppressionV5").TypeConstraint<float>("T").Device(DEVICE_CPU),
    NonMaxSuppressionV5Op<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV5")
                            .TypeConstraint<Eigen::half>("T")
                            .Device(DEVICE_CPU),
                        NonMaxSuppressionV5Op<CPUDevice, Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("NonMaxSuppressionWithOverlaps").Device(DEVICE_CPU),
    NonMaxSuppressionWithOverlapsOp<CPUDevice>);

REGISTER_KERNEL_BUILDER(Name("CombinedNonMaxSuppression").Device(DEVICE_CPU),
                        CombinedNonMaxSuppressionOp<CPUDevice>);

}  // namespace tensorflow
