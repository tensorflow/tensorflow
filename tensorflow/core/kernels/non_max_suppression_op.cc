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

#include "tensorflow/core/kernels/non_max_suppression_op.h"

#include <cmath>
#include <functional>
#include <queue>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
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
              errors::InvalidArgument("scores must be 1-D",
                                      scores.shape().DebugString()));
  OP_REQUIRES(context, scores.dim_size(0) == num_boxes,
              errors::InvalidArgument("scores has incompatible shape"));
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
              errors::InvalidArgument("boxes must be 2-D",
                                      boxes.shape().DebugString()));
  *num_boxes = boxes.dim_size(0);
  OP_REQUIRES(context, boxes.dim_size(1) == 4,
              errors::InvalidArgument("boxes must have 4 columns"));
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
static inline T IOU(typename TTypes<T, 2>::ConstTensor boxes, int i, int j) {
  const T ymin_i = std::min<T>(boxes(i, 0), boxes(i, 2));
  const T xmin_i = std::min<T>(boxes(i, 1), boxes(i, 3));
  const T ymax_i = std::max<T>(boxes(i, 0), boxes(i, 2));
  const T xmax_i = std::max<T>(boxes(i, 1), boxes(i, 3));
  const T ymin_j = std::min<T>(boxes(j, 0), boxes(j, 2));
  const T xmin_j = std::min<T>(boxes(j, 1), boxes(j, 3));
  const T ymax_j = std::max<T>(boxes(j, 0), boxes(j, 2));
  const T xmax_j = std::max<T>(boxes(j, 1), boxes(j, 3));
  const T area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
  const T area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);
  if (area_i <= static_cast<T>(0) || area_j <= static_cast<T>(0)) {
    return static_cast<T>(0.0);
  }
  const T intersection_ymin = std::max<T>(ymin_i, ymin_j);
  const T intersection_xmin = std::max<T>(xmin_i, xmin_j);
  const T intersection_ymax = std::min<T>(ymax_i, ymax_j);
  const T intersection_xmax = std::min<T>(xmax_i, xmax_j);
  const T intersection_area =
      std::max<T>(intersection_ymax - intersection_ymin, static_cast<T>(0.0)) *
      std::max<T>(intersection_xmax - intersection_xmin, static_cast<T>(0.0));
  return intersection_area / (area_i + area_j - intersection_area);
}

template <typename T>
static inline T Overlap(typename TTypes<T, 2>::ConstTensor overlaps, int i,
                        int j) {
  return overlaps(i, j);
}

template <typename T>
static inline std::function<T(int, int)> CreateIOUSimilarityFn(
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
                           const std::function<T(int, int)>& similarity_fn,
                           bool return_scores_tensor = false,
                           bool pad_to_max_output_size = false,
                           int* ptr_num_valid_outputs = nullptr) {
  const int output_size = max_output_size.scalar<int>()();

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
  if (soft_nms_sigma > static_cast<T>(0.0)) {
    scale = static_cast<T>(-0.5) / soft_nms_sigma;
  }

  auto suppress_weight = [similarity_threshold, scale](const T sim) {
    const T weight =
        static_cast<T>(std::exp(static_cast<float>(scale * sim * sim)));
    return sim <= similarity_threshold ? weight : static_cast<T>(0.0);
  };

  std::vector<int> selected;
  std::vector<T> selected_scores;
  T similarity, original_score;
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

      next_candidate.score *= suppress_weight(similarity);

      // First decide whether to perform hard suppression
      if (similarity >= static_cast<T>(similarity_threshold)) {
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

void BatchedNonMaxSuppressionOp(
    OpKernelContext* context, const Tensor& inp_boxes, const Tensor& inp_scores,
    int num_boxes, const int max_size_per_class, const int total_size_per_batch,
    const float score_threshold, const float iou_threshold,
    bool pad_per_class = false, bool clip_boxes = true) {
  int q = inp_boxes.dim_size(2);
  int num_classes = inp_scores.dim_size(2);
  const int num_batches = inp_boxes.dim_size(0);

  // [num_batches, per_batch_size * 4]
  std::vector<std::vector<float>> nmsed_boxes(num_batches);
  // [num_batches, per_batch_size]
  std::vector<std::vector<float>> nmsed_scores(num_batches);
  // [num_batches, per_batch_size]
  std::vector<std::vector<float>> nmsed_classes(num_batches);
  // [num_batches]
  std::vector<int> final_valid_detections;

  int per_batch_size = total_size_per_batch;

  // perform non_max_suppression operation for each batch independently
  for (int batch = 0; batch < num_batches; ++batch) {
    // dims of per_batch_boxes [num_boxes, q, 4]
    Tensor per_batch_boxes = inp_boxes.Slice(batch, batch + 1);
    // dims of per_batch_scores [num_boxes, num_classes]
    Tensor per_batch_scores = inp_scores.Slice(batch, batch + 1);

    struct ResultCandidate {
      int box_index;
      float score;
      int class_idx;
      float box_coord[4];
    };

    std::vector<ResultCandidate> result_candidate_vec;

    float* scores_data = per_batch_scores.unaligned_flat<float>().data();
    float* boxes_data = per_batch_boxes.unaligned_flat<float>().data();

    // Iterate through all classes
    for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
      std::vector<float> class_scores_data;
      class_scores_data.reserve(num_boxes);
      std::vector<float> class_boxes_data;
      class_boxes_data.reserve(num_boxes * 4);

      for (int box = 0; box < num_boxes; ++box) {
        // Get the scores per class
        // class_scores_data dim is [num_boxes].
        class_scores_data.push_back(scores_data[box * num_classes + class_idx]);
        for (int cid = 0; cid < 4; ++cid) {
          if (q > 1) {
            // Get the boxes per class. class_boxes_data dims is [num_boxes, 4]
            class_boxes_data.push_back(
                boxes_data[(box * q + class_idx) * 4 + cid]);
          } else {
            class_boxes_data.push_back(boxes_data[box * 4 + cid]);
          }
        }
      }

      // Copy class_boxes_data to a tensor
      TensorShape boxesShape({num_boxes, 4});
      Tensor boxes(per_batch_boxes.dtype(), boxesShape);
      std::copy_n(class_boxes_data.begin(), class_boxes_data.size(),
                  boxes.unaligned_flat<float>().data());

      const int size_per_class = std::min(max_size_per_class, num_boxes);
      // Do NMS, get the candidate indices of form vector<int>
      // Data structure for selection candidate in NMS.
      struct Candidate {
        int box_index;
        float score;
      };
      auto cmp = [](const Candidate bs_i, const Candidate bs_j) {
        return bs_i.score > bs_j.score;
      };
      std::vector<Candidate> candidate_vector;
      for (int i = 0; i < class_scores_data.size(); ++i) {
        if (class_scores_data[i] > score_threshold) {
          candidate_vector.emplace_back(Candidate({i, class_scores_data[i]}));
        }
      }

      std::vector<int> selected;
      std::vector<float> selected_boxes;
      Candidate next_candidate;

      std::sort(candidate_vector.begin(), candidate_vector.end(), cmp);
      const Tensor const_boxes = boxes;
      typename TTypes<float, 2>::ConstTensor boxes_data =
          const_boxes.tensor<float, 2>();
      int candidate_idx = 0;
      float iou;
      while (selected.size() < size_per_class &&
             candidate_idx < candidate_vector.size()) {
        next_candidate = candidate_vector[candidate_idx++];

        // Overlapping boxes are likely to have similar scores,
        // therefore we iterate through the previously selected boxes backwards
        // in order to see if `next_candidate` should be suppressed.
        bool should_select = true;
        for (int j = selected.size() - 1; j >= 0; --j) {
          iou = IOU<float>(boxes_data, next_candidate.box_index, selected[j]);
          if (iou > iou_threshold) {
            should_select = false;
            break;
          }
        }

        if (should_select) {
          selected.push_back(next_candidate.box_index);
          // Add the selected box to the result candidate. Sorted by score
          int id = next_candidate.box_index;
          ResultCandidate rc = {next_candidate.box_index,
                                next_candidate.score,
                                class_idx,
                                {boxes_data(id, 0), boxes_data(id, 1),
                                 boxes_data(id, 2), boxes_data(id, 3)}};
          result_candidate_vec.push_back(rc);
        }
      }
    }

    auto rc_cmp = [](const ResultCandidate rc_i, const ResultCandidate rc_j) {
      return rc_i.score > rc_j.score;
    };
    std::sort(result_candidate_vec.begin(), result_candidate_vec.end(), rc_cmp);

    int max_detections = 0;
    // If pad_per_class is false, we always pad to max_total_size
    if (!pad_per_class) {
      max_detections =
          std::min((int)result_candidate_vec.size(), total_size_per_batch);
      per_batch_size = total_size_per_batch;
    } else {
      per_batch_size =
          std::min(total_size_per_batch, max_size_per_class * num_classes);
      max_detections =
          std::min(per_batch_size, (int)result_candidate_vec.size());
    }

    final_valid_detections.push_back(max_detections);

    int curr_total_size = max_detections;
    int result_idx = 0;
    // Pick the top max_detections values
    while (curr_total_size > 0 && result_idx < result_candidate_vec.size()) {
      ResultCandidate next_candidate = result_candidate_vec[result_idx++];
      // Add to final output vectors
      if (clip_boxes) {
        const float box_min = 0.0;
        const float box_max = 1.0;
        nmsed_boxes[batch].push_back(
            std::max(std::min(next_candidate.box_coord[0], box_max), box_min));
        nmsed_boxes[batch].push_back(
            std::max(std::min(next_candidate.box_coord[1], box_max), box_min));
        nmsed_boxes[batch].push_back(
            std::max(std::min(next_candidate.box_coord[2], box_max), box_min));
        nmsed_boxes[batch].push_back(
            std::max(std::min(next_candidate.box_coord[3], box_max), box_min));
      } else {
        nmsed_boxes[batch].push_back(next_candidate.box_coord[0]);
        nmsed_boxes[batch].push_back(next_candidate.box_coord[1]);
        nmsed_boxes[batch].push_back(next_candidate.box_coord[2]);
        nmsed_boxes[batch].push_back(next_candidate.box_coord[3]);
      }
      nmsed_scores[batch].push_back(next_candidate.score);
      nmsed_classes[batch].push_back(next_candidate.class_idx);
      curr_total_size--;
    }

    nmsed_boxes[batch].resize(per_batch_size * 4, 0);
    nmsed_scores[batch].resize(per_batch_size, 0);
    nmsed_classes[batch].resize(per_batch_size, 0);
  }

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

  Tensor* valid_detections_t = nullptr;
  TensorShape valid_detections_shape({num_batches});
  OP_REQUIRES_OK(context, context->allocate_output(3, valid_detections_shape,
                                                   &valid_detections_t));
  auto valid_detections_flat = valid_detections_t->template flat<int>();

  for (int i = 0; i < num_batches; ++i) {
    valid_detections_flat(i) = final_valid_detections[i];
    for (int j = 0; j < per_batch_size; ++j) {
      nmsed_scores_flat(i * per_batch_size + j) = nmsed_scores[i][j];
      nmsed_classes_flat(i * per_batch_size + j) = nmsed_classes[i][j];
      for (int k = 0; k < 4; ++k) {
        nmsed_boxes_flat(i * per_batch_size * 4 + j * 4 + k) =
            nmsed_boxes[i][j * 4 + k];
      }
    }
  }
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
    const T iou_threshold_val = iou_threshold.scalar<T>()();

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
    // max_total_size: scalar
    const Tensor& max_total_size = context->input(3);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_total_size.shape()),
        errors::InvalidArgument("max_total_size must be 0-D, got shape ",
                                max_total_size.shape().DebugString()));
    const int max_total_size_per_batch = max_total_size.scalar<int>()();
    OP_REQUIRES(context, max_total_size_per_batch > 0,
                errors::InvalidArgument("max_total_size must be > 0"));
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
