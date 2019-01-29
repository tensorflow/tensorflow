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
#include "tensorflow/core/lib/gtl/stl_util.h"
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

// Return intersection-over-union overlap between boxes i and j
template <typename T>
static inline bool IOUGreaterThanThreshold(
    typename TTypes<T, 2>::ConstTensor boxes, int i, int j, T iou_threshold) {
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
  if (area_i <= static_cast<T>(0) || area_j <= static_cast<T>(0)) return 0;
  const T intersection_ymin = std::max<T>(ymin_i, ymin_j);
  const T intersection_xmin = std::max<T>(xmin_i, xmin_j);
  const T intersection_ymax = std::min<T>(ymax_i, ymax_j);
  const T intersection_xmax = std::min<T>(xmax_i, xmax_j);
  const T intersection_area =
      std::max<T>(intersection_ymax - intersection_ymin, static_cast<T>(0.0)) *
      std::max<T>(intersection_xmax - intersection_xmin, static_cast<T>(0.0));
  const T iou = intersection_area / (area_i + area_j - intersection_area);
  return iou > iou_threshold;
}

static inline bool OverlapsGreaterThanThreshold(
    typename TTypes<float, 2>::ConstTensor overlaps, int i, int j,
    float overlap_threshold) {
  return overlaps(i, j) > overlap_threshold;
}

template <typename T>
static inline std::function<bool(int, int)> CreateIOUSuppressCheckFn(
    const Tensor& boxes, float threshold) {
  typename TTypes<T, 2>::ConstTensor boxes_data = boxes.tensor<T, 2>();
  return std::bind(&IOUGreaterThanThreshold<T>, boxes_data,
                   std::placeholders::_1, std::placeholders::_2,
                   static_cast<T>(threshold));
}

static inline std::function<bool(int, int)> CreateOverlapsSuppressCheckFn(
    const Tensor& overlaps, float threshold) {
  typename TTypes<float, 2>::ConstTensor overlaps_data =
      overlaps.tensor<float, 2>();
  return std::bind(&OverlapsGreaterThanThreshold, overlaps_data,
                   std::placeholders::_1, std::placeholders::_2, threshold);
}

template <typename T>
void DoNonMaxSuppressionOp(
    OpKernelContext* context, const Tensor& scores, int num_boxes,
    const Tensor& max_output_size, const float score_threshold,
    const std::function<bool(int, int)>& suppress_check_fn,
    bool pad_to_max_output_size = false, int* ptr_num_valid_outputs = nullptr) {
  const int output_size = max_output_size.scalar<int>()();

  std::vector<T> scores_data(num_boxes);
  std::copy_n(scores.flat<T>().data(), num_boxes, scores_data.begin());

  // Data structure for selection candidate in NMS.
  struct Candidate {
    int box_index;
    T score;
  };

  auto cmp = [](const Candidate bs_i, const Candidate bs_j) {
    return bs_i.score < bs_j.score;
  };
  std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)>
      candidate_priority_queue(cmp);
  for (int i = 0; i < scores_data.size(); ++i) {
    if (static_cast<float>(scores_data[i]) > score_threshold) {
      candidate_priority_queue.emplace(Candidate({i, scores_data[i]}));
    }
  }

  std::vector<int> selected;
  std::vector<T> selected_scores;
  Candidate next_candidate;

  while (selected.size() < output_size && !candidate_priority_queue.empty()) {
    next_candidate = candidate_priority_queue.top();
    candidate_priority_queue.pop();

    // Overlapping boxes are likely to have similar scores,
    // therefore we iterate through the previously selected boxes backwards
    // in order to see if `next_candidate` should be suppressed.
    bool should_select = true;

    for (int j = static_cast<int>(selected.size()) - 1; j >= 0; --j) {
      if (suppress_check_fn(next_candidate.box_index, selected[j])) {
        should_select = false;
        break;
      }
    }

    if (should_select) {
      selected.push_back(next_candidate.box_index);
      selected_scores.push_back(next_candidate.score);
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
    auto suppress_check_fn =
        CreateIOUSuppressCheckFn<float>(boxes, iou_threshold_);

    const float score_threshold_val = std::numeric_limits<float>::lowest();
    DoNonMaxSuppressionOp<float>(context, scores, num_boxes, max_output_size,
                                 score_threshold_val, suppress_check_fn);
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
    const float iou_threshold_val = iou_threshold.scalar<float>()();

    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }
    auto suppress_check_fn =
        CreateIOUSuppressCheckFn<T>(boxes, iou_threshold_val);

    const float score_threshold_val = std::numeric_limits<float>::lowest();
    DoNonMaxSuppressionOp<T>(context, scores, num_boxes, max_output_size,
                             score_threshold_val, suppress_check_fn);
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
    const float iou_threshold_val = iou_threshold.scalar<float>()();
    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }

    auto suppress_check_fn =
        CreateIOUSuppressCheckFn<T>(boxes, iou_threshold_val);

    DoNonMaxSuppressionOp<T>(context, scores, num_boxes, max_output_size,
                             score_threshold_val, suppress_check_fn);
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
    const float iou_threshold_val = iou_threshold.scalar<float>()();
    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }

    auto suppress_check_fn =
        CreateIOUSuppressCheckFn<T>(boxes, iou_threshold_val);
    int num_valid_outputs;

    DoNonMaxSuppressionOp<T>(context, scores, num_boxes, max_output_size,
                             score_threshold_val, suppress_check_fn,
                             pad_to_max_output_size_, &num_valid_outputs);

    // Allocate scalar output tensor for number of indices computed.
    Tensor* num_outputs_t = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, tensorflow::TensorShape{}, &num_outputs_t));
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
    auto suppress_check_fn =
        CreateOverlapsSuppressCheckFn(overlaps, overlap_threshold_val);

    DoNonMaxSuppressionOp<float>(context, scores, num_boxes, max_output_size,
                                 score_threshold_val, suppress_check_fn);
  }
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
    Name("NonMaxSuppressionWithOverlaps").Device(DEVICE_CPU),
    NonMaxSuppressionWithOverlapsOp<CPUDevice>);

}  // namespace tensorflow
