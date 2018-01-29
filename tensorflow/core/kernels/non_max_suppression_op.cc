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

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

typedef Eigen::ThreadPoolDevice CPUDevice;

static inline void ParseAndCheckBoxSizes(OpKernelContext* context,
                                         const Tensor& boxes,
                                         const Tensor& scores, int* num_boxes) {
  // The shape of 'boxes' is [num_boxes, 4]
  OP_REQUIRES(context, boxes.dims() == 2,
              errors::InvalidArgument("boxes must be 2-D",
                                      boxes.shape().DebugString()));
  *num_boxes = boxes.dim_size(0);
  OP_REQUIRES(context, boxes.dim_size(1) == 4,
              errors::InvalidArgument("boxes must have 4 columns"));

  // The shape of 'scores' is [num_boxes]
  OP_REQUIRES(context, scores.dims() == 1,
              errors::InvalidArgument("scores must be 1-D",
                                      scores.shape().DebugString()));
  OP_REQUIRES(context, scores.dim_size(0) == *num_boxes,
              errors::InvalidArgument("scores has incompatible shape"));
}

static inline void DecreasingArgSort(const std::vector<float>& values,
                                     std::vector<int>* indices) {
  indices->resize(values.size());
  for (int i = 0; i < values.size(); ++i) (*indices)[i] = i;
  std::sort(
      indices->begin(), indices->end(),
      [&values](const int i, const int j) { return values[i] > values[j]; });
}

// Return true if intersection-over-union overlap between boxes i and j
// is greater than iou_threshold.
static inline bool IOUGreaterThanThreshold(
    typename TTypes<float, 2>::ConstTensor boxes, int i, int j,
    float iou_threshold) {
  const float ymin_i = std::min<float>(boxes(i, 0), boxes(i, 2));
  const float xmin_i = std::min<float>(boxes(i, 1), boxes(i, 3));
  const float ymax_i = std::max<float>(boxes(i, 0), boxes(i, 2));
  const float xmax_i = std::max<float>(boxes(i, 1), boxes(i, 3));
  const float ymin_j = std::min<float>(boxes(j, 0), boxes(j, 2));
  const float xmin_j = std::min<float>(boxes(j, 1), boxes(j, 3));
  const float ymax_j = std::max<float>(boxes(j, 0), boxes(j, 2));
  const float xmax_j = std::max<float>(boxes(j, 1), boxes(j, 3));
  const float area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
  const float area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);
  if (area_i <= 0 || area_j <= 0) return 0.0;
  const float intersection_ymin = std::max<float>(ymin_i, ymin_j);
  const float intersection_xmin = std::max<float>(xmin_i, xmin_j);
  const float intersection_ymax = std::min<float>(ymax_i, ymax_j);
  const float intersection_xmax = std::min<float>(xmax_i, xmax_j);
  const float intersection_area =
      std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
      std::max<float>(intersection_xmax - intersection_xmin, 0.0);
  const float iou = intersection_area / (area_i + area_j - intersection_area);
  return iou > iou_threshold;
}

void DoNonMaxSuppressionOp(OpKernelContext* context, const Tensor& boxes,
                           const Tensor& scores, const Tensor& max_output_size,
                           const float iou_threshold) {
  OP_REQUIRES(context, iou_threshold >= 0 && iou_threshold <= 1,
              errors::InvalidArgument("iou_threshold must be in [0, 1]"));

  int num_boxes = 0;
  ParseAndCheckBoxSizes(context, boxes, scores, &num_boxes);
  if (!context->status().ok()) {
    return;
  }

  const int output_size = std::min(max_output_size.scalar<int>()(), num_boxes);
  typename TTypes<float, 2>::ConstTensor boxes_data = boxes.tensor<float, 2>();

  std::vector<float> scores_data(num_boxes);
  std::copy_n(scores.flat<float>().data(), num_boxes, scores_data.begin());
  std::vector<int> sorted_indices;
  DecreasingArgSort(scores_data, &sorted_indices);

  std::vector<int> selected;
  std::vector<int> selected_indices(output_size, 0);
  int num_selected = 0;
  for (int i = 0; i < num_boxes; ++i) {
    if (selected.size() >= output_size) break;
    bool should_select = true;
    // Overlapping boxes are likely to have similar scores,
    // therefore we iterate through the selected boxes backwards.
    for (int j = num_selected - 1; j >= 0; --j) {
      if (IOUGreaterThanThreshold(boxes_data, sorted_indices[i],
                                  sorted_indices[selected_indices[j]],
                                  iou_threshold)) {
        should_select = false;
        break;
      }
    }
    if (should_select) {
      selected.push_back(sorted_indices[i]);
      selected_indices[num_selected++] = i;
    }
  }

  // Allocate output tensor
  Tensor* output = nullptr;
  TensorShape output_shape({static_cast<int>(selected.size())});
  OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
  typename TTypes<int, 1>::Tensor selected_indices_data =
      output->tensor<int, 1>();
  std::copy_n(selected.begin(), selected.size(), selected_indices_data.data());
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

    DoNonMaxSuppressionOp(context, boxes, scores, max_output_size,
                          iou_threshold_);
  }

 private:
  float iou_threshold_;
};

template <typename Device>
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

    DoNonMaxSuppressionOp(context, boxes, scores, max_output_size,
                          iou_threshold_val);
  }
};

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppression").Device(DEVICE_CPU),
                        NonMaxSuppressionOp<CPUDevice>);

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV2").Device(DEVICE_CPU),
                        NonMaxSuppressionV2Op<CPUDevice>);

}  // namespace tensorflow
