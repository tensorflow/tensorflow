/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_pad_for_tensor_cores.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_pad_features.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace gpu {

// We won't pad a conv if doing so increases the total number of bytes in the
// lhs, rhs, or result by more than this amount.
//
// TODO(jlebar): This number was tuned experimentally.  It represents a
// compromise on our current benchmarks; it speeds some up significantly, and
// doesn't slow any down.  But we can observe by changing this value that
// there's additional room for speedups.  Achieving those speedups without
// also slowing other things down will likely require a more sophisticated
// heuristic, possibly some form of auto-tuning.
static constexpr double kMaxBytesTouchedIncrease = 1.35;

static StatusOr<bool> ResolvePadedShapes(
    HloCustomCallInstruction* conv, std::vector<Shape>* new_input_shapes_ptr,
    Shape* new_result_shape_ptr) {
  TF_ASSIGN_OR_RETURN(auto kind, GetCudnnConvKind(conv));
  const auto& dnums = conv->convolution_dimension_numbers();
  auto* lhs = conv->mutable_operand(0);
  auto* rhs = conv->mutable_operand(1);
  const Shape& result_shape = conv->shape().tuple_shapes(0);

  // Nothing to do on non-f16 convolutions.
  if (result_shape.element_type() != PrimitiveType::F16) {
    return false;
  }

  // TODO(timshen): Don't skip forward-activation convs if we find a benchmark
  // where there's a speedup.
  if (kind == CudnnConvKind::kForwardActivation) {
    return false;
  }

  Shape new_lhs_shape = lhs->shape();
  Shape new_rhs_shape = rhs->shape();
  Shape& new_result_shape = *new_result_shape_ptr;
  new_result_shape = conv->shape().tuple_shapes(0);

  // new_{input,filter_output}_shape points to the appropriate one of
  // new_{lhs,rhs,result}_shape.
  Shape* new_input_shape;
  Shape* new_filter_shape;
  Shape* new_output_shape;
  std::tie(new_input_shape, new_filter_shape, new_output_shape) = [&] {
    switch (kind) {
      case CudnnConvKind::kForward:
      case CudnnConvKind::kForwardActivation:
        return std::make_tuple(&new_lhs_shape, &new_rhs_shape,
                               &new_result_shape);
      case CudnnConvKind::kBackwardInput:
        return std::make_tuple(&new_result_shape, &new_rhs_shape,
                               &new_lhs_shape);
      case CudnnConvKind::kBackwardFilter:
        return std::make_tuple(&new_lhs_shape, &new_result_shape,
                               &new_rhs_shape);
    }
  }();

  // If there are 3 input features and 32 or 64 output features, pad the input
  // features to 4.  Otherwise, try padding to multiples of 8 and check that
  // this doesn't make any of the conv buffers too much larger.
  auto input_features =
      new_input_shape->dimensions(dnums.input_feature_dimension());
  auto output_features =
      new_output_shape->dimensions(dnums.output_feature_dimension());
  if (input_features == 3 && (output_features == 32 || output_features == 64)) {
    new_input_shape->set_dimensions(dnums.input_feature_dimension(), 4);
    new_filter_shape->set_dimensions(dnums.kernel_input_feature_dimension(), 4);
  } else {
    auto pad_dim = [](Shape* s, int64 dim) {
      s->set_dimensions(dim, RoundUpToNearest<int64>(s->dimensions(dim), 8));
    };
    pad_dim(new_input_shape, dnums.input_feature_dimension());
    pad_dim(new_filter_shape, dnums.kernel_input_feature_dimension());
    pad_dim(new_filter_shape, dnums.kernel_output_feature_dimension());
    pad_dim(new_output_shape, dnums.output_feature_dimension());

    // Check that padding wouldn't increase the total bytes read/written by this
    // operation too much.
    auto check_size_increase = [&](const Shape& old_shape,
                                   const Shape& new_shape) {
      int64 old_bytes = ShapeUtil::ByteSizeOf(old_shape);
      int64 new_bytes = ShapeUtil::ByteSizeOf(new_shape);
      if (new_bytes <= old_bytes * kMaxBytesTouchedIncrease) {
        return true;
      }
      VLOG(3)
          << "Not padding convolution; doing so would change input / result "
             "shape from "
          << ShapeUtil::HumanString(old_shape) << " to "
          << ShapeUtil::HumanString(new_shape) << ", a size increase of "
          << new_bytes / static_cast<double>(old_bytes) << "x > "
          << kMaxBytesTouchedIncrease << "x: " << conv->ToString();
      return false;
    };

    if (!check_size_increase(lhs->shape(), new_lhs_shape) ||
        !check_size_increase(rhs->shape(), new_rhs_shape) ||
        !check_size_increase(result_shape, new_result_shape)) {
      return false;
    }
  }

  if (ShapeUtil::Equal(lhs->shape(), new_lhs_shape) &&
      ShapeUtil::Equal(rhs->shape(), new_rhs_shape)) {
    VLOG(3) << "No need to pad features of " << conv->ToString();
    return false;
  }

  new_input_shapes_ptr->push_back(new_lhs_shape);
  new_input_shapes_ptr->push_back(new_rhs_shape);
  return true;
}

StatusOr<bool> CudnnConvPadForTensorCores::Run(HloModule* module) {
  CudnnConvPadFeatures impl;
  return impl.Run(module, ResolvePadedShapes);
}

}  // namespace gpu
}  // namespace xla
