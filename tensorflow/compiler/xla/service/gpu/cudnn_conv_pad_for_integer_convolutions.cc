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

#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_pad_for_integer_convolutions.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_pad_features.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace gpu {

// We always pad regardless the memory usage increase.
// We still check the increase for debugging purpose.
static constexpr double kMaxBytesTouchedIncrease = 1.35;

static StatusOr<bool> ResolvePadedShapes(
    HloCustomCallInstruction* conv, std::vector<Shape>* new_input_shapes_ptr,
    Shape* new_result_shape_ptr) {
  TF_ASSIGN_OR_RETURN(auto kind, GetCudnnConvKind(conv));
  const Shape& result_shape = conv->shape().tuple_shapes(0);

  // Integer convolution only
  if (!primitive_util::IsIntegralType(result_shape.element_type())) {
    return false;
  }

  // kForward and kForwardActivation only
  if (kind != CudnnConvKind::kForward &&
      kind != CudnnConvKind::kForwardActivation) {
    return false;
  }

  const auto& dnums = conv->convolution_dimension_numbers();
  std::vector<Shape>& new_input_shapes = *new_input_shapes_ptr;
  for (auto operand : conv->operands()) {
    new_input_shapes.push_back(operand->shape());
  }
  Shape& new_result_shape = *new_result_shape_ptr;
  new_result_shape = conv->shape().tuple_shapes(0);

  // Pad the features to multiples of 4 and check that
  // the conv buffers size changes for debugging purpose.
  {
    auto pad_dim = [](Shape* s, int64 dim) {
      s->set_dimensions(dim, RoundUpToNearest<int64>(s->dimensions(dim), 4));
    };

    switch (kind) {
      case CudnnConvKind::kForward:
        CHECK(new_input_shapes.size() == 2);
        pad_dim(&new_input_shapes[0], dnums.input_feature_dimension()); // Input feature maps
        pad_dim(&new_input_shapes[1], dnums.kernel_input_feature_dimension()); // Kernel for the input feature maps
        pad_dim(&new_input_shapes[1], dnums.kernel_output_feature_dimension()); // Kernel for the output feature maps
        pad_dim(&new_result_shape, dnums.output_feature_dimension()); // Output feature maps
        break;
      case CudnnConvKind::kForwardActivation:
        CHECK(new_input_shapes.size() == 3 || new_input_shapes.size() == 4);
        pad_dim(&new_input_shapes[0], dnums.input_feature_dimension()); // Input feature maps
        pad_dim(&new_input_shapes[1], dnums.kernel_input_feature_dimension()); // Kernel for the input feature maps
        pad_dim(&new_input_shapes[1], dnums.kernel_output_feature_dimension()); // Kernel for the output feature maps
        pad_dim(&new_input_shapes[2], 0); // Bias
        if (new_input_shapes.size() == 4) {
          pad_dim(&new_input_shapes[3], dnums.output_feature_dimension()); // Optional side input
        }
        pad_dim(&new_result_shape, dnums.output_feature_dimension()); // Output feature maps
        break;
      default:
        CHECK(false);
    }
    // Check that padding wouldn't increase the total bytes read/written by this
    // operation too much.
    auto check_size_increase = [&](const Shape& old_shape,
                                   const Shape& new_shape) {
      int64 old_bytes = ShapeUtil::ByteSizeOf(old_shape);
      int64 new_bytes = ShapeUtil::ByteSizeOf(new_shape);
      if (new_bytes <= old_bytes * kMaxBytesTouchedIncrease) {
        return;
      }
      VLOG(3)
          << "Not padding convolution; doing so would change input / result "
             "shape from "
          << ShapeUtil::HumanString(old_shape) << " to "
          << ShapeUtil::HumanString(new_shape) << ", a size increase of "
          << new_bytes / static_cast<double>(old_bytes) << "x > "
          << kMaxBytesTouchedIncrease << "x: " << conv->ToString();
      return;
    };

    for (int64 i = 0; i < conv->operand_count(); ++i) {
      check_size_increase(conv->operand(i)->shape(), new_input_shapes[i]);
    }
    check_size_increase(result_shape, new_result_shape);
  }

  bool changed = false;
  for (int64 i = 0; i < conv->operand_count(); ++i) {
    changed |=
        !ShapeUtil::Equal(conv->operand(i)->shape(), new_input_shapes[i]);
  }
  if (!changed) {
    VLOG(3) << "No need to pad features of " << conv->ToString();
    return false;
  }

  return true;
}

StatusOr<bool> CudnnConvPadForIntegerConvolutions::Run(HloModule* module) {
  CudnnConvPadFeatures impl;
  return impl.Run(module, ResolvePadedShapes);
}
}  // namespace gpu
}  // namespace xla
