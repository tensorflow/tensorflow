/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace cpu {

int64 GetMinimumAlignmentForArray(
    const Shape& shape, const TargetMachineFeatures& target_machine_features) {
  CHECK(shape.IsArray());
  CHECK(!LayoutUtil::HasLayout(shape) || LayoutUtil::IsDense(shape.layout()));

  // We don't require a layout to be set on `shape`.  This only works on CPU
  // because we don't pad our tensors or otherwise have complicated data tiling
  // schemes.

  int64 allocation_size_bytes =
      ShapeUtil::ElementsIn(shape) *
      ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  return target_machine_features.minimum_alignment_for_allocation(
      allocation_size_bytes);
}

bool PotentiallyImplementedAsEigenConvolution(
    const HloInstruction& convolution,
    const TargetMachineFeatures& target_machine_features) {
  // The following conditions are necessary (but not sufficient) for
  // implementing `convolution` with Eigen convolution:
  // - the input and kernel have a non-zero number of elements.
  // - the input is in NHWC order.
  // - the kernel is in HWIO order.
  //
  // To be sufficient, certain layout constraints need to be satisfied as well.
  const Shape& input_shape = convolution.operand(0)->shape();
  const Shape& kernel_shape = convolution.operand(1)->shape();
  const Shape& output_shape = convolution.shape();

  auto is_aligned = [&](const Shape& shape) {
    return GetMinimumAlignmentForArray(shape, target_machine_features) >=
           TargetMachineFeatures::kEigenExpectedTensorAlignment;
  };

  if (!is_aligned(input_shape) || !is_aligned(kernel_shape) ||
      !is_aligned(output_shape)) {
    return false;
  }

  if (ShapeUtil::IsZeroElementArray(input_shape) ||
      ShapeUtil::IsZeroElementArray(kernel_shape)) {
    return false;
  }
  // Make sure input and kernel has the same data type.
  CHECK(
      ShapeUtil::SameElementTypeIgnoringFpPrecision(input_shape, kernel_shape));
  // TODO(b/65408531): Explore using Eigen dot for complex64 type.
  PrimitiveType primitive_type = input_shape.element_type();
  if (primitive_type != F16 && primitive_type != F32) {
    return false;
  }
  if (window_util::HasWindowReversal(convolution.window())) {
    return false;
  }

  const ConvolutionDimensionNumbers& dnums =
      convolution.convolution_dimension_numbers();
  // Only 1D and 2D convolutions are supported at the moment.
  // TODO(b/32897908): add an optimized implementation for 3D convolution.
  const int64 num_spatial_dims = dnums.output_spatial_dimensions_size();
  if (num_spatial_dims > 2) {
    return false;
  }

  for (int64 i = 0; i < num_spatial_dims; ++i) {
    if (dnums.input_spatial_dimensions(i) != i + 1) {
      return false;
    }
    if (dnums.kernel_spatial_dimensions(i) != i) {
      return false;
    }
    if (dnums.output_spatial_dimensions(i) != i + 1) {
      return false;
    }
  }

  return dnums.input_batch_dimension() == 0 &&
         dnums.input_feature_dimension() == input_shape.dimensions_size() - 1 &&
         dnums.output_batch_dimension() == 0 &&
         dnums.output_feature_dimension() ==
             output_shape.dimensions_size() - 1 &&
         dnums.kernel_input_feature_dimension() ==
             kernel_shape.dimensions_size() - 2 &&
         dnums.kernel_output_feature_dimension() ==
             kernel_shape.dimensions_size() - 1;
}

}  // namespace cpu
}  // namespace xla
