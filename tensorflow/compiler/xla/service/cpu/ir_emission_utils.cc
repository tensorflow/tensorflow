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
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace cpu {

bool PotentiallyImplementedAsEigenConvolution(
    const HloInstruction& convolution) {
  // The following conditions are necessary (but not sufficient) for
  // implementing `convolution` with Eigen convolution:
  // - the input and kernel have a non-zero number of elements.
  // - the input is in NHWC or NWHC order.
  // - the kernel is in HWIO or WHIO order.
  // - the spatial dimensions are in the same relative order in the input,
  //   kernel and output.
  //
  // To be sufficient, certain layout constraints need to be satisfied as well.
  if (ShapeUtil::HasZeroElements(convolution.operand(0)->shape()) ||
      ShapeUtil::HasZeroElements(convolution.operand(1)->shape())) {
    return false;
  }
  const ConvolutionDimensionNumbers& dnums =
      convolution.convolution_dimension_numbers();
  // Only 2D convolutions are supported at the moment.
  // TODO(b/32897908): add an optimized implementation for 3D convolution.
  if (dnums.spatial_dimensions_size() != 2) {
    return false;
  }
  bool input_spatial_dims_ascending =
      dnums.spatial_dimensions(0) < dnums.spatial_dimensions(1);
  bool kernel_spatial_dims_ascending =
      dnums.kernel_spatial_dimensions(0) < dnums.kernel_spatial_dimensions(1);
  return dnums.batch_dimension() == 0 && dnums.feature_dimension() == 3 &&
         input_spatial_dims_ascending == kernel_spatial_dims_ascending &&
         dnums.kernel_input_feature_dimension() == 2 &&
         dnums.kernel_output_feature_dimension() == 3;
}

namespace {

// Return whether the given shape is a matrix with no padding.
bool IsRank2WithNoPadding(const Shape& shape) {
  return ShapeUtil::Rank(shape) == 2 && !LayoutUtil::IsPadded(shape);
}

// In a gemm operation where output = lhs * rhs, check whether the given shapes
// are valid for the operation.
bool AreValidGemmShapes(const Shape& lhs_shape, const Shape& rhs_shape,
                        const Shape& output_shape) {
  // The inputs and the output must
  // 1) be matrices with no padding, and
  // 2) have an allowed element type.
  return output_shape.element_type() == F32 &&
         IsRank2WithNoPadding(lhs_shape) && IsRank2WithNoPadding(rhs_shape) &&
         IsRank2WithNoPadding(output_shape);
}
}  // namespace

bool PotentiallyImplementedAsEigenDot(const HloInstruction& hlo) {
  // For certain types of Dot, we can call Eigen
  if (hlo.opcode() == HloOpcode::kDot) {
    const Shape& lhs_shape = hlo.operand(0)->shape();
    const Shape& rhs_shape = hlo.operand(1)->shape();

    if (ShapeUtil::HasZeroElements(lhs_shape) ||
        ShapeUtil::HasZeroElements(rhs_shape)) {
      return false;
    }

    // If gemm can accept the operand shapes, use it rather than a custom
    // kernel.
    if (AreValidGemmShapes(lhs_shape, rhs_shape, hlo.shape())) {
      // The size of the reduction dimension should match. The shape inference
      // guarantees this invariant, so the check here is for programming
      // errors.
      CHECK_EQ(lhs_shape.dimensions(1), rhs_shape.dimensions(0));
      return true;
    }
  }

  if (hlo.opcode() == HloOpcode::kFusion &&
      hlo.fusion_kind() == HloInstruction::FusionKind::kTransposeDot &&
      hlo.fused_expression_root()->opcode() == HloOpcode::kDot) {
    const Shape& lhs_shape = hlo.operand(0)->shape();
    const Shape& rhs_shape = hlo.operand(1)->shape();
    if (ShapeUtil::HasZeroElements(lhs_shape) ||
        ShapeUtil::HasZeroElements(rhs_shape)) {
      return false;
    }
    return true;
  }

  return false;
}

}  // namespace cpu
}  // namespace xla
