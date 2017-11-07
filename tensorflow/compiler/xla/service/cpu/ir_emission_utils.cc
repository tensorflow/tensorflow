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
  const Shape& input_shape = convolution.operand(0)->shape();
  const Shape& kernel_shape = convolution.operand(0)->shape();
  if (ShapeUtil::HasZeroElements(input_shape) ||
      ShapeUtil::HasZeroElements(kernel_shape)) {
    return false;
  }
  // TODO(b/65408531): Explore using Eigen dot for complex64 type.
  if (ShapeUtil::ElementIsComplex(input_shape) ||
      ShapeUtil::ElementIsComplex(kernel_shape)) {
    return false;
  }

  const ConvolutionDimensionNumbers& dnums =
      convolution.convolution_dimension_numbers();
  // Only 1D and 2D convolutions are supported at the moment.
  // TODO(b/32897908): add an optimized implementation for 3D convolution.
  if (dnums.spatial_dimensions_size() > 2) {
    return false;
  }

  bool input_spatial_dims_ascending = std::is_sorted(
      dnums.spatial_dimensions().begin(), dnums.spatial_dimensions().end());
  bool kernel_spatial_dims_ascending =
      std::is_sorted(dnums.kernel_spatial_dimensions().begin(),
                     dnums.kernel_spatial_dimensions().end());

  const Shape& output_shape = convolution.shape();
  return dnums.input_batch_dimension() == 0 &&
         dnums.input_feature_dimension() == input_shape.dimensions_size() - 1 &&
         dnums.output_batch_dimension() == 0 &&
         dnums.output_feature_dimension() ==
             output_shape.dimensions_size() - 1 &&
         input_spatial_dims_ascending == kernel_spatial_dims_ascending &&
         dnums.kernel_input_feature_dimension() ==
             kernel_shape.dimensions_size() - 2 &&
         dnums.kernel_output_feature_dimension() ==
             kernel_shape.dimensions_size() - 1;
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

    if (ProfitableToImplementDotInLlvmIr(hlo) == DotInLlvmIrProfitable::kYes) {
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
    auto* dot = hlo.fused_expression_root();
    const Shape& lhs_shape = dot->operand(0)->shape();
    const Shape& rhs_shape = dot->operand(1)->shape();
    if (ShapeUtil::HasZeroElements(lhs_shape) ||
        ShapeUtil::HasZeroElements(rhs_shape)) {
      return false;
    }
    return true;
  }

  return false;
}

DotInLlvmIrProfitable ProfitableToImplementDotInLlvmIr(
    const HloInstruction& dot) {
  if (dot.opcode() == HloOpcode::kDot && dot.shape().dimensions_size() == 2) {
    const Shape& result_shape = dot.shape();
    // kReductionDimensionThresholdBytes was chosen to be 1/4 of a typical L1
    // cache line size, so that we can have the reduction dimension of both the
    // LHS and RHS matrices and still have some space "left over".  This needs
    // to be tuned further.
    const int64 kReductionDimensionThresholdBytes = 8 * 1024;
    const bool single_threaded_eigen =
        !dot.GetModule()->config().debug_options().xla_cpu_multi_thread_eigen();

    // This is the point at which it is better to call into Eigen and shard the
    // dot across multiple worker threads.  This is a rough estimate by running
    // a matmult benchmark on my local machine, and it can be tuned further.
    const int64 kMaxSingleThreadedFlops = 16 * 1024;

    const int64 M = result_shape.dimensions(0);
    const int64 N = result_shape.dimensions(1);
    const int64 K = dot.operand(1)->shape().dimensions(0);
    const int64 primitive_type_size =
        ShapeUtil::ByteSizeOfPrimitiveType(result_shape.element_type());
    if (M == 1 &&
        K * primitive_type_size <= kReductionDimensionThresholdBytes &&
        (single_threaded_eigen || M * K * N <= kMaxSingleThreadedFlops)) {
      // Heuristics:
      //
      //  - Look for a configuration where we will likely be able to keep LHS in
      //    L1 and do a cache-optimal traversal of RHS.
      //
      //  - Bail out on matrices that are large enough that Eigen can profitably
      //    shard the computation across multiple cores.  This only applies when
      //    multi-threading is enabled.
      return LayoutUtil::IsMonotonicWithDim0Major(
                 dot.operand(1)->shape().layout())
                 ? DotInLlvmIrProfitable::kWithColumnMajorRhs
                 : DotInLlvmIrProfitable::kYes;
    }
  }
  return DotInLlvmIrProfitable::kNo;
}

}  // namespace cpu
}  // namespace xla
