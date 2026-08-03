/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/ir_emission_utils.h"

#include <cstdint>

#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/cpu/cpu_runtime.h"
#include "xla/shape_util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

bool IsElementalKernelOpcode(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kAbs:
    case HloOpcode::kAcos:
    case HloOpcode::kAcosh:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAsin:
    case HloOpcode::kAsinh:
    case HloOpcode::kAtan2:
    case HloOpcode::kAtanh:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCbrt:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kCosh:
    case HloOpcode::kDivide:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kGather:
    case HloOpcode::kImag:
    case HloOpcode::kIota:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kMap:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMulhi:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kPad:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kPower:
    case HloOpcode::kReal:
    case HloOpcode::kReduce:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kRemainder:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSinh:
    case HloOpcode::kSlice:
    case HloOpcode::kSqrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kTranspose:
    case HloOpcode::kXor:
      return true;
    default:
      return false;
  }
}

int64_t GetMinimumAlignmentForArray(
    const Shape& shape, const TargetMachineFeatures& target_machine_features) {
  CHECK(shape.IsArray());

  // We don't require a layout to be set on `shape`.  This only works on CPU
  // because we don't pad our tensors or otherwise have complicated data tiling
  // schemes.

  int64_t allocation_size_bytes =
      ShapeUtil::ElementsIn(shape) *
      ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  return target_machine_features.minimum_alignment_for_allocation(
      allocation_size_bytes);
}

bool PotentiallyImplementedAsEigenConvolution(
    const HloInstruction& convolution,
    const TargetMachineFeatures& target_machine_features) {
  if (convolution.opcode() != HloOpcode::kConvolution) {
    return false;
  }
  if (window_util::HasWindowReversal(convolution.window())) {
    return false;
  }

  const ConvolutionDimensionNumbers& dnums =
      convolution.convolution_dimension_numbers();
  const int64_t num_spatial_dims = dnums.output_spatial_dimensions_size();
  if (num_spatial_dims < 1 || num_spatial_dims > 3) {
    return false;
  }

  const Shape& input_shape = convolution.operand(0)->shape();
  const Shape& kernel_shape = convolution.operand(1)->shape();
  const Shape& output_shape = convolution.shape();

  for (int64_t i = 0; i < num_spatial_dims; ++i) {
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

  if (dnums.input_batch_dimension() != 0 ||
      dnums.input_feature_dimension() != input_shape.dimensions().size() - 1 ||
      dnums.output_batch_dimension() != 0 ||
      dnums.output_feature_dimension() !=
          output_shape.dimensions().size() - 1 ||
      dnums.kernel_input_feature_dimension() !=
          kernel_shape.dimensions().size() - 2 ||
      dnums.kernel_output_feature_dimension() !=
          kernel_shape.dimensions().size() - 1) {
    return false;
  }

  // The following conditions are necessary (but not sufficient) for
  // implementing `convolution` with Eigen convolution:
  // - the input and kernel have a non-zero number of elements.
  // - the input is in NHWC order.
  // - the kernel is in HWIO order.
  //
  // To be sufficient, certain layout constraints need to be satisfied as well.
  // Alignment is guaranteed by XLA CPU buffer allocations.

  auto is_aligned = [&](const HloInstruction* operand) {
    return GetMinimumAlignmentForArray(operand->shape(),
                                       target_machine_features) >=
           TargetMachineFeatures::kEigenExpectedTensorAlignment;
  };
  if (!is_aligned(&convolution) || !is_aligned(convolution.operand(0)) ||
      !is_aligned(convolution.operand(1))) {
    return false;
  }

  // Make sure input and kernel has the same data type.
  CHECK(
      ShapeUtil::SameElementTypeIgnoringFpPrecision(input_shape, kernel_shape));
  // TODO(b/65408531): Explore using Eigen dot for complex64 type.
  PrimitiveType primitive_type = input_shape.element_type();
  return primitive_type == F16 || primitive_type == F32;
}

bool CanUseEigenConvolution(
    const HloInstruction& convolution,
    const TargetMachineFeatures& target_machine_features) {
  if (!PotentiallyImplementedAsEigenConvolution(convolution,
                                                target_machine_features)) {
    return false;
  }

  const Shape& input_shape = convolution.operand(0)->shape();
  const Shape& kernel_shape = convolution.operand(1)->shape();
  const Shape& output_shape = convolution.shape();

  return LayoutUtil::IsMonotonicWithDim0Major(input_shape.layout()) &&
         LayoutUtil::IsMonotonicWithDim0Major(kernel_shape.layout()) &&
         LayoutUtil::IsMonotonicWithDim0Major(output_shape.layout());
}

}  // namespace cpu
}  // namespace xla
