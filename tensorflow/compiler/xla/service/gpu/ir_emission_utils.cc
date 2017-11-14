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

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"

#include <algorithm>
#include <vector>

#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {
namespace gpu {

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
  // 1) be matrices with no padding and a non-zero number of elements,
  // 2) have an allowed element type.
  bool type_is_allowed = (output_shape.element_type() == F32 ||
                          output_shape.element_type() == F64);
  return type_is_allowed && IsRank2WithNoPadding(lhs_shape) &&
         IsRank2WithNoPadding(rhs_shape) &&
         IsRank2WithNoPadding(output_shape) &&
         !ShapeUtil::HasZeroElements(lhs_shape) &&
         !ShapeUtil::HasZeroElements(rhs_shape);
}
}  // namespace

bool ImplementedAsGemm(const HloInstruction& hlo) {
  // We can only do this if the HLO is unnested.
  if (hlo.parent() != hlo.GetModule()->entry_computation()) {
    return false;
  }

  // For certain types of Dot, we can call pre-canned BLAS gemm.
  if (hlo.opcode() == HloOpcode::kDot) {
    const Shape& lhs_shape = hlo.operand(0)->shape();
    const Shape& rhs_shape = hlo.operand(1)->shape();

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
    return true;
  }

  return false;
}

bool ImplementedAsDnnConvolution(const HloInstruction& hlo) {
  // We can only do this if the HLO is unnested.
  if (hlo.parent() != hlo.GetModule()->entry_computation()) {
    return false;
  }

  // Forward convolution.
  if (hlo.opcode() == HloOpcode::kConvolution) {
    const ConvolutionDimensionNumbers& dnums =
        hlo.convolution_dimension_numbers();
    if (dnums.spatial_dimensions_size() > 3) {
      return false;
    }

    // CuDNN does not accept zero-element arguments
    if (ShapeUtil::HasZeroElements(hlo.operand(0)->shape()) ||
        ShapeUtil::HasZeroElements(hlo.operand(1)->shape())) {
      return false;
    }

    return true;
  }

  // Backward convolution.
  if (hlo.opcode() == HloOpcode::kFusion &&
      (hlo.fusion_kind() == HloInstruction::FusionKind::kConvBackwardFilter ||
       hlo.fusion_kind() == HloInstruction::FusionKind::kConvBackwardInput)) {
    return true;
  }

  return false;
}

bool ImplementedAsLibraryCall(const HloInstruction& hlo) {
  return ImplementedAsGemm(hlo) || ImplementedAsDnnConvolution(hlo);
}

bool IsReductionToVector(const HloInstruction& reduce) {
  if (HloOpcode::kReduce != reduce.opcode()) {
    return false;
  }
  const HloInstruction* input = reduce.operand(0);
  std::vector<int64> dims_to_keep;
  for (int64 dim = 0; dim < input->shape().dimensions().size(); ++dim) {
    if (!std::count(reduce.dimensions().begin(), reduce.dimensions().end(),
                    dim)) {
      dims_to_keep.push_back(dim);
    }
  }
  return LayoutUtil::AreDimensionsConsecutive(input->shape().layout(),
                                              dims_to_keep) &&
         ShapeUtil::Equal(reduce.shape(), ShapeUtil::FilterDimensions(
                                              [&dims_to_keep](int64 dim) {
                                                return std::count(
                                                    dims_to_keep.begin(),
                                                    dims_to_keep.end(), dim);
                                              },
                                              input->shape()));
}

// This emits a device-side call to
// "i32 vprintf(i8* fmt, arguments_type* arguments)" in the driver; see
// http://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#system-calls
llvm::Value* EmitPrintf(tensorflow::StringPiece fmt,
                        tensorflow::gtl::ArraySlice<llvm::Value*> arguments,
                        llvm::IRBuilder<>* builder) {
  std::vector<llvm::Type*> argument_types;
  for (auto argument : arguments) {
    argument_types.push_back(argument->getType());
  }
  auto* arguments_type = llvm::StructType::create(argument_types);
  llvm::Value* arguments_ptr = builder->CreateAlloca(arguments_type);
  for (size_t i = 0; i < arguments.size(); ++i) {
    builder->CreateStore(
        arguments[i],
        builder->CreateGEP(arguments_ptr,
                           {builder->getInt64(0), builder->getInt32(i)}));
  }
  return builder->CreateCall(
      builder->GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
          "vprintf",
          llvm::FunctionType::get(builder->getInt32Ty(),
                                  {builder->getInt8Ty()->getPointerTo(),
                                   arguments_type->getPointerTo()},
                                  /*isVarArg=*/false)),
      {builder->CreateGlobalStringPtr(llvm_ir::AsStringRef(fmt)),
       arguments_ptr});
}

llvm::Value* EmitShuffleDown(llvm::Value* value, llvm::Value* offset,
                             llvm::IRBuilder<>* builder) {
  int bit_width = value->getType()->getPrimitiveSizeInBits();

  // Special case for efficiency
  if (value->getType()->isFloatTy() && bit_width == 32) {
    return llvm_ir::EmitCallToIntrinsic(
        llvm::Intrinsic::nvvm_shfl_down_f32,
        {value, offset, builder->getInt32(kWarpSize - 1)}, {}, builder);
  }

  // We must split values wider than 32 bits as the "shfl" instruction operates
  // on 32-bit values.
  int num_segments = CeilOfRatio(bit_width, 32);
  llvm::Value* x = builder->CreateBitCast(
      builder->CreateZExt(
          builder->CreateBitCast(value, builder->getIntNTy(bit_width)),
          builder->getIntNTy(32 * num_segments)),
      llvm::VectorType::get(builder->getInt32Ty(), num_segments));
  for (int i = 0; i < num_segments; ++i) {
    x = builder->CreateInsertElement(
        x,
        llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::nvvm_shfl_down_i32,
                                     {builder->CreateExtractElement(x, i),
                                      offset, builder->getInt32(kWarpSize - 1)},
                                     {}, builder),
        i);
  }
  return builder->CreateBitCast(
      builder->CreateTrunc(
          builder->CreateBitCast(x, builder->getIntNTy(32 * num_segments)),
          builder->getIntNTy(bit_width)),
      value->getType());
}

}  // namespace gpu
}  // namespace xla
