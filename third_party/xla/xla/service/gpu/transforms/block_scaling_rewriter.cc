/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/block_scaling_rewriter.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

// Expand builder into a new instruction that will replace the old one.
absl::StatusOr<HloInstruction*> ExpandInstructionUsingBuilder(
    XlaBuilder& builder, HloInstruction* old_instruction) {
  TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());
  TF_ASSIGN_OR_RETURN(
      HloComputation * computation,
      XlaComputationToHloComputation(xla_computation,
                                     old_instruction->parent()->parent()));
  return old_instruction->parent()->AddInstruction(HloInstruction::CreateCall(
      old_instruction->shape(), old_instruction->operands(), computation));
}

// Determine block size from the shapes.
absl::StatusOr<int> GetBlockSize(const Shape& quant_shape,
                                 const Shape& scale_shape) {
  int rank = quant_shape.dimensions_size();
  TF_RET_CHECK(rank >= 1 && rank == scale_shape.dimensions_size());
  TF_RET_CHECK(quant_shape.dimensions().subspan(0, rank - 1) ==
               scale_shape.dimensions().subspan(0, rank - 1));
  int m = quant_shape.dimensions(rank - 1);
  int n = scale_shape.dimensions(rank - 1);
  TF_RET_CHECK(m > 0 && n > 0 && m % n == 0);
  return m / n;
}

// ----- Quantization

// Build HLO for quantize op.
absl::StatusOr<XlaOp> BuildQuantize(XlaBuilder& builder,
                                    const Shape& input_shape,
                                    const Shape& output_shape) {
  // Get block size from output shape.
  const Shape& quant_shape = output_shape.tuple_shapes(0);
  const Shape& scale_shape = output_shape.tuple_shapes(1);
  TF_ASSIGN_OR_RETURN(int block_size, GetBlockSize(quant_shape, scale_shape));

  // Reshape input into blocks.
  std::vector<int64_t> new_dims(scale_shape.dimensions().begin(),
                                scale_shape.dimensions().end());
  new_dims.push_back(block_size);
  XlaOp input = Parameter(&builder, 0, input_shape, "input");
  XlaOp input_blocks = Reshape(input, new_dims);

  // Calculate AMAX (maximum absolute value per block).
  XlaBuilder amax_builder("amax");
  Shape scalar = ShapeUtil::MakeShape(input_shape.element_type(), {});
  XlaOp out = Max(Abs(Parameter(&amax_builder, 0, scalar, "a")),
                  Abs(Parameter(&amax_builder, 1, scalar, "b")));
  TF_ASSIGN_OR_RETURN(XlaComputation amax_comp, amax_builder.Build(out));
  XlaOp amax = Reduce(input_blocks, ConstantLiteral(&builder, Literal(scalar)),
                      amax_comp, {scale_shape.dimensions_size()});

  // Use EMAX of the quantization type as the denominator.
  double emax_value =
      1ll << (primitive_util::OverflowExponent(quant_shape.element_type()) - 1);
  Literal denominator_literal(scalar);
  TF_RETURN_IF_ERROR(denominator_literal.SetFromDouble({}, emax_value));
  XlaOp denominator = ConstantLiteral(&builder, denominator_literal);
  XlaOp amax_norm = Div(amax, denominator);

  // Calculate scale tensor values and convert back to input type.
  XlaOp scale = ConvertElementType(amax_norm, scale_shape.element_type());
  XlaOp scale_cvt = ConvertElementType(scale, scalar.element_type());

  // Broadcast scale to input shape.
  std::vector<int64_t> broadcast_dims(scale_shape.dimensions_size());
  absl::c_iota(broadcast_dims, 0);
  XlaOp scale_bc = BroadcastInDim(scale_cvt, new_dims, broadcast_dims);
  new_dims.pop_back();
  new_dims.back() *= block_size;
  XlaOp scale_rs = Reshape(scale_bc, new_dims);

  // Divide input by scale to get quantized result.
  XlaOp result = Div(input, scale_rs);
  result = ConvertElementType(result, quant_shape.element_type());
  return Tuple(&builder, {result, scale});
}

// Convert quantize custom call to HLO computation.
absl::StatusOr<HloInstruction*> ExpandQuantizeCustomCall(
    HloInstruction* instruction) {
  // Check operand count and output shape.
  if (instruction->operand_count() != 1) {
    return InvalidArgument("Incorrect number of operands for quantize op");
  }
  if (instruction->shape().tuple_shapes().size() != 2 ||
      instruction->operand(0)->shape().dimensions() !=
          instruction->shape().tuple_shapes(0).dimensions()) {
    return InvalidArgument("Incorrect output shape for quantize op");
  }

  // Build replacement instruction sequence.
  XlaBuilder builder(std::string(instruction->name()));
  TF_RETURN_IF_ERROR(BuildQuantize(builder, instruction->operand(0)->shape(),
                                   instruction->shape())
                         .status());
  return ExpandInstructionUsingBuilder(builder, instruction);
}

// ----- Dequantization

// Build HLO for dequantize op.
absl::StatusOr<XlaOp> BuildDequantize(XlaOp input_op, XlaOp scale_op,
                                      PrimitiveType result_type) {
  // Get block size from input shapes.
  XlaBuilder& builder = *input_op.builder();
  TF_ASSIGN_OR_RETURN(Shape input_shape, builder.GetShape(input_op));
  TF_ASSIGN_OR_RETURN(Shape scale_shape, builder.GetShape(scale_op));
  TF_ASSIGN_OR_RETURN(int block_size, GetBlockSize(input_shape, scale_shape));

  // Convert input parameters to the same type.
  input_op = ConvertElementType(input_op, result_type);
  scale_op = ConvertElementType(scale_op, result_type);

  // Broadcast scale to input shape.
  std::vector<int64_t> new_dims(scale_shape.dimensions().begin(),
                                scale_shape.dimensions().end());
  new_dims.push_back(block_size);
  std::vector<int64_t> broadcast_dims(scale_shape.dimensions_size());
  absl::c_iota(broadcast_dims, 0);
  scale_op = BroadcastInDim(scale_op, new_dims, broadcast_dims);
  new_dims.pop_back();
  new_dims.back() *= block_size;
  scale_op = Reshape(scale_op, new_dims);

  // Multiply input by broadcasted scale.
  return Mul(input_op, scale_op);
}

// Convert dequantize custom call to HLO computation.
absl::StatusOr<HloInstruction*> ExpandDequantizeCustomCall(
    HloInstruction* instruction) {
  // Check operand count and output shape.
  if (instruction->operand_count() != 2) {
    return InvalidArgument("Incorrect number of operands for dequantize op");
  }
  if (instruction->operand(0)->shape().dimensions() !=
      instruction->shape().dimensions()) {
    return InvalidArgument("Incorrect output shape for dequantize op");
  }

  // Build replacement instruction sequence.
  XlaBuilder builder(std::string(instruction->name()));
  TF_RETURN_IF_ERROR(
      BuildDequantize(
          Parameter(&builder, 0, instruction->operand(0)->shape(), "input"),
          Parameter(&builder, 1, instruction->operand(1)->shape(), "scale"),
          instruction->shape().element_type())
          .status());
  return ExpandInstructionUsingBuilder(builder, instruction);
}

// ----- Block scaled dot (cuDNN)

enum class CudnnMxType {
  // Not a supported composite type.
  UNSUPPORTED_TYPE,
  // Input: E4M3FN, scale: E8M0FNU, block size: 32.
  MXFP8_E4M3FN,
  // Input: E5M2, scale: E8M0FNU, block size: 32.
  MXFP8_E5M2,
  // Input: E2M1FN, scale: E4M3FN, block size: 16.
  NVFP4,
};

CudnnMxType GetCudnnMxType(const Shape& input_shape, const Shape& scale_shape) {
  // Determine the block size from shapes.
  int block_size = GetBlockSize(input_shape, scale_shape).value_or(0);

  // MXFP8: the input could be either E4M3FN or E5M2.
  if (input_shape.element_type() == PrimitiveType::F8E4M3FN &&
      scale_shape.element_type() == PrimitiveType::F8E8M0FNU &&
      block_size == BlockScalingRewriter::kBlockSizeMXFP8) {
    return CudnnMxType::MXFP8_E4M3FN;
  }
  if (input_shape.element_type() == PrimitiveType::F8E5M2 &&
      scale_shape.element_type() == PrimitiveType::F8E8M0FNU &&
      block_size == BlockScalingRewriter::kBlockSizeMXFP8) {
    return CudnnMxType::MXFP8_E5M2;
  }

  // NVFP4: the input is E2M1FN and the scale is E4M3FN.
  if (input_shape.element_type() == PrimitiveType::F4E2M1FN &&
      scale_shape.element_type() == PrimitiveType::F8E4M3FN &&
      block_size == BlockScalingRewriter::kBlockSizeNVFP4) {
    return CudnnMxType::NVFP4;
  }

  return CudnnMxType::UNSUPPORTED_TYPE;
}

bool IsSupportedByCudnn(CudnnMxType lhs, CudnnMxType rhs) {
  // cuDNN supports mixing input types for MXFP8, but the E5M2/E5M2 combination
  // is not supported.
  return (lhs == CudnnMxType::MXFP8_E4M3FN &&
          rhs == CudnnMxType::MXFP8_E4M3FN) ||
         (lhs == CudnnMxType::MXFP8_E4M3FN && rhs == CudnnMxType::MXFP8_E5M2) ||
         (lhs == CudnnMxType::MXFP8_E5M2 && rhs == CudnnMxType::MXFP8_E4M3FN) ||
         (lhs == CudnnMxType::NVFP4 && rhs == CudnnMxType::NVFP4);
}

// Reshape inputs to shapes compatible with cuDNN.
absl::StatusOr<std::tuple<XlaOp, XlaOp, int64_t>> BuildScaledDotInputs(
    XlaOp input_op, XlaOp scale_op) {
  // Get shapes from the inputs.
  XlaBuilder& builder = *input_op.builder();
  TF_ASSIGN_OR_RETURN(Shape input_shape, builder.GetShape(input_op));
  TF_ASSIGN_OR_RETURN(Shape scale_shape, builder.GetShape(scale_op));
  TF_RET_CHECK(input_shape.dimensions_size() == 2 ||
               input_shape.dimensions_size() == 3);

  // Calculate output shape size.
  int64_t batch_size =
      input_shape.dimensions_size() == 3 ? input_shape.dimensions(0) : 1;
  int64_t size_contracting = input_shape.dimensions().back();
  int64_t size_noncontracting =
      input_shape.dimensions(input_shape.dimensions_size() - 2);
  int64_t scale_contracting = scale_shape.dimensions().back();

  // Reshape inputs, if necessary.
  if (input_shape.dimensions_size() != 3) {
    input_op = Reshape(input_op, {1, size_noncontracting, size_contracting});
    scale_op = Reshape(scale_op, {1, size_noncontracting, scale_contracting});
  }

  // cuDNN kernel imposes constraints on the input shape sizes.
  const int64_t kInputNonContractingTileSize = 128;
  const int64_t kScaleContractingTileSize = 4;

  // Pad inputs, if necessary.
  if (size_noncontracting % kInputNonContractingTileSize != 0 ||
      scale_contracting % kScaleContractingTileSize != 0) {
    // Calculate new output shape sizes.
    int64_t padded_noncontracting =
        RoundUpTo(size_noncontracting, kInputNonContractingTileSize);
    int64_t padded_contracting =
        RoundUpTo(scale_contracting, kScaleContractingTileSize);

    // Build padding configs.
    PaddingConfig input_padding_config = MakeNoPaddingConfig(/*rank=*/3);
    if (size_noncontracting != padded_noncontracting) {
      input_padding_config.mutable_dimensions(1)->set_edge_padding_high(
          padded_noncontracting - size_noncontracting);
    }
    PaddingConfig scale_padding_config = input_padding_config;
    if (scale_contracting != padded_contracting) {
      scale_padding_config.mutable_dimensions(2)->set_edge_padding_high(
          padded_contracting - scale_contracting);
    }

    // Build padding ops with zero neutral value.
    if (size_noncontracting != padded_noncontracting) {
      input_op = Pad(input_op, Zero(&builder, input_shape.element_type()),
                     input_padding_config);
    }
    scale_op = Pad(scale_op, Zero(&builder, scale_shape.element_type()),
                   scale_padding_config);
  }

  // Swizzle scales to match the cuDNN kernel.
  //
  // Transposing scales is necessary to match the `scale_vec::1X` layout in
  // TMEM. This transpose can potentially be done in the kernel (at the cost of
  // using non-vectorized loads or using an extra shared memory buffer).
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
  TF_ASSIGN_OR_RETURN(Shape scale_valid_shape, builder.GetShape(scale_op));
  int64_t scale_rows = scale_valid_shape.dimensions(1);
  int64_t scale_cols = scale_valid_shape.dimensions(2);
  scale_op =
      Reshape(scale_op, {batch_size, scale_rows / kInputNonContractingTileSize,
                         4, 32, scale_cols / kScaleContractingTileSize,
                         kScaleContractingTileSize});
  scale_op = Transpose(scale_op, {0, 1, 4, 3, 2, 5});
  scale_op = Reshape(scale_op, scale_valid_shape.dimensions());

  return std::make_tuple(input_op, scale_op, size_noncontracting);
}

// Build HLO for cuDNN custom call op.
absl::StatusOr<XlaOp> BuildCudnnScaledDot(XlaOp lhs_input, XlaOp rhs_input,
                                          XlaOp lhs_scale, XlaOp rhs_scale,
                                          const DotDimensionNumbers& dnums,
                                          PrimitiveType result_type) {
  // Get inputs from parameters.
  TF_ASSIGN_OR_RETURN(auto lhs_ops_and_size,
                      BuildScaledDotInputs(lhs_input, lhs_scale));
  auto [lhs_input_op, lhs_scale_op, lhs_size] = lhs_ops_and_size;

  TF_ASSIGN_OR_RETURN(auto rhs_ops_and_size,
                      BuildScaledDotInputs(rhs_input, rhs_scale));
  auto [rhs_input_op, rhs_scale_op, rhs_size] = rhs_ops_and_size;

  // Calculate output shape.
  XlaBuilder& builder = *lhs_input.builder();
  TF_ASSIGN_OR_RETURN(Shape lhs_shape, builder.GetShape(lhs_input_op));
  TF_ASSIGN_OR_RETURN(Shape rhs_shape, builder.GetShape(rhs_input_op));
  Shape result_shape = ShapeUtil::MakeShape(
      result_type, {lhs_shape.dimensions(0), lhs_shape.dimensions(1),
                    rhs_shape.dimensions(1)});
  Shape scratch_shape = ShapeUtil::MakeShape(PrimitiveType::U8, {0});
  Shape output_shape = ShapeUtil::MakeTupleShape({result_shape, scratch_shape});

  // Build custom call to cuDNN.
  std::string custom_call_target{kCudnnBlockScaledDotCallTarget};
  XlaOp custom_call = CustomCall(
      &builder, custom_call_target,
      {lhs_input_op, rhs_input_op, lhs_scale_op, rhs_scale_op}, output_shape);
  XlaOp result = GetTupleElement(custom_call, 0);

  // Slice the result, if necessary.
  if (lhs_size != lhs_shape.dimensions(1) ||
      rhs_size != rhs_shape.dimensions(1)) {
    std::vector<int64_t> limit{lhs_shape.dimensions(0), lhs_size, rhs_size};
    result = Slice(result, {0, 0, 0}, limit, {1, 1, 1});
  }
  return result;
}

// ----- Block scaled dot (general)

// Build HLO for scaled dot op.
absl::StatusOr<XlaOp> BuildBlockScaledDot(
    XlaBuilder& builder, const HloInstruction* lhs_input,
    const HloInstruction* rhs_input, const HloInstruction* lhs_scale,
    const HloInstruction* rhs_scale, const DotDimensionNumbers& dnums,
    PrimitiveType result_type, bool allow_cudnn) {
  // Get dot LHS parameter(s).
  XlaOp lhs_op = Parameter(&builder, 0, lhs_input->shape(), "lhs");
  XlaOp lhs_scale_op = Parameter(&builder, 2, lhs_scale->shape(), "lhs_scale");

  // Get dot RHS parameter(s).
  XlaOp rhs_op = Parameter(&builder, 1, rhs_input->shape(), "rhs");
  XlaOp rhs_scale_op;
  if (rhs_scale != nullptr) {
    rhs_scale_op = Parameter(&builder, 3, rhs_scale->shape(), "rhs_scale");
  }

  // Use cuDNN kernel, if possible.
  if (allow_cudnn && rhs_scale_op.valid() &&
      IsSupportedByCudnn(
          GetCudnnMxType(lhs_input->shape(), lhs_scale->shape()),
          GetCudnnMxType(rhs_input->shape(), rhs_scale->shape()))) {
    return BuildCudnnScaledDot(lhs_op, rhs_op, lhs_scale_op, rhs_scale_op,
                               dnums, result_type);
  }

  // Build general dot op.
  TF_ASSIGN_OR_RETURN(lhs_op,
                      BuildDequantize(lhs_op, lhs_scale_op, result_type));
  if (rhs_scale_op.valid()) {
    TF_ASSIGN_OR_RETURN(rhs_op,
                        BuildDequantize(rhs_op, rhs_scale_op, result_type));
  }
  return DotGeneral(lhs_op, rhs_op, dnums, /*precision_config=*/nullptr,
                    /*preferred_element_type=*/result_type);
}

// Convert scaled dot custom call to HLO computation.
absl::StatusOr<HloInstruction*> ExpandBlockScaledDotCustomCall(
    HloInstruction* instruction, bool allow_cudnn) {
  PrimitiveType result_type = instruction->shape().element_type();

  // Check operand count.
  if (instruction->operand_count() != 3 && instruction->operand_count() != 4) {
    return InvalidArgument(
        "Incorrect number of operands for block scaled dot op");
  }

  // Check output shape.
  const Shape& lhs_shape = instruction->operand(0)->shape();
  const Shape& rhs_shape = instruction->operand(1)->shape();
  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(lhs_shape.dimensions_size() - 1);
  dnums.add_rhs_contracting_dimensions(rhs_shape.dimensions_size() - 1);
  if (lhs_shape.dimensions_size() == 3) {
    dnums.add_lhs_batch_dimensions(0);
    dnums.add_rhs_batch_dimensions(0);
  }

  TF_ASSIGN_OR_RETURN(Shape inferred_shape,
                      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape,
                                                      dnums, result_type));
  if (inferred_shape != instruction->shape()) {
    return InvalidArgument("Incorrect output shape for block scaled dot op");
  }

  // Build replacement instruction sequence.
  XlaBuilder builder(std::string(instruction->name()));
  auto operands = absl::MakeSpan(instruction->operands());
  TF_ASSIGN_OR_RETURN(
      XlaOp block_scaled_dot,
      BuildBlockScaledDot(builder, operands[0], operands[1], operands[2],
                          operands.size() == 4 ? operands[3] : nullptr, dnums,
                          result_type, allow_cudnn));

  // Reshape to the expected output shape.
  // This should only happen when a unit-sized dimension is added by the pass.
  TF_ASSIGN_OR_RETURN(Shape result_shape, builder.GetShape(block_scaled_dot));
  if (result_shape != instruction->shape()) {
    CHECK_EQ(ShapeUtil::ElementsIn(instruction->shape()),
             ShapeUtil::ElementsIn(result_shape));
    Reshape(instruction->shape(), block_scaled_dot);
  }
  return ExpandInstructionUsingBuilder(builder, instruction);
}

}  // namespace

bool BlockScalingRewriter::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kCustomCall &&
         (instruction->custom_call_target() == kQuantizeCustomCallTarget ||
          instruction->custom_call_target() == kDequantizeCustomCallTarget ||
          instruction->custom_call_target() == kBlockScaledDotCustomCallTarget);
}

absl::StatusOr<HloInstruction*> BlockScalingRewriter::ExpandInstruction(
    HloInstruction* instruction) {
  if (instruction->custom_call_target() == kQuantizeCustomCallTarget) {
    return ExpandQuantizeCustomCall(instruction);
  }
  if (instruction->custom_call_target() == kDequantizeCustomCallTarget) {
    return ExpandDequantizeCustomCall(instruction);
  }
  if (instruction->custom_call_target() == kBlockScaledDotCustomCallTarget) {
    return ExpandBlockScaledDotCustomCall(instruction, allow_cudnn_);
  }
  LOG(FATAL) << "Unexpected custom call target: "
             << instruction->custom_call_target();
}

}  // namespace xla::gpu
