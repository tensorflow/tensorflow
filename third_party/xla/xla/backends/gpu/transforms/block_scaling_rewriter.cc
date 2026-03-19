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

#include "xla/backends/gpu/transforms/block_scaling_rewriter.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

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

  // Fix broadcast layouts (they cannot be inferred correctly).
  for (HloInstruction* instruction : computation->instructions()) {
    auto broadcast = DynCast<HloBroadcastInstruction>(instruction);
    if (broadcast != nullptr && !LayoutUtil::IsMonotonicWithDim0Major(
                                    broadcast->operand(0)->shape().layout())) {
      // Previous instruction is a convert, next one is a reshape.
      int rank = broadcast->shape().dimensions().size();
      const HloInstruction* convert = broadcast->operand(0);
      CHECK(convert->opcode() == HloOpcode::kConvert &&
            convert->shape().dimensions().size() == rank - 1);
      HloInstruction* reshape = broadcast->users()[0];
      CHECK(reshape->opcode() == HloOpcode::kReshape &&
            reshape->shape().dimensions().size() == rank - 1);

      // Increase the layout index of the dimensions after the last one.
      // Example: {2,0,1} -> {3,0,2,1}
      int last_idx = convert->shape().layout().minor_to_major().back();
      auto broadcast_layout = broadcast->mutable_shape()->mutable_layout();
      for (int i = 0; i < rank - 1; ++i) {
        int idx = convert->shape().layout().minor_to_major(i);
        broadcast_layout->set_minor_to_major(i, idx + (idx >= last_idx));
      }
      broadcast_layout->set_minor_to_major(rank - 1, last_idx);
      *reshape->mutable_shape()->mutable_layout() = convert->shape().layout();
    }
  }

  return old_instruction->parent()->AddInstruction(HloInstruction::CreateCall(
      old_instruction->shape(), old_instruction->operands(), computation));
}

// Determine block size from the shapes.
absl::StatusOr<int> GetBlockSize(const Shape& quant_shape,
                                 const Shape& scale_shape) {
  int rank = quant_shape.dimensions().size();
  TF_RET_CHECK(rank >= 1 && rank == scale_shape.dimensions().size());
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
  std::vector<int64_t> broadcast_dims(scale_shape.dimensions().size());
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

  // Output/scale dimensions should match, except the last one (block scaled).
  const Shape& output_shape = instruction->shape().tuple_shapes(0);
  const Shape& scale_shape = instruction->shape().tuple_shapes(1);
  int64_t rank = output_shape.dimensions().size();
  if (output_shape.dimensions().subspan(0, rank - 1) !=
      scale_shape.dimensions().subspan(0, rank - 1)) {
    return InvalidArgument("Output and scale shape dimensions do not match");
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
  std::vector<int64_t> broadcast_dims(scale_shape.dimensions().size());
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

  // Input/scale dimensions should match, except the last one (block scaled).
  const Shape& input_shape = instruction->operand(0)->shape();
  const Shape& scale_shape = instruction->operand(1)->shape();
  int64_t rank = input_shape.dimensions().size();
  if (input_shape.dimensions().subspan(0, rank - 1) !=
      scale_shape.dimensions().subspan(0, rank - 1)) {
    return InvalidArgument("Input and scale shape dimensions do not match");
  }

  // Build replacement instruction sequence.
  XlaBuilder builder(std::string(instruction->name()));
  TF_RETURN_IF_ERROR(
      BuildDequantize(Parameter(&builder, 0, input_shape, "input"),
                      Parameter(&builder, 1, scale_shape, "scale"),
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

CudnnMxType GetCudnnMxType(const Shape& input_shape, const Shape& scale_shape,
                           std::optional<int64_t> block_size) {
  // Non-default layout is not supported.
  if (!LayoutUtil::IsMonotonicWithDim0Major(input_shape.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(scale_shape.layout())) {
    return CudnnMxType::UNSUPPORTED_TYPE;
  }

  // Determine the block size from shapes, unless explicitly given.
  int64_t actual_block_size =
      block_size.has_value()
          ? block_size.value()
          : GetBlockSize(input_shape, scale_shape).value_or(0);
  int64_t contracting_size = input_shape.dimensions().back();

  // MXFP8: the input could be either E4M3FN or E5M2.
  if (input_shape.element_type() == PrimitiveType::F8E4M3FN &&
      scale_shape.element_type() == PrimitiveType::F8E8M0FNU &&
      actual_block_size == BlockScalingRewriter::kBlockSizeMXFP8 &&
      contracting_size % actual_block_size == 0) {
    return CudnnMxType::MXFP8_E4M3FN;
  }
  if (input_shape.element_type() == PrimitiveType::F8E5M2 &&
      scale_shape.element_type() == PrimitiveType::F8E8M0FNU &&
      actual_block_size == BlockScalingRewriter::kBlockSizeMXFP8 &&
      contracting_size % actual_block_size == 0) {
    return CudnnMxType::MXFP8_E5M2;
  }

  // NVFP4: the input is E2M1FN and the scale is E4M3FN.
  if (input_shape.element_type() == PrimitiveType::F4E2M1FN &&
      scale_shape.element_type() == PrimitiveType::F8E4M3FN &&
      actual_block_size == BlockScalingRewriter::kBlockSizeNVFP4 &&
      contracting_size % actual_block_size == 0) {
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
absl::StatusOr<std::tuple<XlaOp, XlaOp, int64_t>> BuildCudnnScaledDotInput(
    XlaOp input_op, XlaOp scale_op, std::optional<int64_t> block_size,
    bool pad_input) {
  // Get shapes from the inputs.
  XlaBuilder& builder = *input_op.builder();
  TF_ASSIGN_OR_RETURN(Shape input_shape, builder.GetShape(input_op));
  TF_ASSIGN_OR_RETURN(Shape scale_shape, builder.GetShape(scale_op));
  int64_t rank = input_shape.dimensions().size();
  TF_RET_CHECK(rank == 2 || rank == 3);

  // Calculate output shape size.
  int64_t batch_size = rank == 3 ? input_shape.dimensions(0) : 1;
  int64_t size_contracting = input_shape.dimensions().back();
  int64_t size_noncontracting = input_shape.dimensions(rank - 2);
  int64_t scale_contracting = scale_shape.dimensions().back();
  int64_t scale_noncontracting = scale_shape.dimensions(rank - 2);

  // Validate input/shape dimensions.
  int64_t actual_block_size =
      block_size.has_value() ? block_size.value()
                             : GetBlockSize(input_shape, scale_shape).value();
  TF_RET_CHECK(size_contracting <= scale_contracting * actual_block_size);
  TF_RET_CHECK(size_noncontracting <= scale_noncontracting);
  TF_RET_CHECK(rank == 2 || scale_shape.dimensions(0) == batch_size);

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

    // Pad input tensor, if necessary.
    if (size_noncontracting != padded_noncontracting && pad_input) {
      PaddingConfig input_padding_config = MakeNoPaddingConfig(rank);
      input_padding_config.mutable_dimensions(rank - 2)->set_edge_padding_high(
          padded_noncontracting - size_noncontracting);
      input_op = Pad(input_op, Zero(&builder, input_shape.element_type()),
                     input_padding_config);
    }

    // Pad scale tensor, if necessary.
    if (scale_noncontracting != padded_noncontracting ||
        scale_contracting != padded_contracting) {
      PaddingConfig scale_padding_config = MakeNoPaddingConfig(rank);
      scale_padding_config.mutable_dimensions(rank - 2)->set_edge_padding_high(
          padded_noncontracting - scale_noncontracting);
      scale_padding_config.mutable_dimensions(rank - 1)->set_edge_padding_high(
          padded_contracting - scale_contracting);
      scale_op = Pad(scale_op, Zero(&builder, scale_shape.element_type()),
                     scale_padding_config);
    }
  }

  // Swizzle scales to match the cuDNN kernel.
  //
  // Transposing scales is necessary to match the `scale_vec::1X` layout in
  // TMEM. This transpose can potentially be done in the kernel (at the cost of
  // using non-vectorized loads or using an extra shared memory buffer).
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
  TF_ASSIGN_OR_RETURN(Shape scale_valid_shape, builder.GetShape(scale_op));
  int64_t scale_rows = scale_valid_shape.dimensions(rank - 2);
  int64_t scale_cols = scale_valid_shape.dimensions(rank - 1);
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
                                          XlaOp global_scale,
                                          const DotDimensionNumbers& dnums,
                                          PrimitiveType result_type,
                                          std::optional<int64_t> block_size,
                                          se::dnn::VersionInfo cudnn_version) {
  bool cudnn_supports_global_scale =
      cudnn_version >= kCudnnSupportsBlockScaledDotWithGlobalScale;

  // Get inputs from parameters.
  TF_ASSIGN_OR_RETURN(auto lhs_ops_and_size,
                      BuildCudnnScaledDotInput(lhs_input, lhs_scale, block_size,
                                               /*pad_input=*/true));
  auto [lhs_input_op, lhs_scale_op, lhs_size] = lhs_ops_and_size;

  TF_ASSIGN_OR_RETURN(auto rhs_ops_and_size,
                      BuildCudnnScaledDotInput(rhs_input, rhs_scale, block_size,
                                               /*pad_input=*/true));
  auto [rhs_input_op, rhs_scale_op, rhs_size] = rhs_ops_and_size;

  // Calculate output shape.
  XlaBuilder& builder = *lhs_input.builder();
  TF_ASSIGN_OR_RETURN(Shape lhs_shape, builder.GetShape(lhs_input_op));
  TF_ASSIGN_OR_RETURN(Shape rhs_shape, builder.GetShape(rhs_input_op));
  int rank = lhs_shape.dimensions().size();
  std::vector<int64_t> result_dims{lhs_shape.dimensions(rank - 2),
                                   rhs_shape.dimensions(rank - 2)};
  if (rank == 3) {
    result_dims.insert(result_dims.begin(), lhs_shape.dimensions(0));
  }
  Shape result_shape = ShapeUtil::MakeShape(result_type, result_dims);
  Shape scratch_shape = ShapeUtil::MakeShape(PrimitiveType::U8, {0});
  Shape output_shape = ShapeUtil::MakeTupleShape({result_shape, scratch_shape});

  // Build custom call to cuDNN.
  std::string custom_call_target{kCudnnBlockScaledDotCallTarget};
  std::vector<XlaOp> custom_call_operands{lhs_input_op, rhs_input_op,
                                          lhs_scale_op, rhs_scale_op};
  if (global_scale.valid() && cudnn_supports_global_scale) {
    custom_call_operands.push_back(global_scale);
  }
  XlaOp custom_call = CustomCall(&builder, custom_call_target,
                                 custom_call_operands, output_shape);
  XlaOp result = GetTupleElement(custom_call, 0);

  // Apply global scale outside the graph for older cuDNN versions.
  if (global_scale.valid() && !cudnn_supports_global_scale) {
    result = Mul(result, global_scale,
                 /*broadcast_dimensions=*/{});
  }

  // Slice the result, if necessary.
  if (lhs_size != lhs_shape.dimensions(rank - 2) ||
      rhs_size != rhs_shape.dimensions(rank - 2)) {
    std::vector<int64_t> start(rank, 0);
    std::vector<int64_t> strides(rank, 1);
    result_dims[rank - 2] = lhs_size;
    result_dims[rank - 1] = rhs_size;
    result = Slice(result, start, result_dims, strides);
  }
  return result;
}

// ----- Block scaled dot (general)

// Build HLO for scaled dot op input.
absl::StatusOr<XlaOp> BuildBlockScaledDotInput(
    XlaOp input_op, XlaOp scale_op, PrimitiveType result_type,
    std::optional<int64_t> block_size) {
  // Get shapes of the input and scales.
  XlaBuilder& builder = *input_op.builder();
  TF_ASSIGN_OR_RETURN(Shape input_shape, builder.GetShape(input_op));
  TF_ASSIGN_OR_RETURN(Shape scale_shape, builder.GetShape(scale_op));

  // Make sure the input and scale shapes are compatible (scales may be padded).
  int64_t rank = input_shape.dimensions().size();
  TF_RET_CHECK(scale_shape.dimensions().size() == rank);
  bool truncate_scale = false;

  // Check the batch dimension.
  if (rank == 3) {
    TF_RET_CHECK(input_shape.dimensions(0) == scale_shape.dimensions(0));
  }

  // Check the noncontracting dimension.
  int64_t input_noncontracting_size = input_shape.dimensions(rank - 2);
  int64_t scale_noncontracting_size = scale_shape.dimensions(rank - 2);

  TF_RET_CHECK(input_noncontracting_size <= scale_noncontracting_size);
  truncate_scale |= input_noncontracting_size < scale_noncontracting_size;
  scale_shape.set_dimensions(rank - 2, input_noncontracting_size);

  // Check the contracting dimension if explicit block size is passed.
  if (block_size.has_value()) {
    int64_t input_contracting_size = input_shape.dimensions(rank - 1);
    int64_t scale_contracting_size = scale_shape.dimensions(rank - 1);
    int64_t dq_size = *block_size * scale_contracting_size;

    TF_RET_CHECK(input_contracting_size <= dq_size);
    truncate_scale |= input_contracting_size < dq_size;
    scale_shape.set_dimensions(
        rank - 1, CeilOfRatio(input_contracting_size, *block_size));
  }

  // Slice the scale tensor to match the input shape.
  if (truncate_scale) {
    std::vector<int64_t> start_indices(rank, 0);
    std::vector<int64_t> limit_indices(scale_shape.dimensions().begin(),
                                       scale_shape.dimensions().end());
    std::vector<int64_t> strides(rank, 1);
    scale_op = Slice(scale_op, start_indices, limit_indices, strides);
  }

  return BuildDequantize(input_op, scale_op, result_type);
}

// Build HLO for scaled dot op.
absl::StatusOr<XlaOp> BuildBlockScaledDot(
    XlaBuilder& builder, const HloInstruction* lhs_input,
    const HloInstruction* rhs_input, const HloInstruction* lhs_scale,
    const HloInstruction* rhs_scale, const HloInstruction* global_scale,
    const DotDimensionNumbers& dnums, PrimitiveType result_type,
    std::optional<int64_t> block_size, se::dnn::VersionInfo cudnn_version) {
  // Get dot LHS parameter(s).
  XlaOp lhs_op = Parameter(&builder, 0, lhs_input->shape(), "lhs");
  XlaOp lhs_scale_op = Parameter(&builder, 2, lhs_scale->shape(), "lhs_scale");

  // Get dot RHS parameter(s).
  XlaOp rhs_op = Parameter(&builder, 1, rhs_input->shape(), "rhs");
  XlaOp rhs_scale_op;
  if (rhs_scale != nullptr) {
    rhs_scale_op = Parameter(&builder, 3, rhs_scale->shape(), "rhs_scale");
  }

  // Get global scale parameter, if present.
  XlaOp global_scale_op;
  if (global_scale != nullptr) {
    global_scale_op =
        Parameter(&builder, 4, global_scale->shape(), "global_scale");
  }

  // Use cuDNN kernel, if possible.
  if (cudnn_version >= kCudnnSupportsBlockScaledDot && rhs_scale_op.valid() &&
      IsSupportedByCudnn(
          GetCudnnMxType(lhs_input->shape(), lhs_scale->shape(), block_size),
          GetCudnnMxType(rhs_input->shape(), rhs_scale->shape(), block_size))) {
    return BuildCudnnScaledDot(lhs_op, rhs_op, lhs_scale_op, rhs_scale_op,
                               global_scale_op, dnums, result_type, block_size,
                               std::move(cudnn_version));
  }

  // Build general dot op.
  TF_ASSIGN_OR_RETURN(
      lhs_op,
      BuildBlockScaledDotInput(lhs_op, lhs_scale_op, result_type, block_size));
  if (rhs_scale_op.valid()) {
    TF_ASSIGN_OR_RETURN(
        rhs_op, BuildBlockScaledDotInput(rhs_op, rhs_scale_op, result_type,
                                         block_size));
  }
  XlaOp result = DotGeneral(lhs_op, rhs_op, dnums, /*precision_config=*/nullptr,
                            /*preferred_element_type=*/result_type);

  // Apply global scale, if present.
  if (global_scale_op.valid()) {
    result = Mul(result, global_scale_op,
                 /*broadcast_dimensions=*/{});
  }
  return result;
}

// Convert scaled dot custom call to HLO computation.
absl::StatusOr<HloInstruction*> ExpandBlockScaledDotCustomCall(
    HloInstruction* instruction, se::dnn::VersionInfo cudnn_version) {
  PrimitiveType result_type = instruction->shape().element_type();

  // Check operand count.
  if (instruction->operand_count() < 3 || instruction->operand_count() > 5) {
    return InvalidArgument(
        "Incorrect number of operands for block scaled dot op");
  }

  // Check output shape.
  const Shape& lhs_shape = instruction->operand(0)->shape();
  const Shape& rhs_shape = instruction->operand(1)->shape();
  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(lhs_shape.dimensions().size() - 1);
  dnums.add_rhs_contracting_dimensions(rhs_shape.dimensions().size() - 1);
  if (lhs_shape.dimensions().size() == 3) {
    dnums.add_lhs_batch_dimensions(0);
    dnums.add_rhs_batch_dimensions(0);
  }

  TF_ASSIGN_OR_RETURN(Shape inferred_shape,
                      ShapeInference::InferDotOpShape(lhs_shape, rhs_shape,
                                                      dnums, result_type));
  if (inferred_shape != instruction->shape()) {
    return InvalidArgument("Incorrect output shape for block scaled dot op");
  }

  // Check global scale shape.
  if (instruction->operand_count() == 5) {
    const Shape& global_scale_shape = instruction->operand(4)->shape();
    if (!ShapeUtil::IsScalar(global_scale_shape) ||
        global_scale_shape.element_type() != result_type) {
      return InvalidArgument(
          "Global scale shape must be a scalar with the result's type");
    }
  }

  // If an explicit block size is passed in the backend config, use it.
  // This is needed when the scale tensor is padded, the block size cannot be
  // implied in this case.
  auto backend_config = instruction->backend_config<GpuBackendConfig>();
  std::optional<int64_t> block_size =
      backend_config.ok() &&
              backend_config->has_block_scaled_dot_backend_config()
          ? std::make_optional(
                backend_config->block_scaled_dot_backend_config().block_size())
          : std::nullopt;

  // Build replacement instruction sequence.
  XlaBuilder builder(std::string(instruction->name()));
  auto operands = absl::MakeSpan(instruction->operands());
  TF_RETURN_IF_ERROR(
      BuildBlockScaledDot(builder, operands[0], operands[1], operands[2],
                          operands.size() >= 4 ? operands[3] : nullptr,
                          operands.size() == 5 ? operands[4] : nullptr, dnums,
                          result_type, block_size, std::move(cudnn_version))
          .status());
  return ExpandInstructionUsingBuilder(builder, instruction);
}

// ----- cuDNN scale swizzling

absl::StatusOr<HloComputation*> CreateScaleSwizzleComputation(
    const HloInstruction* input, const HloInstruction* scale) {
  // Create XLA builder and parameters.
  std::string name = absl::StrCat(scale->name(), "_swizzle");
  XlaBuilder builder(name);
  XlaOp input_op = Parameter(&builder, 0, input->shape(), "input");
  XlaOp scale_op = Parameter(&builder, 1, scale->shape(), "scale");

  // Build swizzle computation.
  TF_ASSIGN_OR_RETURN(
      auto ops_and_size,
      BuildCudnnScaledDotInput(input_op, scale_op, /*block_size=*/std::nullopt,
                               /*pad_input=*/false));
  auto [result_input_op, result_scale_op, _] = ops_and_size;
  Tuple(&builder, {result_input_op, result_scale_op});

  TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, builder.Build());
  TF_ASSIGN_OR_RETURN(
      HloComputation * computation,
      XlaComputationToHloComputation(xla_computation, input->GetModule()));

  for (HloInstruction* instr : computation->instructions()) {
    // Replace reshapes with bitcasts (post layout assignment).
    if (instr->opcode() == HloOpcode::kReshape) {
      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(
          instr, computation->AddInstruction(HloInstruction::CreateBitcast(
                     instr->shape(), instr->mutable_operand(0)))));
    }
    // Fix transpose layouts (generated as no-ops).
    if (instr->opcode() == HloOpcode::kTranspose) {
      *instr->mutable_shape()->mutable_layout() =
          LayoutUtil::GetDefaultLayoutForShape(instr->shape());
    }
  }
  return computation;
}

absl::Status SliceScaledDotOperands(HloInstruction* scaled_dot) {
  // Create scaled dot operation with noncontracting dimensions sliced.
  int rank = scaled_dot->shape().dimensions().size();
  HloComputation* computation = scaled_dot->parent();

  // Create slice operations for LHS/RHS.
  std::vector<HloInstruction*> new_operands(scaled_dot->operands().begin(),
                                            scaled_dot->operands().end());
  for (int i = 0; i < 2; ++i) {
    const Shape& input_shape = scaled_dot->operand(i)->shape();
    const Shape& scale_shape = scaled_dot->operand(i + 2)->shape();
    if (input_shape.dimensions(rank - 2) != scale_shape.dimensions(rank - 2)) {
      std::vector<int64_t> start(rank, 0);
      std::vector<int64_t> strides(rank, 1);
      std::vector<int64_t> limit(input_shape.dimensions().begin(),
                                 input_shape.dimensions().end());
      limit[rank - 1] = scale_shape.dimensions(rank - 1);
      new_operands[i + 2] =
          computation->AddInstruction(HloInstruction::CreateSlice(
              ShapeUtil::MakeShape(scale_shape.element_type(), limit),
              scaled_dot->mutable_operand(i + 2), start, limit, strides));
    }
  }

  // Replace scaled dot instruction operands.
  HloInstruction* new_scaled_dot = computation->AddInstruction(
      scaled_dot->CloneWithNewOperands(scaled_dot->shape(), new_operands));
  return computation->ReplaceInstruction(scaled_dot, new_scaled_dot);
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
    return ExpandBlockScaledDotCustomCall(instruction, cudnn_version_);
  }
  LOG(FATAL) << "Unexpected custom call target: "
             << instruction->custom_call_target();
}

bool CudnnScaledDotHelper::IsSupported(
    const HloScaledDotInstruction* scaled_dot) {
  const HloInstruction* lhs_input = scaled_dot->operand(0);
  const HloInstruction* rhs_input = scaled_dot->operand(1);
  const HloInstruction* lhs_scale = scaled_dot->operand(2);
  const HloInstruction* rhs_scale = scaled_dot->operand(3);

  // Input fusion is not supported, as the underlying kernel reads from HBM.
  auto is_parameter = [](const HloInstruction* instr, int index) {
    return instr->opcode() == HloOpcode::kParameter &&
           instr->parameter_number() == index && instr->user_count() == 1;
  };
  if (!is_parameter(lhs_input, 0) || !is_parameter(rhs_input, 1) ||
      !is_parameter(lhs_scale, 2) || !is_parameter(rhs_scale, 3)) {
    return false;
  }

  // The dot dimension numbers must have fixed order: batch dimension first
  // (if present) and contracting dimension last.
  const DotDimensionNumbers& dnums = scaled_dot->dot_dimension_numbers();
  int rank = lhs_input->shape().dimensions().size();
  if (dnums.lhs_contracting_dimensions()[0] != rank - 1 ||
      dnums.rhs_contracting_dimensions()[0] != rank - 1 ||
      (rank == 3 && (dnums.lhs_batch_dimensions()[0] != 0 ||
                     dnums.rhs_batch_dimensions()[0] != 0))) {
    return false;
  }

  // cuDNN kernel supports a subset of block scaled types.
  return IsSupportedByCudnn(
      GetCudnnMxType(lhs_input->shape(), lhs_scale->shape(), std::nullopt),
      GetCudnnMxType(rhs_input->shape(), rhs_scale->shape(), std::nullopt));
}

absl::StatusOr<HloInstruction*> CudnnScaledDotHelper::AddScaleSwizzle(
    HloFusionInstruction* fusion) {
  HloComputation* parent = fusion->parent();
  int rank = fusion->shape().dimensions().size();

  // Add swizzling to LHS/RHS.
  std::vector<HloInstruction*> swizzled_operands(4);
  for (int i = 0; i < 2; ++i) {
    TF_ASSIGN_OR_RETURN(HloComputation * swizzle_computation,
                        CreateScaleSwizzleComputation(fusion->operand(i),
                                                      fusion->operand(i + 2)));
    HloInstruction* call = parent->AddInstruction(HloInstruction::CreateCall(
        swizzle_computation->root_instruction()->shape(),
        {fusion->mutable_operand(i), fusion->mutable_operand(i + 2)},
        swizzle_computation));
    for (int j = 0; j < 2; ++j) {
      swizzled_operands[i + j * 2] =
          parent->AddInstruction(HloInstruction::CreateGetTupleElement(
              call->shape().tuple_shapes(j), call, j));
    }
  }

  // Update fusion computation parameter shapes, if needed.
  HloComputation* computation = fusion->fused_instructions_computation();
  bool need_slicing = false;
  for (int i = 0; i < 4; ++i) {
    HloInstruction* param = computation->parameter_instruction(i);
    const Shape& swizzled_shape = swizzled_operands[i]->shape();
    Shape* param_shape = param->mutable_shape();
    if (*param_shape != swizzled_shape) {
      need_slicing |= param_shape->dimensions(rank - 2) !=
                      swizzled_shape.dimensions(rank - 2);
      *param_shape = swizzled_shape;
    }
  }

  // Replace scaled dot if any inputs need slicing.
  if (need_slicing) {
    HloInstruction* scaled_dot =
        computation->parameter_instruction(0)->users()[0];
    TF_RETURN_IF_ERROR(SliceScaledDotOperands(scaled_dot));
  }

  // Create new fusion with the swizzled operands.
  HloInstruction* new_fusion =
      parent->AddInstruction(HloInstruction::CreateFusion(
          computation->root_instruction()->shape(), fusion->fusion_kind(),
          swizzled_operands, fusion->fused_instructions_computation()));
  TF_RETURN_IF_ERROR(parent->ReplaceInstruction(fusion, new_fusion));
  return new_fusion;
}

}  // namespace xla::gpu
