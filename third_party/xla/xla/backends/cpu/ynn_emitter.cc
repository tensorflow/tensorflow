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

#include "xla/backends/cpu/ynn_emitter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
#include "xla/backends/cpu/ynn_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// A mapping from HloInstruction to YNNPACK subgraph tensor id.
using TensorIdMap = absl::flat_hash_map<const HloInstruction*, uint32_t>;

//===----------------------------------------------------------------------===//
// XLA <-> YNNPACK type conversion library.
//===----------------------------------------------------------------------===//

static std::vector<size_t> YnnDimensions(const Shape& shape) {
  absl::Span<const int64_t> dims = shape.dimensions();
  return {dims.begin(), dims.end()};
}

//===----------------------------------------------------------------------===//
// XLA <-> YNNPACK emitters.
//===----------------------------------------------------------------------===//

static absl::StatusOr<uint32_t> FindTensorValue(const TensorIdMap& tensor_ids,
                                                const HloInstruction* instr) {
  if (auto it = tensor_ids.find(instr); it != tensor_ids.end()) {
    return it->second;
  }
  return Internal("Can't fine YNNPACK tensor value for instruction %s",
                  instr->ToString());
}

static absl::StatusOr<uint32_t> DefineTensorValue(ynn_subgraph_t subgraph,
                                                  const HloInstruction* instr) {
  // We do not support instructions with multiple results (tuples).
  if (!instr->shape().IsArray()) {
    return Internal("Unsupported YNNPACK instruction shape: %s",
                    instr->ToString());
  }

  auto dims = YnnDimensions(instr->shape());
  TF_ASSIGN_OR_RETURN(auto type, YnnType(instr->shape().element_type()));

  uint32_t tensor_id = YNN_INVALID_VALUE_ID;
  uint32_t tensor_flags = 0;

  // If instruction is a root instruction of the parent computation we assign it
  // an external tensor id corresponding to the result index.
  const HloComputation* computation = instr->parent();
  if (computation->root_instruction() == instr) {
    tensor_id = computation->num_parameters();
    tensor_flags = YNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }

  YNN_RETURN_IF_ERROR(ynn_define_tensor_value(
      subgraph, type, dims.size(), dims.data(), /*data=*/nullptr,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*scale_id=*/YNN_INVALID_VALUE_ID, tensor_flags, &tensor_id));
  return tensor_id;
}

static absl::StatusOr<uint32_t> DefineConstant(
    ynn_subgraph_t subgraph, std::vector<std::unique_ptr<Literal>>& literals,
    const HloInstruction* instr) {
  // We do not support instructions with multiple results (tuples).
  if (!instr->shape().IsArray()) {
    return Internal("Unsupported YNNPACK instruction shape: %s",
                    instr->ToString());
  }

  auto dims = YnnDimensions(instr->shape());
  TF_ASSIGN_OR_RETURN(auto type, YnnType(instr->shape().element_type()));

  uint32_t tensor_id = YNN_INVALID_VALUE_ID;

  literals.push_back(instr->literal().CloneToUnique());
  const void* value = literals.back()->untyped_data();

  YNN_RETURN_IF_ERROR(ynn_define_tensor_value(
      subgraph, type, dims.size(), dims.data(), /*data=*/value,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*scale_id=*/YNN_INVALID_VALUE_ID,
      /*flags=*/0, &tensor_id));

  return tensor_id;
}

static absl::StatusOr<uint32_t> DefineParameter(ynn_subgraph_t subgraph,
                                                const HloInstruction* param) {
  VLOG(3) << absl::StreamFormat("Define tensor value for parameter: %s",
                                param->ToString());

  auto dims = YnnDimensions(param->shape());
  TF_ASSIGN_OR_RETURN(auto type, YnnType(param->shape().element_type()));

  uint32_t tensor_id = param->parameter_number();
  YNN_RETURN_IF_ERROR(ynn_define_tensor_value(
      subgraph, type, dims.size(), dims.data(), /*data=*/nullptr,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*scale_id=*/YNN_INVALID_VALUE_ID, YNN_VALUE_FLAG_EXTERNAL_INPUT,
      &tensor_id));

  return tensor_id;
}

static absl::StatusOr<uint32_t> DefineBitcastOp(ynn_subgraph_t subgraph,
                                                TensorIdMap& tensor_ids,
                                                const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for bitcast op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kBitcast);
  const HloInstruction* input = instr->operand(0);
  CHECK_EQ(input->shape().element_type(), instr->shape().element_type());
  TF_ASSIGN_OR_RETURN(auto in, FindTensorValue(tensor_ids, input));
  TF_ASSIGN_OR_RETURN(auto out, DefineTensorValue(subgraph, instr));

  auto dims = YnnDimensions(instr->shape());
  YNN_RETURN_IF_ERROR(ynn_define_static_reshape(subgraph, dims.size(),
                                                dims.data(), in, &out,
                                                /*flags=*/0));
  return out;
}

static absl::StatusOr<uint32_t> DefineUnaryOp(ynn_subgraph_t subgraph,
                                              TensorIdMap& tensor_ids,
                                              const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for unary op: %s",
                                instr->ToString());
  TF_ASSIGN_OR_RETURN(auto unary_op, YnnUnaryOperator(instr->opcode()));

  TF_ASSIGN_OR_RETURN(auto in, FindTensorValue(tensor_ids, instr->operand(0)));
  TF_ASSIGN_OR_RETURN(auto out, DefineTensorValue(subgraph, instr));

  VLOG(3) << absl::StreamFormat("  tensors: in=%d, out=%d", in, out);

  YNN_RETURN_IF_ERROR(
      ynn_define_unary(subgraph, unary_op, in, &out, /*flags=*/0));

  return out;
}

static absl::StatusOr<uint32_t> DefineBinaryOp(ynn_subgraph_t subgraph,
                                               TensorIdMap& tensor_ids,
                                               const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for binary op: %s",
                                instr->ToString());

  TF_ASSIGN_OR_RETURN(auto binary_op, YnnBinaryOperator(instr->opcode()));

  TF_ASSIGN_OR_RETURN(auto lhs, FindTensorValue(tensor_ids, instr->operand(0)));
  TF_ASSIGN_OR_RETURN(auto rhs, FindTensorValue(tensor_ids, instr->operand(1)));
  TF_ASSIGN_OR_RETURN(auto out, DefineTensorValue(subgraph, instr));

  VLOG(3) << absl::StreamFormat("  tensors: lhs=%d, rhs=%d, out=%d", lhs, rhs,
                                out);

  YNN_RETURN_IF_ERROR(
      ynn_define_binary(subgraph, binary_op, lhs, rhs, &out, /*flags=*/0));

  return out;
}

static absl::StatusOr<uint32_t> DefineReduceOp(ynn_subgraph_t subgraph,
                                               TensorIdMap& tensor_ids,
                                               const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for reduce op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kReduce);
  const HloReduceInstruction* reduce_instr = Cast<HloReduceInstruction>(instr);
  const HloInstruction* input = instr->operand(0);
  const HloInstruction* init = instr->operand(1);
  CHECK_EQ(input->shape().element_type(), instr->shape().element_type());
  CHECK_EQ(init->shape().element_type(), instr->shape().element_type());

  ynn_reduce_operator ynn_reduce_op = ynn_reduce_invalid;
  CHECK_EQ(reduce_instr->to_apply()->num_parameters(), 2);
  CHECK_EQ(reduce_instr->to_apply()->instruction_count(), 3);

  switch (reduce_instr->to_apply()->root_instruction()->opcode()) {
    case HloOpcode::kAdd:
      ynn_reduce_op = ynn_reduce_sum;
      break;
    case HloOpcode::kMaximum:
      ynn_reduce_op = ynn_reduce_max;
      break;
    case HloOpcode::kMinimum:
      ynn_reduce_op = ynn_reduce_min;
      break;
    default:
      LOG(FATAL) << "Unsupported reduction: " << instr->to_apply()->ToString();
  }

  const absl::Span<const int64_t> reduce_dims = reduce_instr->dimensions();
  const std::vector<int32_t> dims(reduce_dims.begin(), reduce_dims.end());
  TF_ASSIGN_OR_RETURN(auto in, FindTensorValue(tensor_ids, input));
  TF_ASSIGN_OR_RETURN(auto init_id, FindTensorValue(tensor_ids, init));
  TF_ASSIGN_OR_RETURN(auto out, DefineTensorValue(subgraph, instr));

  YNN_RETURN_IF_ERROR(
      ynn_define_reduce(subgraph, ynn_reduce_op, /*num_axes=*/dims.size(),
                        /*axes=*/dims.data(), in, init_id, &out, /*flags=*/0));
  return out;
}

//===----------------------------------------------------------------------===//
// Emit YNNPACK subgraph for the given HLO computation.
//===----------------------------------------------------------------------===//

static absl::StatusOr<YnnSubgraph> EmitYnnSubgraph(
    const HloComputation* computation,
    std::vector<std::unique_ptr<Literal>>& literals) {
  VLOG(3) << "Emit YNNPACK subgraph for computation: " << computation->name();

  TF_ASSIGN_OR_RETURN(
      YnnSubgraph subgraph, CreateYnnSubgraph([&](ynn_subgraph_t* subgraph) {
        return ynn_create_subgraph(
            /*external_value_ids=*/computation->num_parameters() + 1,
            YnnFlags(computation->parent()->config().debug_options()),
            subgraph);
      }));

  // Traverse fused computation in post-order and define YNNPACK operations
  // corresponding to each HLO instruction.
  TensorIdMap tensor_ids;
  auto instructions = computation->MakeInstructionPostOrder();

  for (const HloInstruction* instr : instructions) {
    if (!IsLayoutSupportedByYnn(instr->shape())) {
      return InvalidArgument(
          "Instruction with unsupported layout in YNN fusion: %s",
          instr->ToString());
    }

    if (instr->IsConstant()) {
      if (!IsConstantSupportedByYnn(instr)) {
        return InvalidArgument(
            "Unsupported constant instruction in YNN fusion: %s",
            instr->ToString());
      }
      TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                          DefineConstant(subgraph.get(), literals, instr));
      continue;
    }

    if (instr->IsElementwise()) {
      if (!IsElementwiseOpSupportedByYnn(instr)) {
        return InvalidArgument(
            "Unsupported elementwise instruction in YNN fusion: %s",
            instr->ToString());
      }
      if (instr->operand_count() == 1) {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineUnaryOp(subgraph.get(), tensor_ids, instr));
      } else if (instr->operand_count() == 2) {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineBinaryOp(subgraph.get(), tensor_ids, instr));
      } else {
        LOG(FATAL) << "Unexpected operand count " << instr->operand_count();
      }
      continue;
    }

    switch (instr->opcode()) {
      case HloOpcode::kParameter: {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineParameter(subgraph.get(), instr));
      } break;

      case HloOpcode::kBitcast: {
        if (!IsBitcastOpSupportedByYnn(instr)) {
          return InvalidArgument(
              "Unsupported bitcast instruction in YNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineBitcastOp(subgraph.get(), tensor_ids, instr));
      } break;

      case HloOpcode::kReduce: {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineReduceOp(subgraph.get(), tensor_ids, instr));
      } break;

      default: {
        return InvalidArgument("Unsupported fusion instruction: %s",
                               instr->ToString());
      }
    }
  }

  ynn_status status = ynn_optimize_subgraph(
      subgraph.get(), /*threadpool=*/nullptr, /*flags=*/0);
  TF_RETURN_IF_ERROR(YnnStatusToStatus(status));

  return subgraph;
}

//===----------------------------------------------------------------------===//
// Emit YNNPACK subgraph for the given HLO dot instruction.
//===----------------------------------------------------------------------===//

// TODO(ashaposhnikov): Use DefineBatchMatrixMultiply in EmitYnnSubgraph.
static ynn_status DefineBatchMatrixMultiply(ynn_subgraph_t subgraph,
                                            uint32_t input1_id,
                                            uint32_t input2_id,
                                            uint32_t output_id, size_t b_rank,
                                            bool transpose_b) {
  if (transpose_b) {
    uint32_t input2_id_transposed = YNN_INVALID_VALUE_ID;
    std::array<int32_t, YNN_MAX_TENSOR_RANK> perm;
    std::iota(perm.begin(), perm.end(), 0);
    CHECK_LT(b_rank, YNN_MAX_TENSOR_RANK);
    std::swap(perm[b_rank - 1], perm[b_rank - 2]);
    ynn_status status = ynn_define_static_transpose(
        subgraph,
        /*num_dims=*/b_rank, perm.data(), input2_id, &input2_id_transposed,
        /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
    input2_id = input2_id_transposed;
  }

  return ynn_define_dot(subgraph, /*num_k_dims=*/1, input1_id, input2_id,
                        YNN_INVALID_VALUE_ID, &output_id, /*flags=*/0);
}

static absl::StatusOr<YnnSubgraph> EmitYnnDotSubgraph(
    const HloDotInstruction* dot,
    std::vector<std::unique_ptr<Literal>>& literals,
    absl::Span<const se::DeviceMemoryBase> arguments_buffers,
    bool capture_rhs) {
  TF_ASSIGN_OR_RETURN(
      YnnSubgraph subgraph, CreateYnnSubgraph([&](ynn_subgraph_t* subgraph) {
        return ynn_create_subgraph(
            /*external_value_ids=*/3,
            YnnFlags(dot->GetModule()->config().debug_options()), subgraph);
      }));

  uint32_t lhs_id = 0;
  uint32_t rhs_id = 1;
  uint32_t out_id = 2;

  const HloInstruction* lhs = dot->operand(0);
  const HloInstruction* rhs = dot->operand(1);

  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();
  const Shape& out_shape = dot->shape();

  auto dims = [](absl::Span<const int64_t> dims) -> std::vector<size_t> {
    return {dims.begin(), dims.end()};
  };

  std::vector<size_t> lhs_dims = dims(lhs_shape.dimensions());
  std::vector<size_t> rhs_dims = dims(rhs_shape.dimensions());
  std::vector<size_t> out_dims = dims(out_shape.dimensions());

  TF_ASSIGN_OR_RETURN(ynn_type ynn_lhs_type, YnnType(lhs_shape.element_type()));
  TF_ASSIGN_OR_RETURN(ynn_type ynn_rhs_type, YnnType(rhs_shape.element_type()));
  TF_ASSIGN_OR_RETURN(ynn_type ynn_out_type, YnnType(out_shape.element_type()));

  const uint32_t input_tensor_flags = YNN_VALUE_FLAG_EXTERNAL_INPUT;
  YNN_RETURN_IF_ERROR(ynn_define_tensor_value(
      subgraph.get(), ynn_lhs_type, lhs_dims.size(), lhs_dims.data(),
      /*data=*/nullptr,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*scale_id=*/YNN_INVALID_VALUE_ID, input_tensor_flags, &lhs_id));

  YNN_RETURN_IF_ERROR(ynn_define_tensor_value(
      subgraph.get(), ynn_rhs_type, rhs_dims.size(), rhs_dims.data(),
      capture_rhs ? arguments_buffers[1].opaque() : nullptr,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*scale_id=*/YNN_INVALID_VALUE_ID, input_tensor_flags, &rhs_id));

  const uint32_t output_tensor_flags = YNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  YNN_RETURN_IF_ERROR(ynn_define_tensor_value(
      subgraph.get(), ynn_out_type, out_dims.size(), out_dims.data(),
      /*data=*/nullptr,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*scale_id=*/YNN_INVALID_VALUE_ID, output_tensor_flags, &out_id));

  DotDimensionNumbers dot_dimensions = dot->dot_dimension_numbers();
  TF_ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                      rhs_shape, out_shape));

  TF_ASSIGN_OR_RETURN(DotCanonicalDims dot_canonical_dims,
                      GetDotCanonicalDims(dot_dimensions, dot_shape));

  const size_t b_rank = rhs_shape.dimensions().size();
  const bool transpose_b = !dot_canonical_dims.rhs_canonical;
  YNN_RETURN_IF_ERROR(DefineBatchMatrixMultiply(subgraph.get(), lhs_id, rhs_id,
                                                out_id, b_rank, transpose_b));

  ynn_status status = ynn_optimize_subgraph(
      subgraph.get(), /*threadpool=*/nullptr, /*flags=*/0);
  TF_RETURN_IF_ERROR(YnnStatusToStatus(status));

  return subgraph;
}

absl::StatusOr<absl::AnyInvocable<absl::StatusOr<YnnSubgraph>(
    absl::Span<const se::DeviceMemoryBase> arguments_buffers)>>
EmitYnnFusionBuilder(const HloComputation* computation) {
  // We do not support non-array parameters for YNNPACK operations.
  for (auto& param : computation->parameter_instructions()) {
    if (!param->shape().IsArray()) {
      return InvalidArgument(
          "YNNPACK fusion parameters must have array shapes, got %s",
          param->shape().ToString());
    }
  }

  // Result also must be a single array.
  if (!computation->root_instruction()->shape().IsArray()) {
    return InvalidArgument("YNNPACK fusion result must be an array, got %s",
                           computation->root_instruction()->shape().ToString());
  }

  return [computation, literals = std::vector<std::unique_ptr<Literal>>()](
             absl::Span<const se::DeviceMemoryBase> arguments_buffers) mutable {
    return EmitYnnSubgraph(computation, literals);
  };
}

absl::StatusOr<absl::AnyInvocable<absl::StatusOr<YnnSubgraph>(
    absl::Span<const se::DeviceMemoryBase> arguments_buffers)>>
EmitYnnDotBuilder(const HloDotInstruction* dot, bool capture_rhs) {
  return [dot, capture_rhs, literals = std::vector<std::unique_ptr<Literal>>()](
             absl::Span<const se::DeviceMemoryBase> arguments_buffers) mutable {
    return EmitYnnDotSubgraph(dot, literals, arguments_buffers, capture_rhs);
  };
}

}  // namespace xla::cpu
