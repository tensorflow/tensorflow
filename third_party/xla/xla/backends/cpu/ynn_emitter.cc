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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
#include "xla/backends/cpu/ynn_support.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/shape.h"
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
  std::vector<size_t> dims;
  for (auto& dim : shape.dimensions()) {
    dims.push_back(dim);
  }
  return dims;
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
            /*flags=*/0, subgraph);
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

      default: {
        return InvalidArgument("Unsupported fusion instruction: %s",
                               instr->ToString());
      }
    }
  }

  return subgraph;
}

absl::StatusOr<absl::AnyInvocable<absl::StatusOr<YnnSubgraph>()>>
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

  return [computation,
          literals = std::vector<std::unique_ptr<Literal>>()]() mutable {
    return EmitYnnSubgraph(computation, literals);
  };
}

}  // namespace xla::cpu
