/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/xnn_emitter.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "xnnpack.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"
#include "xla/backends/cpu/xnn_fusion.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// A mapping from HloInstruction to XNNPACK subgraph tensor id.
using TensorIdMap = absl::flat_hash_map<const HloInstruction*, uint32_t>;

//===----------------------------------------------------------------------===//
// XLA <-> XNNPACK type conversion library.
//===----------------------------------------------------------------------===//

static absl::StatusOr<xnn_datatype> XnnDatatype(const PrimitiveType& type) {
  switch (type) {
    case BF16:
      return xnn_datatype_bf16;
    case F16:
      return xnn_datatype_fp16;
    case F32:
      return xnn_datatype_fp32;
    default:
      return InvalidArgument("Unsupported XNNPACK data type: %s",
                             primitive_util::LowercasePrimitiveTypeName(type));
  }
}

static absl::StatusOr<xnn_unary_operator> XnnUnaryOperator(
    const HloOpcode& opcode) {
  switch (opcode) {
    case HloOpcode::kConvert:
      return xnn_unary_convert;
    default:
      return InvalidArgument("Unsupported XNNPACK unary operator: %s",
                             HloOpcodeString(opcode));
  }
}

static absl::StatusOr<xnn_binary_operator> XnnBinaryOperator(
    const HloOpcode& opcode) {
  switch (opcode) {
    case HloOpcode::kAdd:
      return xnn_binary_add;
    case HloOpcode::kMultiply:
      return xnn_binary_multiply;
    case HloOpcode::kSubtract:
      return xnn_binary_subtract;
    default:
      return InvalidArgument("Unsupported XNNPACK binary operator: %s",
                             HloOpcodeString(opcode));
  }
}

static std::vector<size_t> XnnDimensions(const Shape& shape) {
  std::vector<size_t> dims;
  for (auto& dim : shape.dimensions()) {
    dims.push_back(dim);
  }
  return dims;
}

//===----------------------------------------------------------------------===//
// XLA <-> XNNPACK emitters.
//===----------------------------------------------------------------------===//

static absl::StatusOr<uint32_t> FindTensorValue(const TensorIdMap& tensor_ids,
                                                const HloInstruction* instr) {
  if (auto it = tensor_ids.find(instr); it != tensor_ids.end()) {
    return it->second;
  }
  return Internal("Can't fine XNNPACK tensor value for instruction %s",
                  instr->ToString());
}

static absl::StatusOr<uint32_t> DefineTensorValue(xnn_subgraph_t subgraph,
                                                  const HloInstruction* instr) {
  // We do not support instructions with multiple results (tuples).
  if (!instr->shape().IsArray()) {
    return Internal("Unsupported XNNPACK instruction shape: %s",
                    instr->ToString());
  }

  auto dims = XnnDimensions(instr->shape());
  TF_ASSIGN_OR_RETURN(auto type, XnnDatatype(instr->shape().element_type()));

  uint32_t tensor_id = XNN_INVALID_VALUE_ID;
  uint32_t tensor_flags = 0;

  // If instruction is a root instruction of the parent computation we assign it
  // an external tensor id corresponding to the result index.
  const HloComputation* computation = instr->parent();
  if (computation->root_instruction() == instr) {
    tensor_id = computation->num_parameters();
    tensor_flags = XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, type, dims.size(), dims.data(), nullptr,
      /*external_id=*/tensor_id, tensor_flags, &tensor_id));

  return tensor_id;
}

static absl::StatusOr<uint32_t> DefineConstant(
    xnn_subgraph_t subgraph, std::vector<std::unique_ptr<Literal>>& literals,
    const HloInstruction* instr) {
  // We do not support instructions with multiple results (tuples).
  if (!instr->shape().IsArray()) {
    return Internal("Unsupported XNNPACK instruction shape: %s",
                    instr->ToString());
  }

  auto dims = XnnDimensions(instr->shape());
  TF_ASSIGN_OR_RETURN(auto type, XnnDatatype(instr->shape().element_type()));

  uint32_t tensor_id = XNN_INVALID_VALUE_ID;
  uint32_t tensor_flags = 0;

  literals.push_back(instr->literal().CloneToUnique());
  const void* value = literals.back()->untyped_data();

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, type, dims.size(), dims.data(), value,
      /*external_id=*/tensor_id, tensor_flags, &tensor_id));

  return tensor_id;
}

static absl::StatusOr<uint32_t> DefineParameter(xnn_subgraph_t subgraph,
                                                const HloInstruction* param) {
  VLOG(3) << absl::StreamFormat("Define tensor value for parameter: %s",
                                param->ToString());

  auto dims = XnnDimensions(param->shape());
  TF_ASSIGN_OR_RETURN(auto type, XnnDatatype(param->shape().element_type()));

  uint32_t tensor_id = param->parameter_number();
  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, type, dims.size(), dims.data(), nullptr,
      /*external_id=*/tensor_id, XNN_VALUE_FLAG_EXTERNAL_INPUT, &tensor_id));

  return tensor_id;
}

static absl::StatusOr<uint32_t> DefineUnaryOp(xnn_subgraph_t subgraph,
                                              TensorIdMap& tensor_ids,
                                              const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for unary op: %s",
                                instr->ToString());
  TF_ASSIGN_OR_RETURN(auto unary_op, XnnUnaryOperator(instr->opcode()));

  TF_ASSIGN_OR_RETURN(auto in, FindTensorValue(tensor_ids, instr->operand(0)));
  TF_ASSIGN_OR_RETURN(auto out, DefineTensorValue(subgraph, instr));

  VLOG(3) << absl::StreamFormat("  tensors: in=%d, out=%d", in, out);

  xnn_unary_params params;
  XNN_RETURN_IF_ERROR(
      xnn_define_unary(subgraph, unary_op, &params, in, out, /*flags=*/0));

  return out;
}

static absl::StatusOr<uint32_t> DefineBinaryOp(xnn_subgraph_t subgraph,
                                               TensorIdMap& tensor_ids,
                                               const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for binary op: %s",
                                instr->ToString());

  TF_ASSIGN_OR_RETURN(auto binary_op, XnnBinaryOperator(instr->opcode()));

  TF_ASSIGN_OR_RETURN(auto lhs, FindTensorValue(tensor_ids, instr->operand(0)));
  TF_ASSIGN_OR_RETURN(auto rhs, FindTensorValue(tensor_ids, instr->operand(1)));
  TF_ASSIGN_OR_RETURN(auto out, DefineTensorValue(subgraph, instr));

  VLOG(3) << absl::StreamFormat("  tensors: lhs=%d, rhs=%d, out=%d", lhs, rhs,
                                out);

  xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity()};

  XNN_RETURN_IF_ERROR(xnn_define_binary(subgraph, binary_op, &params, lhs, rhs,
                                        out, /*flags=*/0));

  return out;
}

static absl::StatusOr<uint32_t> DefineBatchMatMul(xnn_subgraph_t subgraph,
                                                  TensorIdMap& tensor_ids,
                                                  const HloInstruction* instr) {
  // Verify that this Dot is supported by XNNPACK.
  const DotDimensionNumbers& dnums = instr->dot_dimension_numbers();
  const Shape& lhs_shape = instr->operand(0)->shape();
  const Shape& rhs_shape = instr->operand(1)->shape();
  TF_ASSIGN_OR_RETURN(
      bool is_supported,
      IsXnnDotSupported(dnums, lhs_shape, rhs_shape, instr->shape()));

  if (!is_supported) {
    if (subgraph != nullptr) XNN_LOG_IF_ERROR(xnn_delete_subgraph(subgraph));
    return InvalidArgument("Unsupported XNNPACK Dot op variation: %s",
                           instr->ToString());
  }

  VLOG(3) << "Define tensor values for batch_matrix_multiply op";

  TF_ASSIGN_OR_RETURN(uint32_t lhs,
                      FindTensorValue(tensor_ids, instr->operand(0)));
  TF_ASSIGN_OR_RETURN(uint32_t rhs,
                      FindTensorValue(tensor_ids, instr->operand(1)));
  TF_ASSIGN_OR_RETURN(uint32_t out, DefineTensorValue(subgraph, instr));

  VLOG(3) << absl::StreamFormat("  tensors: lhs=%d, rhs=%d, out=%d", lhs, rhs,
                                out);

  // IsXnnDotSupported has verified that rhs_contracting_dimensions has size 1.
  bool rhs_canonical =
      dnums.rhs_contracting_dimensions(0) == dnums.rhs_batch_dimensions_size();
  XNN_RETURN_IF_ERROR(xnn_define_batch_matrix_multiply(
      subgraph, lhs, rhs, out,
      /*flags=*/rhs_canonical ? 0 : XNN_FLAG_TRANSPOSE_B));

  return out;
}

//===----------------------------------------------------------------------===//
// Emit XNNPACK subgraph for the given HLO computation.
//===----------------------------------------------------------------------===//

static absl::StatusOr<xnn_subgraph_t> EmitXnnSubgraph(
    const HloComputation* computation,
    std::vector<std::unique_ptr<Literal>>& literals) {
  VLOG(3) << "Emit XNNPACK subgraph for computation: " << computation->name();

  xnn_subgraph_t subgraph = nullptr;
  XNN_RETURN_IF_ERROR(xnn_create_subgraph(
      /*external_value_ids=*/computation->num_parameters() + 1,
      /*flags=*/0, &subgraph));

  // Traverse fused computation in post-order and define XNNPACK operations
  // corresponding to each HLO instruction.
  TensorIdMap tensor_ids;
  auto instructions = computation->MakeInstructionPostOrder();

  for (const HloInstruction* instr : instructions) {
    switch (instr->opcode()) {
      case HloOpcode::kParameter: {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineParameter(subgraph, instr));
      } break;

      case HloOpcode::kConstant: {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineConstant(subgraph, literals, instr));
      } break;

      case HloOpcode::kConvert: {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineUnaryOp(subgraph, tensor_ids, instr));
      } break;

      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply: {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineBinaryOp(subgraph, tensor_ids, instr));
      } break;

      case HloOpcode::kDot: {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineBatchMatMul(subgraph, tensor_ids, instr));
      } break;

      default: {
        XNN_LOG_IF_ERROR(xnn_delete_subgraph(subgraph));
        return InvalidArgument("Unsupported XNNPACK fusion instruction: %s",
                               instr->ToString());
      }
    }
  }

  return subgraph;
}

absl::StatusOr<absl::AnyInvocable<absl::StatusOr<xnn_subgraph_t>()>>
EmitXnnFusionBuilder(const HloComputation* computation) {
  // We do not support non-array parameters for XNNPACK operations.
  for (auto& param : computation->parameter_instructions()) {
    if (!param->shape().IsArray()) {
      return InvalidArgument(
          "XNNPACK fusion parameters must have array shapes, got %s",
          param->shape().ToString());
    }
  }

  // Result also must be a single array.
  if (!computation->root_instruction()->shape().IsArray()) {
    return InvalidArgument("XNNPACK fusion result must be an array, got %s",
                           computation->root_instruction()->shape().ToString());
  }

  return [computation,
          literals = std::vector<std::unique_ptr<Literal>>()]() mutable {
    return EmitXnnSubgraph(computation, literals);
  };
}

}  // namespace xla::cpu
