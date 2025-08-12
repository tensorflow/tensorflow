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

#include "xla/backends/cpu/onednn_emitter.h"

#include <cstddef>
#include <cstdint>

#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/onednn_fusion.h"
#include "xla/backends/cpu/onednn_fusion_graph.h"
#include "xla/backends/cpu/runtime/onednn/onednn_interop.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

// A mapping from HloInstruction to oneDNN logical tensor.
using LogicalTensorMap =
    absl::flat_hash_map<const HloInstruction*, dnnl::graph::logical_tensor>;

//===----------------------------------------------------------------------===//
// XLA <-> oneDNN type conversion library.
//===----------------------------------------------------------------------===//

static absl::StatusOr<dnnl::graph::logical_tensor::data_type> OneDnnDatatype(
    const PrimitiveType& type) {
  switch (type) {
    case F32:
      return dnnl::graph::logical_tensor::data_type::f32;
    default:
      return InvalidArgument("Unsupported oneDNN data type: %s",
                             primitive_util::LowercasePrimitiveTypeName(type));
  }
}

static absl::StatusOr<dnnl::graph::op::kind> OneDnnUnaryOperator(
    const HloOpcode& opcode) {
  switch (opcode) {
    case HloOpcode::kExp:
      return dnnl::graph::op::kind::Exp;
    default:
      return InvalidArgument("Unsupported oneDNN unary operator: %s",
                             HloOpcodeString(opcode));
  }
}

static absl::StatusOr<dnnl::graph::op::kind> OneDnnBinaryOperator(
    const HloOpcode& opcode) {
  switch (opcode) {
    case HloOpcode::kAdd:
      return dnnl::graph::op::kind::Add;
    case HloOpcode::kMultiply:
      return dnnl::graph::op::kind::Multiply;
    case HloOpcode::kDot:
      return dnnl::graph::op::kind::MatMul;
    default:
      return InvalidArgument("Unsupported oneDNN unary operator: %s",
                             HloOpcodeString(opcode));
  }
}

static dnnl::graph::logical_tensor::dims OneDnnDimensions(const Shape& shape) {
  dnnl::graph::logical_tensor::dims dims;
  for (auto& dim : shape.dimensions()) {
    dims.push_back(dim);
  }
  return dims;
}

static dnnl::graph::logical_tensor::dims OneDnnStrides(const Shape& shape) {
  dnnl::graph::logical_tensor::dims strides(shape.dimensions_size());
  int64_t stride = 1;
  for (int i : shape.layout().minor_to_major()) {
    strides.at(i) = stride;
    stride *= shape.dimensions(i);
  }
  return strides;
}

//===----------------------------------------------------------------------===//
// XLA <-> oneDNN emitters.
//===----------------------------------------------------------------------===//

static absl::StatusOr<dnnl::graph::logical_tensor> FindLogicalTensor(
    const LogicalTensorMap& logical_tensors, const HloInstruction* instr) {
  if (auto it = logical_tensors.find(instr); it != logical_tensors.end()) {
    return it->second;
  }
  return Internal("Can't fine oneDNN logical tensor for instruction %s",
                  instr->ToString());
}

static absl::StatusOr<dnnl::graph::logical_tensor> CreateLogicalTensor(
    size_t tensor_id, const Shape& shape) {
  TF_ASSIGN_OR_RETURN(auto type, OneDnnDatatype(shape.element_type()));

  dnnl::graph::logical_tensor::dims dims = OneDnnDimensions(shape);
  dnnl::graph::logical_tensor::dims strides = OneDnnStrides(shape);

  return dnnl::graph::logical_tensor(tensor_id, type, dims, strides);
}

static absl::StatusOr<dnnl::graph::logical_tensor> DefineParameter(
    const HloInstruction* param) {
  VLOG(3) << absl::StreamFormat("Define logical tensor for parameter: %s",
                                param->ToString());

  return CreateLogicalTensor(param->parameter_number(), param->shape());
}

static absl::StatusOr<dnnl::graph::logical_tensor> DefineUnaryOp(
    dnnl::graph::graph& graph, size_t op_id, LogicalTensorMap& logical_tensors,
    const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define logical tensor value for unary op: %s",
                                instr->ToString());

  TF_ASSIGN_OR_RETURN(auto unary_op, OneDnnUnaryOperator(instr->opcode()));

  TF_ASSIGN_OR_RETURN(auto input,
                      FindLogicalTensor(logical_tensors, instr->operand(0)));

  size_t output_id = logical_tensors.size();
  TF_ASSIGN_OR_RETURN(auto output,
                      CreateLogicalTensor(output_id, instr->shape()));

  VLOG(3) << absl::StreamFormat("  tensors: input=%d, output=%d",
                                input.get_id(), output.get_id());

  dnnl::graph::op op(op_id, unary_op, {input}, {output});
  ONEDNN_RETURN_IF_ERROR(graph.add_op(op));

  return output;
}

static absl::StatusOr<dnnl::graph::logical_tensor> DefineBinaryOp(
    dnnl::graph::graph& graph, size_t op_id, LogicalTensorMap& logical_tensors,
    const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define logical tensor value for binary op: %s",
                                instr->ToString());

  TF_ASSIGN_OR_RETURN(auto binary_op, OneDnnBinaryOperator(instr->opcode()));

  TF_ASSIGN_OR_RETURN(auto lhs,
                      FindLogicalTensor(logical_tensors, instr->operand(0)));
  TF_ASSIGN_OR_RETURN(auto rhs,
                      FindLogicalTensor(logical_tensors, instr->operand(1)));

  size_t output_id = logical_tensors.size();
  TF_ASSIGN_OR_RETURN(auto output,
                      CreateLogicalTensor(output_id, instr->shape()));

  VLOG(3) << absl::StreamFormat("  tensors: lhs=%d, rhs=%d, output=%d",
                                lhs.get_id(), rhs.get_id(), output.get_id());

  dnnl::graph::op op(op_id, binary_op, {lhs, rhs}, {output});
  ONEDNN_RETURN_IF_ERROR(graph.add_op(op));

  return output;
}

static absl::StatusOr<dnnl::graph::logical_tensor> DefineMatMul(
    dnnl::graph::graph& graph, size_t op_id, LogicalTensorMap& logical_tensors,
    const HloInstruction* instr) {
  // Verify that this Dot is supported by XNNPACK.
  const DotDimensionNumbers& dnums = instr->dot_dimension_numbers();
  const Shape& lhs_shape = instr->operand(0)->shape();
  const Shape& rhs_shape = instr->operand(1)->shape();
  TF_ASSIGN_OR_RETURN(
      bool is_supported,
      IsOneDnnDotSupported(dnums, lhs_shape, rhs_shape, instr->shape()));

  if (!is_supported) {
    return InvalidArgument("Unsupported oneDNN Dot op variation: %s",
                           instr->ToString());
  }

  VLOG(3) << absl::StreamFormat("Define logical tensor value for MatMul: %s",
                                instr->ToString());

  TF_ASSIGN_OR_RETURN(auto matmul_op, OneDnnBinaryOperator(instr->opcode()));
  TF_ASSIGN_OR_RETURN(auto lhs,
                      FindLogicalTensor(logical_tensors, instr->operand(0)));
  TF_ASSIGN_OR_RETURN(auto rhs,
                      FindLogicalTensor(logical_tensors, instr->operand(1)));

  size_t output_id = logical_tensors.size();
  TF_ASSIGN_OR_RETURN(auto output,
                      CreateLogicalTensor(output_id, instr->shape()));

  VLOG(3) << absl::StreamFormat("  tensors: lhs=%d, rhs=%d, output=%d",
                                lhs.get_id(), rhs.get_id(), output.get_id());

  dnnl::graph::op op(op_id, matmul_op, {lhs, rhs}, {output});
  ONEDNN_RETURN_IF_ERROR(graph.add_op(op));

  return output;
}

//===----------------------------------------------------------------------===//
// Emit oneDNN graph for the given HLO computation.
//===----------------------------------------------------------------------===//

static absl::StatusOr<OneDnnFusion> EmitOneDnnFusion(
    const HloComputation* computation) {
  VLOG(3) << "Emit oneDNN graph for computation: " << computation->name();

  dnnl::graph::graph graph(dnnl::engine::kind::cpu);

  // Traverse fused computation in post-order and define oneDNN operations
  // corresponding to each HLO instruction.
  LogicalTensorMap logical_tensors;
  auto instructions = computation->MakeInstructionPostOrder();

  size_t op_id = 0;

  for (const HloInstruction* instr : instructions) {
    switch (instr->opcode()) {
      case HloOpcode::kParameter: {
        TF_ASSIGN_OR_RETURN(logical_tensors[instr], DefineParameter(instr));
      } break;

      // Unary elementwise ops.
      case HloOpcode::kExp: {
        TF_ASSIGN_OR_RETURN(
            logical_tensors[instr],
            DefineUnaryOp(graph, op_id++, logical_tensors, instr));
      } break;

      // Binary elementwise ops.
      case HloOpcode::kAdd:
      case HloOpcode::kMultiply: {
        TF_ASSIGN_OR_RETURN(
            logical_tensors[instr],
            DefineBinaryOp(graph, op_id++, logical_tensors, instr));
      } break;

      case HloOpcode::kDot: {
        TF_ASSIGN_OR_RETURN(
            logical_tensors[instr],
            DefineMatMul(graph, op_id++, logical_tensors, instr));
      } break;

      default: {
        return InvalidArgument("Unsupported oneDNN fusion instruction: %s",
                               instr->ToString());
      }
    }
  }

  // Finalize the graph after visiting all instructions.
  graph.finalize();

  // Collect logical tensors for all parameters.
  std::vector<dnnl::graph::logical_tensor> arguments;
  for (auto p : computation->parameter_instructions()) {
    arguments.push_back(logical_tensors.at(p));
  }

  // Root instruction defines the logical tensor for the result.
  std::vector<dnnl::graph::logical_tensor> results = {
      logical_tensors.at(computation->root_instruction())};

  return OneDnnFusion{std::move(arguments), std::move(results),
                      std::move(graph)};
}

absl::StatusOr<absl::AnyInvocable<absl::StatusOr<OneDnnFusion>()>>
EmitOneDnnFusionBuilder(const HloComputation* computation) {
  // We do not support non-array parameters for oneDNN operations.
  for (auto& param : computation->parameter_instructions()) {
    if (!param->shape().IsArray()) {
      return InvalidArgument(
          "oneDNN fusion parameters must have array shapes, got %s",
          param->shape().ToString());
    }
  }

  // Result also must be a single array.
  if (!computation->root_instruction()->shape().IsArray()) {
    return InvalidArgument("oneDNN fusion result must be an array, got %s",
                           computation->root_instruction()->shape().ToString());
  }

  return [computation] { return EmitOneDnnFusion(computation); };
}

}  // namespace xla::cpu
