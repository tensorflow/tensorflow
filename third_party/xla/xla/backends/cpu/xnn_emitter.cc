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

#include <algorithm>
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
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"
#include "xla/backends/cpu/xnn_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
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

static absl::StatusOr<uint32_t> DefineTensorValue(
    xnn_subgraph_t subgraph, xnn_datatype type, absl::Span<const size_t> dims) {
  uint32_t tensor_id = XNN_INVALID_VALUE_ID;
  uint32_t tensor_flags = 0;

  XNN_RETURN_IF_ERROR(xnn_define_tensor_value(
      subgraph, type, dims.size(), dims.data(), nullptr,
      /*external_id=*/tensor_id, tensor_flags, &tensor_id));

  return tensor_id;
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

static absl::StatusOr<uint32_t> DefineBitcastOp(xnn_subgraph_t subgraph,
                                                TensorIdMap& tensor_ids,
                                                const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for bitcast op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kBitcast);
  const HloInstruction* input = instr->operand(0);
  CHECK_EQ(input->shape().element_type(), instr->shape().element_type());
  TF_ASSIGN_OR_RETURN(auto in, FindTensorValue(tensor_ids, input));
  TF_ASSIGN_OR_RETURN(auto out, DefineTensorValue(subgraph, instr));

  auto dims = XnnDimensions(instr->shape());
  XNN_RETURN_IF_ERROR(xnn_define_static_reshape(subgraph, dims.size(),
                                                dims.data(), in, out,
                                                /*flags=*/0));
  return out;
}

static absl::StatusOr<uint32_t> DefineBroadcastOp(xnn_subgraph_t subgraph,
                                                  TensorIdMap& tensor_ids,
                                                  const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for broadcast op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kBroadcast);
  const HloBroadcastInstruction* broadcast_instr =
      Cast<HloBroadcastInstruction>(instr);
  const HloInstruction* input = broadcast_instr->operand(0);
  CHECK_EQ(input->shape().element_type(), instr->shape().element_type());

  const absl::Span<const int64_t> input_dims = input->shape().dimensions();
  const absl::Span<const int64_t> output_dims = instr->shape().dimensions();
  const absl::Span<const int64_t> dims = broadcast_instr->dimensions();
  CHECK(std::is_sorted(dims.begin(), dims.end()));
  CHECK_LE(input_dims.size(), output_dims.size());

  const size_t num_new_axes = output_dims.size() - input_dims.size();
  // New axis positions used by XNNPACK expand_dims.
  std::vector<size_t> xnn_expand_dims_new_axes;
  xnn_expand_dims_new_axes.reserve(num_new_axes);
  std::vector<size_t> xnn_expand_dims_dimensions;
  xnn_expand_dims_dimensions.reserve(output_dims.size());

  // Mask used by XNNPACK broadcast.
  std::vector<size_t> xnn_new_shape;
  xnn_new_shape.reserve(output_dims.size());

  for (size_t dim_idx = 0; dim_idx < output_dims.size(); ++dim_idx) {
    const auto it = std::find(dims.begin(), dims.end(), dim_idx);
    if (it == dims.end()) {
      // New dimension case.
      xnn_expand_dims_new_axes.push_back(dim_idx);
      xnn_expand_dims_dimensions.push_back(1u);
      // Broadcasted dimension.
      xnn_new_shape.push_back(output_dims[dim_idx]);
    } else {
      // Pass through the input dimension.
      const size_t input_dim_idx = it - dims.begin();
      CHECK_EQ(*it, dim_idx);
      const size_t input_dim = input_dims[input_dim_idx];
      CHECK_EQ(input_dim, output_dims[dim_idx]);
      xnn_expand_dims_dimensions.push_back(input_dim);
      // 0 means keeping the dimension of the input.
      // See the description of xnn_define_static_broadcast in xnnpack.h
      xnn_new_shape.push_back(0u);
    }
  }

  CHECK_EQ(xnn_expand_dims_dimensions.size(), output_dims.size());
  CHECK_EQ(xnn_expand_dims_new_axes.size(), num_new_axes);
  CHECK_EQ(xnn_new_shape.size(), output_dims.size());

  TF_ASSIGN_OR_RETURN(auto type, XnnDatatype(input->shape().element_type()));
  TF_ASSIGN_OR_RETURN(auto in, FindTensorValue(tensor_ids, input));
  TF_ASSIGN_OR_RETURN(
      auto xnn_dims_expanded,
      DefineTensorValue(subgraph, type, xnn_expand_dims_dimensions));
  TF_ASSIGN_OR_RETURN(auto xnn_broadcast, DefineTensorValue(subgraph, instr));

  XNN_RETURN_IF_ERROR(xnn_define_static_expand_dims(
      subgraph, num_new_axes, xnn_expand_dims_new_axes.data(), in,
      xnn_dims_expanded, /*flags=*/0));

  XNN_RETURN_IF_ERROR(xnn_define_static_broadcast(
      subgraph, xnn_new_shape.size(), xnn_new_shape.data(), xnn_dims_expanded,
      xnn_broadcast, /*flags=*/0));

  return xnn_broadcast;
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

  // In XLA, broadcasts are explicit ops, allowing XNNPACK to assume there is no
  // broadcasting in the elementwise operation itself, which simplifies data
  // dependencies.
  const uint32_t flags = XNN_FLAG_NO_BROADCAST;
  XNN_RETURN_IF_ERROR(xnn_define_binary(subgraph, binary_op, &params, lhs, rhs,
                                        out, /*flags=*/flags));

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
      IsDotSupportedByXnn(dnums, lhs_shape, rhs_shape, instr->shape()));

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

  // In XLA, broadcasts are explicit ops, allowing XNNPACK to assume there is no
  // broadcasting in the elementwise operation itself, which simplifies data
  // dependencies.
  uint32_t flags = XNN_FLAG_NO_BROADCAST;
  // IsXnnDotSupported has verified that rhs_contracting_dimensions has size 1.
  if (dnums.rhs_contracting_dimensions(0) !=
      dnums.rhs_batch_dimensions_size()) {
    flags |= XNN_FLAG_TRANSPOSE_B;
  }
  XNN_RETURN_IF_ERROR(xnn_define_batch_matrix_multiply(subgraph, lhs, rhs, out,
                                                       /*flags=*/flags));

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
    if (!IsLayoutSupportedByXnn(instr->shape())) {
      XNN_LOG_IF_ERROR(xnn_delete_subgraph(subgraph));
      return InvalidArgument(
          "Instruction with unsupported layout in XNN fusion: %s",
          instr->ToString());
    }

    if (instr->IsConstant()) {
      if (!IsConstantSupportedByXnn(instr)) {
        XNN_LOG_IF_ERROR(xnn_delete_subgraph(subgraph));
        return InvalidArgument(
            "Unsupported constant instruction in XNN fusion: %s",
            instr->ToString());
      }
      TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                          DefineConstant(subgraph, literals, instr));
      continue;
    }

    if (instr->IsElementwise()) {
      if (!IsElementwiseOpSupportedByXnn(instr)) {
        XNN_LOG_IF_ERROR(xnn_delete_subgraph(subgraph));
        return InvalidArgument(
            "Unsupported elementwise instruction in XNN fusion: %s",
            instr->ToString());
      }
      if (instr->operand_count() == 1) {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineUnaryOp(subgraph, tensor_ids, instr));
      } else if (instr->operand_count() == 2) {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineBinaryOp(subgraph, tensor_ids, instr));
      } else {
        LOG(FATAL) << "Unexpected operand count " << instr->operand_count();
      }
      continue;
    }

    switch (instr->opcode()) {
      case HloOpcode::kParameter: {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineParameter(subgraph, instr));
      } break;

      case HloOpcode::kBitcast: {
        if (!IsBitcastOpSupportedByXnn(instr)) {
          XNN_LOG_IF_ERROR(xnn_delete_subgraph(subgraph));
          return InvalidArgument(
              "Unsupported bitcast instruction in XNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineBitcastOp(subgraph, tensor_ids, instr));
      } break;

      case HloOpcode::kBroadcast: {
        if (!IsBroadcastOpSupportedByXnn(instr)) {
          XNN_LOG_IF_ERROR(xnn_delete_subgraph(subgraph));
          return InvalidArgument(
              "Unsupported broadcast instruction in XNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineBroadcastOp(subgraph, tensor_ids, instr));
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
