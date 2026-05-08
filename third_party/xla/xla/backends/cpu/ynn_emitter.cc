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
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/dot_dims.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
#include "xla/backends/cpu/ynn_support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// A mapping from HloInstruction to YNNPACK subgraph tensor id.
using TensorIdMap = absl::flat_hash_map<const HloInstruction*, uint32_t>;

namespace {

//===----------------------------------------------------------------------===//
// XLA <-> YNNPACK type conversion library.
//===----------------------------------------------------------------------===//

std::vector<size_t> YnnDimensions(const Shape& shape) {
  absl::Span<const int64_t> dims = shape.dimensions();
  return {dims.begin(), dims.end()};
}

absl::StatusOr<double> ReduceIdentity(const HloOpcode& opcode) {
  switch (opcode) {
    case HloOpcode::kAdd:
      return 0.0;
    case HloOpcode::kMaximum:
      return -std::numeric_limits<double>::infinity();
    case HloOpcode::kMinimum:
      return std::numeric_limits<double>::infinity();
    default:
      return InvalidArgument("Unsupported YNNPACK reduce operator: %s",
                             HloOpcodeString(opcode));
  }
}

//===----------------------------------------------------------------------===//
// XLA <-> YNNPACK emitters.
//===----------------------------------------------------------------------===//

absl::StatusOr<uint32_t> FindTensorValue(const TensorIdMap& tensor_ids,
                                         const HloInstruction* instr) {
  if (auto it = tensor_ids.find(instr); it != tensor_ids.end()) {
    return it->second;
  }
  return Internal("Can't fine YNNPACK tensor value for instruction %s",
                  instr->ToString());
}

absl::StatusOr<uint32_t> DefineTensorValue(
    ynn_subgraph_t subgraph, const HloInstruction* instr,
    uint32_t output_id = YNN_INVALID_VALUE_ID) {
  // We do not support instructions with multiple results (tuples).
  if (!instr->shape().IsArray()) {
    return Internal("Unsupported YNNPACK instruction shape: %s",
                    instr->ToString());
  }

  auto dims = YnnDimensions(instr->shape());
  TF_ASSIGN_OR_RETURN(auto type, YnnType(instr->shape().element_type()));

  if (output_id == YNN_INVALID_VALUE_ID) {
    // If instruction is a root instruction of the parent computation we assign
    // it an external tensor id corresponding to the result index.
    const HloComputation* computation = instr->parent();
    if (computation->root_instruction() == instr) {
      output_id = computation->num_parameters();
    }
  }

  uint32_t tensor_flags = 0;
  if (output_id != YNN_INVALID_VALUE_ID) {
    tensor_flags = YNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }

  YNN_RETURN_IF_ERROR(ynn_define_tensor(subgraph, type, dims.size(),
                                        dims.data(), /*data=*/nullptr,
                                        tensor_flags, &output_id));
  return output_id;
}

class Literals {
  absl::Mutex mutex_;
  std::vector<std::unique_ptr<Literal>> literals_;

 public:
  Literals() = default;

  Literals(const Literals&) = delete;
  Literals& operator=(const Literals&) = delete;

  Literals(Literals&& rhs) : literals_(std::move(rhs.literals_)) {}

  Literals& operator=(Literals&& rhs) {
    if (this != &rhs) {
      literals_ = std::move(rhs.literals_);
    }
    return *this;
  }

  const void* Add(std::unique_ptr<Literal> literal) {
    absl::MutexLock lock(mutex_);
    literals_.push_back(std::move(literal));
    return literals_.back()->untyped_data();
  }
};

absl::StatusOr<uint32_t> DefineConstant(ynn_subgraph_t subgraph,
                                        Literals& literals,
                                        const HloInstruction* instr) {
  // We do not support instructions with multiple results (tuples).
  if (!instr->shape().IsArray()) {
    return Internal("Unsupported YNNPACK instruction shape: %s",
                    instr->ToString());
  }

  auto dims = YnnDimensions(instr->shape());
  TF_ASSIGN_OR_RETURN(auto type, YnnType(instr->shape().element_type()));

  uint32_t tensor_id = YNN_INVALID_VALUE_ID;

  const void* value = literals.Add(instr->literal().CloneToUnique());

  YNN_RETURN_IF_ERROR(ynn_define_tensor(subgraph, type, dims.size(),
                                        dims.data(), /*data=*/value,
                                        /*flags=*/0, &tensor_id));

  return tensor_id;
}

absl::StatusOr<uint32_t> DefineParameter(ynn_subgraph_t subgraph,
                                         const HloInstruction* param,
                                         const void* data = nullptr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for parameter: %s",
                                param->ToString());

  auto dims = YnnDimensions(param->shape());
  TF_ASSIGN_OR_RETURN(auto type, YnnType(param->shape().element_type()));

  uint32_t tensor_id = param->parameter_number();
  uint32_t flags = (data == nullptr) ? YNN_VALUE_FLAG_EXTERNAL_INPUT : 0;
  YNN_RETURN_IF_ERROR(ynn_define_tensor(subgraph, type, dims.size(),
                                        dims.data(), /*data=*/data, flags,
                                        &tensor_id));

  return tensor_id;
}

absl::StatusOr<uint32_t> DefineBitcastOp(ynn_subgraph_t subgraph,
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

absl::StatusOr<uint32_t> DefineReshapeOp(ynn_subgraph_t subgraph,
                                         TensorIdMap& tensor_ids,
                                         const HloInstruction* instr) {
  return DefineBitcastOp(subgraph, tensor_ids, instr);
}

absl::StatusOr<uint32_t> DefineTransposeOp(ynn_subgraph_t subgraph,
                                           TensorIdMap& tensor_ids,
                                           const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for transpose op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kTranspose);
  const HloInstruction* input = instr->operand(0);
  TF_ASSIGN_OR_RETURN(auto in, FindTensorValue(tensor_ids, input));
  TF_ASSIGN_OR_RETURN(auto out, DefineTensorValue(subgraph, instr));

  auto dimensions = instr->dimensions();
  std::vector<int32_t> perm(dimensions.begin(), dimensions.end());

  YNN_RETURN_IF_ERROR(ynn_define_static_transpose(subgraph, perm.size(),
                                                  perm.data(), in, &out,
                                                  /*flags=*/0));
  return out;
}

absl::StatusOr<uint32_t> DefineBroadcastOp(ynn_subgraph_t subgraph,
                                           TensorIdMap& tensor_ids,
                                           const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for broadcast op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kBroadcast);
  const HloInstruction* input = instr->operand(0);
  TF_ASSIGN_OR_RETURN(auto in, FindTensorValue(tensor_ids, input));
  TF_ASSIGN_OR_RETURN(auto out, DefineTensorValue(subgraph, instr));

  auto dimensions = instr->dimensions();
  auto output_dims = instr->shape().dimensions();

  // We need to reshape the input to have 1s in the broadcasted dimensions.
  std::vector<int32_t> new_dims;
  for (int i = 0; i < output_dims.size(); ++i) {
    if (!absl::c_linear_search(dimensions, i)) {
      new_dims.push_back(i);
    }
  }

  uint32_t new_dims_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_static_expand_dims(subgraph, new_dims.size(),
                                                    new_dims.data(), in,
                                                    &new_dims_id, /*flags=*/0));

  auto out_ynn_dims = YnnDimensions(instr->shape());
  YNN_RETURN_IF_ERROR(ynn_define_static_broadcast(
      subgraph, out_ynn_dims.size(), out_ynn_dims.data(), new_dims_id, &out,
      /*flags=*/0));

  return out;
}

absl::StatusOr<uint32_t> DefineConcatenateOp(ynn_subgraph_t subgraph,
                                             TensorIdMap& tensor_ids,
                                             const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for concatenate op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kConcatenate);

  std::vector<uint32_t> inputs;
  for (const HloInstruction* operand : instr->operands()) {
    TF_ASSIGN_OR_RETURN(auto in, FindTensorValue(tensor_ids, operand));
    inputs.push_back(in);
  }
  TF_ASSIGN_OR_RETURN(auto out, DefineTensorValue(subgraph, instr));

  YNN_RETURN_IF_ERROR(
      ynn_define_concatenate(subgraph, instr->concatenate_dimension(),
                             inputs.size(), inputs.data(), &out, /*flags=*/0));
  return out;
}

absl::StatusOr<uint32_t> DefineSliceOp(ynn_subgraph_t subgraph,
                                       TensorIdMap& tensor_ids,
                                       const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for slice op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kSlice);
  const HloInstruction* input = instr->operand(0);
  TF_ASSIGN_OR_RETURN(auto in, FindTensorValue(tensor_ids, input));
  TF_ASSIGN_OR_RETURN(auto out, DefineTensorValue(subgraph, instr));

  const std::vector<int64_t>& starts = instr->slice_starts();
  const std::vector<int64_t>& limits = instr->slice_limits();
  const std::vector<int64_t>& strides = instr->slice_strides();

  int rank = input->shape().dimensions().size();
  std::vector<int32_t> axes(rank);
  absl::c_iota(axes, 0);

  YNN_RETURN_IF_ERROR(ynn_define_static_slice(
      subgraph, rank, axes.data(), starts.data(), limits.data(), strides.data(),
      in, &out, /*flags=*/0));
  return out;
}

absl::StatusOr<uint32_t> DefinePadOp(ynn_subgraph_t subgraph,
                                     TensorIdMap& tensor_ids,
                                     const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for pad op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kPad);
  const HloInstruction* input = instr->operand(0);
  const HloInstruction* padding_value = instr->operand(1);
  TF_ASSIGN_OR_RETURN(auto in, FindTensorValue(tensor_ids, input));
  TF_ASSIGN_OR_RETURN(auto pad_val, FindTensorValue(tensor_ids, padding_value));
  TF_ASSIGN_OR_RETURN(auto out, DefineTensorValue(subgraph, instr));

  const PaddingConfig& config = instr->padding_config();
  int rank = input->shape().dimensions().size();
  std::vector<int32_t> axes(rank);
  absl::c_iota(axes, 0);

  std::vector<int64_t> pad_low(rank);
  std::vector<int64_t> pad_high(rank);
  for (int i = 0; i < rank; ++i) {
    pad_low[i] = config.dimensions(i).edge_padding_low();
    pad_high[i] = config.dimensions(i).edge_padding_high();
  }

  YNN_RETURN_IF_ERROR(ynn_define_static_pad(subgraph, rank, axes.data(),
                                            pad_low.data(), pad_high.data(), in,
                                            pad_val, &out, /*flags=*/0));
  return out;
}

absl::StatusOr<uint32_t> DefineIotaOp(ynn_subgraph_t subgraph,
                                      const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for iota op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kIota);
  const HloIotaInstruction* iota = Cast<HloIotaInstruction>(instr);

  TF_ASSIGN_OR_RETURN(uint32_t out_id, DefineTensorValue(subgraph, instr));

  const Shape& shape = instr->shape();
  int64_t rank = shape.dimensions().size();
  int64_t iota_dim = iota->iota_dimension();

  PrimitiveType element_type = shape.element_type();
  TF_ASSIGN_OR_RETURN(auto ynn_element_type, YnnType(element_type));

  auto stride_shape = ShapeUtil::MakeShape(element_type, {rank});
  TF_ASSIGN_OR_RETURN(auto stride_value, Literal::Make(stride_shape));

  for (int64_t i = 0; i < rank; ++i) {
    int value = (i == iota_dim) ? 1 : 0;
    if (primitive_util::IsIntegralType(element_type)) {
      TF_RETURN_IF_ERROR(stride_value.SetIntegralAsS64({i}, value));
    } else {
      TF_RETURN_IF_ERROR(stride_value.SetFromDouble({i}, value));
    }
  }

  uint32_t stride_id = YNN_INVALID_VALUE_ID;
  const size_t stride_dims[] = {static_cast<size_t>(rank)};
  YNN_RETURN_IF_ERROR(ynn_define_tensor(
      subgraph, ynn_element_type, 1, stride_dims, stride_value.untyped_data(),
      YNN_VALUE_FLAG_COPY_DATA, &stride_id));

  auto out_ynn_dims = YnnDimensions(shape);
  YNN_RETURN_IF_ERROR(ynn_define_iota(subgraph, ynn_element_type, rank,
                                      out_ynn_dims.data(), YNN_INVALID_VALUE_ID,
                                      stride_id, &out_id,
                                      /*flags=*/0));
  return out_id;
}

absl::StatusOr<uint32_t> DefineUnaryOp(ynn_subgraph_t subgraph,
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

absl::StatusOr<uint32_t> DefineBinaryOp(ynn_subgraph_t subgraph,
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

absl::StatusOr<uint32_t> DefineReduceOp(ynn_subgraph_t subgraph,
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

  CHECK_EQ(reduce_instr->to_apply()->num_parameters(), 2);
  CHECK_EQ(reduce_instr->to_apply()->instruction_count(), 3);

  TF_ASSIGN_OR_RETURN(
      auto ynn_reduce_op,
      YnnReduceOperator(
          reduce_instr->to_apply()->root_instruction()->opcode()));

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

absl::StatusOr<uint32_t> DefineDotOp(
    ynn_subgraph_t subgraph, TensorIdMap& tensor_ids,
    const HloInstruction* instr, uint32_t output_id = YNN_INVALID_VALUE_ID) {
  VLOG(3) << absl::StreamFormat("Define tensor value for dot op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kDot);
  const HloInstruction* lhs = instr->operand(0);
  const HloInstruction* rhs = instr->operand(1);

  TF_ASSIGN_OR_RETURN(auto lhs_id, FindTensorValue(tensor_ids, lhs));
  TF_ASSIGN_OR_RETURN(auto rhs_id, FindTensorValue(tensor_ids, rhs));
  TF_ASSIGN_OR_RETURN(output_id, DefineTensorValue(subgraph, instr, output_id));

  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();
  const Shape& out_shape = instr->shape();

  DotDimensionNumbers dot_dimensions = instr->dot_dimension_numbers();
  TF_ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                      rhs_shape, out_shape));

  TF_ASSIGN_OR_RETURN(DotCanonicalDims dot_canonical_dims,
                      GetDotCanonicalDims(dot_dimensions, dot_shape));

  const size_t b_rank = rhs_shape.dimensions().size();
  const bool transpose_b = !dot_canonical_dims.rhs_canonical;

  if (transpose_b) {
    uint32_t rhs_id_transposed = YNN_INVALID_VALUE_ID;
    std::array<int32_t, YNN_MAX_TENSOR_RANK> perm;
    absl::c_iota(perm, 0);
    CHECK_LT(b_rank, YNN_MAX_TENSOR_RANK);
    CHECK_GE(b_rank, 2);
    std::swap(perm[b_rank - 1], perm[b_rank - 2]);
    ynn_status status = ynn_define_static_transpose(
        subgraph,
        /*num_dims=*/b_rank, perm.data(), rhs_id, &rhs_id_transposed,
        /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
    rhs_id = rhs_id_transposed;
  }

  YNN_RETURN_IF_ERROR(ynn_define_dot(subgraph, /*num_k_dims=*/1, lhs_id, rhs_id,
                                     YNN_INVALID_VALUE_ID, &output_id,
                                     /*flags=*/0));
  return output_id;
}

absl::StatusOr<uint32_t> DefineReduceWindowOp(ynn_subgraph_t subgraph,
                                              TensorIdMap& tensor_ids,
                                              const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for reduce window op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kReduceWindow);

  const HloInstruction* input = instr->operand(0);
  const HloInstruction* init = instr->operand(1);

  TF_ASSIGN_OR_RETURN(auto input_id, FindTensorValue(tensor_ids, input));
  TF_ASSIGN_OR_RETURN(auto init_id, FindTensorValue(tensor_ids, init));
  TF_ASSIGN_OR_RETURN(auto output_id, DefineTensorValue(subgraph, instr));

  HloOpcode to_apply_opcode = instr->to_apply()->root_instruction()->opcode();
  TF_ASSIGN_OR_RETURN(auto ynn_reduce_op, YnnReduceOperator(to_apply_opcode));

  const Window& window = instr->window();
  int rank = window.dimensions().size();

  std::vector<int32_t> pad_axes;
  std::vector<int64_t> pad_pre;
  std::vector<int64_t> pad_post;

  std::vector<int32_t> stencil_axes;
  std::vector<int32_t> new_axes;
  std::vector<size_t> stencil_dims;
  std::vector<size_t> stencil_strides;
  std::vector<size_t> stencil_dilations;
  std::vector<int32_t> reduce_axes;

  // Track the number of new dimensions.
  int new_axis_count = 0;

  for (int i = 0; i < rank; ++i) {
    const auto& dim = window.dimensions(i);
    pad_axes.push_back(i);
    pad_pre.push_back(dim.padding_low());
    pad_post.push_back(dim.padding_high());

    if (dim.size() > 1 || dim.stride() > 1) {
      stencil_axes.push_back(i);
      // The new dimension is inserted after the current dimension, accounting
      // for previously added dimensions.
      int32_t new_axis_idx = i + new_axis_count + 1;
      new_axes.push_back(new_axis_idx);
      stencil_dims.push_back(dim.size());
      stencil_strides.push_back(dim.stride());
      stencil_dilations.push_back(dim.window_dilation());

      reduce_axes.push_back(new_axis_idx);
      new_axis_count++;
    }
  }

  uint32_t current_input_id = input_id;
  auto is_nonzero = [](int64_t pad) { return pad != 0; };
  if (absl::c_any_of(pad_pre, is_nonzero) ||
      absl::c_any_of(pad_post, is_nonzero)) {
    // The padding should be the identity value of the reduction.
    PrimitiveType input_type = input->shape().element_type();

    TF_ASSIGN_OR_RETURN(double identity_float, ReduceIdentity(to_apply_opcode));

    TF_ASSIGN_OR_RETURN(
        auto identity_literal,
        LiteralUtil::CreateR0<double>(identity_float).Convert(input_type));

    TF_ASSIGN_OR_RETURN(ynn_type ynn_type, YnnType(input_type));

    uint32_t identity_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(
        ynn_define_tensor(subgraph, ynn_type, /*rank=*/0, /*dims=*/nullptr,
                          /*data=*/identity_literal.untyped_data(),
                          /*flags=*/YNN_VALUE_FLAG_COPY_DATA, &identity_id));

    uint32_t padded_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(ynn_define_static_pad(
        subgraph, pad_axes.size(), pad_axes.data(), pad_pre.data(),
        pad_post.data(), current_input_id, identity_id, &padded_id,
        /*flags=*/0));
    current_input_id = padded_id;
  }

  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  if (!stencil_axes.empty()) {
    YNN_RETURN_IF_ERROR(ynn_define_stencil_copy(
        subgraph, stencil_axes.size(), stencil_axes.data(), new_axes.data(),
        stencil_dims.data(), stencil_strides.data(), stencil_dilations.data(),
        current_input_id, YNN_INVALID_VALUE_ID, &stencil_id, /*flags=*/0));
  } else {
    stencil_id = current_input_id;
  }

  YNN_RETURN_IF_ERROR(ynn_define_reduce(
      subgraph, ynn_reduce_op, reduce_axes.size(), reduce_axes.data(),
      stencil_id, init_id, &output_id, /*flags=*/0));

  return output_id;
}

absl::StatusOr<uint32_t> DefineConvolutionOp(
    ynn_subgraph_t subgraph, TensorIdMap& tensor_ids,
    const HloInstruction* instr, uint32_t output_id = YNN_INVALID_VALUE_ID) {
  VLOG(3) << absl::StreamFormat("Define tensor value for convolution op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kConvolution);
  const HloConvolutionInstruction* conv =
      Cast<HloConvolutionInstruction>(instr);

  const HloInstruction* lhs = conv->operand(0);
  const HloInstruction* rhs = conv->operand(1);

  TF_ASSIGN_OR_RETURN(auto lhs_id, FindTensorValue(tensor_ids, lhs));
  TF_ASSIGN_OR_RETURN(auto rhs_id, FindTensorValue(tensor_ids, rhs));
  TF_ASSIGN_OR_RETURN(output_id, DefineTensorValue(subgraph, instr, output_id));

  TF_ASSIGN_OR_RETURN(ynn_type ynn_lhs_type,
                      YnnType(lhs->shape().element_type()));
  TF_ASSIGN_OR_RETURN(ynn_type ynn_out_type,
                      YnnType(conv->shape().element_type()));

  Window conv_window = conv->window();
  ConvolutionDimensionNumbers conv_dims = conv->convolution_dimension_numbers();

  int64_t input_channels =
      conv->operand(0)->shape().dimensions(conv_dims.input_feature_dimension());
  int64_t kernel_output_channels = conv->operand(1)->shape().dimensions(
      conv_dims.kernel_output_feature_dimension());

  const int size = conv_window.dimensions().size();
  std::vector<int32_t> stencil_axes(size);
  std::vector<size_t> stencil_dims(size);
  std::vector<size_t> stencil_strides(size);
  std::vector<size_t> stencil_dilations(size);
  std::vector<int64_t> padding_lows(size);
  std::vector<int64_t> padding_highs(size);
  for (int i = 0; i < size; ++i) {
    stencil_axes[i] = conv_dims.input_spatial_dimensions(i);
    stencil_dims[i] = conv_window.dimensions(i).size();
    stencil_strides[i] = conv_window.dimensions(i).stride();
    stencil_dilations[i] = conv_window.dimensions(i).window_dilation();
    padding_lows[i] = conv_window.dimensions(i).padding_low();
    padding_highs[i] = conv_window.dimensions(i).padding_high();
  }

  auto dims = [](absl::Span<const int64_t> dims) -> std::vector<size_t> {
    return {dims.begin(), dims.end()};
  };

  std::vector<size_t> rhs_dims = dims(rhs->shape().dimensions());
  std::vector<size_t> out_dims = dims(conv->shape().dimensions());

  const size_t feature_group_count = conv->feature_group_count();

  ynn_status status;

  // We will need to create an intermediate buffer for the output if it's
  // grouped convolution.
  uint32_t output_unfused_id =
      feature_group_count != 1 ? YNN_INVALID_VALUE_ID : output_id;

  if (feature_group_count != 1) {
    uint32_t split_id = YNN_INVALID_VALUE_ID;
    CHECK_EQ(rhs_dims.size(), 4);
    // [kh, kw, ci/g, co] -> [kh, kw, ci/g, g, co/g].
    size_t filter_split[] = {feature_group_count,
                             kernel_output_channels / feature_group_count};
    status = ynn_define_split_dim(subgraph, /*axis=*/-1, /*num_splits=*/2,
                                  filter_split, rhs_id, &split_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
    rhs_id = split_id;

    uint32_t transposed_rhs_id = YNN_INVALID_VALUE_ID;
    // [kh, kw, ci/g, g, co/g] -> [g, kh, kw, ci/g, co/g]
    int32_t swap_co_ci[5] = {3, 0, 1, 2, 4};
    status =
        ynn_define_static_transpose(subgraph, /*rank=*/5, swap_co_ci, rhs_id,
                                    &transposed_rhs_id, /*flags=*/0);

    if (status != ynn_status_success) {
      return status;
    }
    rhs_id = transposed_rhs_id;

    // Create intermediate output buffer.
    std::vector<size_t> unfused_dims(out_dims.begin(), out_dims.end() - 1);
    unfused_dims.push_back(feature_group_count);
    unfused_dims.push_back(1);
    unfused_dims.push_back(kernel_output_channels / feature_group_count);
    status = ynn_define_tensor(subgraph, ynn_out_type,
                               /*rank=*/out_dims.size() + 2,
                               /*dims=*/unfused_dims.data(),
                               /*data=*/nullptr,
                               /*flags=*/0, &output_unfused_id);
    if (status != ynn_status_success) {
      return status;
    }
  }

  // If any of paddings is not zero, define a padding value and pad the input.
  if (absl::c_any_of(padding_lows, [](int32_t i) { return i != 0; }) ||
      absl::c_any_of(padding_highs, [](int32_t i) { return i != 0; })) {
    uint32_t padding_id = YNN_INVALID_VALUE_ID;

    // Define padding value.
    uint64_t padding_value = 0;
    status = ynn_define_tensor(subgraph, ynn_lhs_type,
                               /*rank=*/0, /*dims=*/nullptr,
                               /*data=*/&padding_value,
                               /*flags=*/YNN_VALUE_FLAG_COPY_DATA, &padding_id);

    if (status != ynn_status_success) {
      return status;
    }

    uint32_t padded_id = YNN_INVALID_VALUE_ID;
    status = ynn_define_static_pad(
        subgraph, stencil_axes.size(), stencil_axes.data(), padding_lows.data(),
        padding_highs.data(), lhs_id, padding_id, &padded_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
    lhs_id = padded_id;
    padding_id = YNN_INVALID_VALUE_ID;
  }

  size_t num_k_dims = stencil_dims.size() + 1;

  std::vector<int32_t> new_axes;
  if (feature_group_count != 1) {
    // (n, h, w, c) -> (n, h, w, [g, 1,] kh, kw, c / g)
    stencil_dims.push_back(feature_group_count);
    stencil_dims.push_back(1);
    stencil_axes.push_back(3);
    stencil_axes.push_back(3);
    // We need to insert stencil dimensions [kh, kw] right before the channel
    // dimension and [g, 1] before stencil dimensions.
    new_axes = {-3, -2, -5, -4};
    stencil_strides.push_back(1);
    stencil_strides.push_back(1);
    stencil_dilations.push_back(input_channels / feature_group_count);
    stencil_dilations.push_back(1);
  } else {
    // We need to insert stencil dimensions [kh, kw] right before the channel
    // dimension.
    new_axes = {-3, -2};
  }

  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  // Make a stenciled view of the input [n, h, w, ci] -> [n, h, w, kh, kw, ci].
  status = ynn_define_stencil_copy(
      subgraph, /*num_stencils=*/stencil_dims.size(), stencil_axes.data(),
      new_axes.data(), stencil_dims.data(), stencil_strides.data(),
      stencil_dilations.data(), lhs_id, YNN_INVALID_VALUE_ID, &stencil_id,
      /*flags=*/0);
  if (status != ynn_status_success) {
    return status;
  }

  status = ynn_define_dot(subgraph, num_k_dims, stencil_id, rhs_id,
                          YNN_INVALID_VALUE_ID, &output_unfused_id,
                          /*flags=*/0);

  if (status != ynn_status_success) {
    return status;
  }

  if (feature_group_count > 1) {
    // The output of the grouped convolution is [n, h, w, g, 1, co/g], so we
    // need to fuse three of the innermost dimensions.
    status = ynn_define_fuse_dim(subgraph, /*axis=*/-3, /*axes_count=*/3,
                                 output_unfused_id, &output_id,
                                 /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
  }

  return output_id;
}

//===----------------------------------------------------------------------===//
// Emit YNNPACK subgraph for the given HLO computation.
//===----------------------------------------------------------------------===//

absl::StatusOr<YnnSubgraph> EmitYnnSubgraph(
    const HloComputation* computation, Literals& literals,
    absl::Span<const se::DeviceAddressBase> arguments_buffers,
    absl::Span<const int64_t> captured_parameters) {
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
        const void* data = nullptr;
        if (absl::c_linear_search(captured_parameters,
                                  instr->parameter_number())) {
          data = arguments_buffers[instr->parameter_number()].opaque();
        }
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineParameter(subgraph.get(), instr, data));
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

      case HloOpcode::kReshape: {
        if (!IsReshapeOpSupportedByYnn(instr)) {
          return InvalidArgument(
              "Unsupported reshape instruction in YNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineReshapeOp(subgraph.get(), tensor_ids, instr));
      } break;

      case HloOpcode::kTranspose: {
        if (!IsTransposeOpSupportedByYnn(instr)) {
          return InvalidArgument(
              "Unsupported transpose instruction in YNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(
            tensor_ids[instr],
            DefineTransposeOp(subgraph.get(), tensor_ids, instr));
      } break;

      case HloOpcode::kBroadcast: {
        if (!IsBroadcastOpSupportedByYnn(instr)) {
          return InvalidArgument(
              "Unsupported broadcast instruction in YNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(
            tensor_ids[instr],
            DefineBroadcastOp(subgraph.get(), tensor_ids, instr));
      } break;

      case HloOpcode::kConcatenate: {
        if (!IsConcatenateOpSupportedByYnn(instr)) {
          return InvalidArgument(
              "Unsupported concatenate instruction in YNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(
            tensor_ids[instr],
            DefineConcatenateOp(subgraph.get(), tensor_ids, instr));
      } break;

      case HloOpcode::kSlice: {
        if (!IsSliceOpSupportedByYnn(instr)) {
          return InvalidArgument(
              "Unsupported slice instruction in YNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineSliceOp(subgraph.get(), tensor_ids, instr));
      } break;

      case HloOpcode::kPad: {
        if (!IsPadOpSupportedByYnn(instr)) {
          return InvalidArgument(
              "Unsupported pad instruction in YNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefinePadOp(subgraph.get(), tensor_ids, instr));
      } break;

      case HloOpcode::kIota: {
        if (!IsIotaSupportedByYnn(instr)) {
          return InvalidArgument(
              "Unsupported iota instruction in YNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineIotaOp(subgraph.get(), instr));
      } break;

      case HloOpcode::kDot: {
        if (!IsDotSupportedByYnn(instr).value_or(false)) {
          return InvalidArgument(
              "Unsupported dot instruction in YNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineDotOp(subgraph.get(), tensor_ids, instr));
      } break;

      case HloOpcode::kReduce: {
        TF_ASSIGN_OR_RETURN(tensor_ids[instr],
                            DefineReduceOp(subgraph.get(), tensor_ids, instr));
      } break;

      case HloOpcode::kReduceWindow: {
        if (!IsReduceLikeOpSupportedByYnn(instr)) {
          return InvalidArgument(
              "Unsupported reduce window instruction in YNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(
            tensor_ids[instr],
            DefineReduceWindowOp(subgraph.get(), tensor_ids, instr));
      } break;

      case HloOpcode::kConvolution: {
        if (!IsConvolutionOpSupportedByYnn(instr)) {
          return InvalidArgument(
              "Unsupported convolution instruction in YNN fusion: %s",
              instr->ToString());
        }
        TF_ASSIGN_OR_RETURN(
            tensor_ids[instr],
            DefineConvolutionOp(subgraph.get(), tensor_ids, instr));
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

}  // namespace

absl::StatusOr<absl::AnyInvocable<absl::StatusOr<YnnSubgraph>(
    absl::Span<const se::DeviceAddressBase> arguments_buffers)>>
EmitYnnFusionBuilder(const HloComputation* computation,
                     std::vector<int64_t> captured_parameters) {
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

  return
      [computation, literals = Literals(),
       captured_ids = std::move(captured_parameters)](
          absl::Span<const se::DeviceAddressBase> arguments_buffers) mutable {
        return EmitYnnSubgraph(computation, literals, arguments_buffers,
                               captured_ids);
      };
}

}  // namespace xla::cpu
