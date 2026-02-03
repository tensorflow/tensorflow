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
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
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

namespace {

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

}  // anonymous namespace

static absl::StatusOr<uint32_t> DefineConstant(ynn_subgraph_t subgraph,
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

static absl::StatusOr<uint32_t> DefineDotOp(ynn_subgraph_t subgraph,
                                            TensorIdMap& tensor_ids,
                                            const HloInstruction* instr) {
  VLOG(3) << absl::StreamFormat("Define tensor value for dot op: %s",
                                instr->ToString());
  CHECK_EQ(instr->opcode(), HloOpcode::kDot);
  const HloInstruction* lhs = instr->operand(0);
  const HloInstruction* rhs = instr->operand(1);
  CHECK_EQ(lhs->shape().element_type(), instr->shape().element_type());
  CHECK_EQ(rhs->shape().element_type(), instr->shape().element_type());

  TF_ASSIGN_OR_RETURN(auto lhs_id, FindTensorValue(tensor_ids, lhs));
  TF_ASSIGN_OR_RETURN(auto rhs_id, FindTensorValue(tensor_ids, rhs));
  TF_ASSIGN_OR_RETURN(auto output_id, DefineTensorValue(subgraph, instr));

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

//===----------------------------------------------------------------------===//
// Emit YNNPACK subgraph for the given HLO computation.
//===----------------------------------------------------------------------===//

static absl::StatusOr<YnnSubgraph> EmitYnnSubgraph(
    const HloComputation* computation, Literals& literals) {
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
    absl::c_iota(perm, 0);
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

static ynn_status DefineConvolution(
    ynn_subgraph_t subgraph, ynn_type input1_id_type, ynn_type output_id_type,
    uint32_t input1_id, uint32_t input2_id, uint32_t output_id,
    const std::vector<size_t>& filter_dims, const std::vector<size_t>& out_dims,
    size_t feature_group_count, size_t input_channels,
    size_t kernel_output_channels, std::vector<int32_t> stencil_axes,
    std::vector<size_t> stencil_dims, std::vector<size_t> stencil_strides,
    std::vector<size_t> stencil_dilations,
    const std::vector<int64_t>& padding_lows,
    const std::vector<int64_t>& padding_highs) {
  size_t num_k_dims = stencil_dims.size() + 1;
  ynn_status status;

  // We will need to create an intermediate buffer for the output if it's
  // grouped convolution.
  uint32_t output_unfused_id =
      feature_group_count != 1 ? YNN_INVALID_VALUE_ID : output_id;

  if (feature_group_count != 1) {
    uint32_t split_id = YNN_INVALID_VALUE_ID;
    CHECK_EQ(filter_dims.size(), 4);
    // [kh, kw, ci/g, co] -> [kh, kw, ci/g, g, co/g].
    size_t filter_split[] = {feature_group_count,
                             kernel_output_channels / feature_group_count};
    status =
        ynn_define_split_dim(subgraph, /*axis=*/-1, /*num_splits=*/2,
                             filter_split, input2_id, &split_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
    input2_id = split_id;

    uint32_t transposed_filter_id = YNN_INVALID_VALUE_ID;
    // [kh, kw, ci/g, g, co/g] -> [g, kh, kw, ci/g, co/g]
    int32_t swap_co_ci[5] = {3, 0, 1, 2, 4};
    status =
        ynn_define_static_transpose(subgraph, /*rank=*/5, swap_co_ci, input2_id,
                                    &transposed_filter_id, /*flags=*/0);

    if (status != ynn_status_success) {
      return status;
    }
    input2_id = transposed_filter_id;

    // Create intermediate output buffer.
    std::vector<size_t> unfused_dims(out_dims.begin(), out_dims.end() - 1);
    unfused_dims.push_back(feature_group_count);
    unfused_dims.push_back(1);
    unfused_dims.push_back(kernel_output_channels / feature_group_count);
    status = ynn_define_tensor_value(subgraph, output_id_type,
                                     /*rank=*/out_dims.size() + 2,
                                     /*dims=*/unfused_dims.data(),
                                     /*data=*/nullptr,
                                     /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                                     /*scale_id=*/YNN_INVALID_VALUE_ID,
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
    status = ynn_define_tensor_value(subgraph, input1_id_type,
                                     /*rank=*/0, /*dims=*/nullptr,
                                     /*data=*/&padding_value,
                                     /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                                     /*scale_id=*/YNN_INVALID_VALUE_ID,
                                     /*flags=*/YNN_VALUE_FLAG_COPY_DATA,
                                     &padding_id);

    if (status != ynn_status_success) {
      return status;
    }

    uint32_t padded_id = YNN_INVALID_VALUE_ID;
    status = ynn_define_static_pad(
        subgraph, stencil_axes.size(), stencil_axes.data(), padding_lows.data(),
        padding_highs.data(), input1_id, padding_id, &padded_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return status;
    }
    input1_id = padded_id;
    padding_id = YNN_INVALID_VALUE_ID;
  }

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
      stencil_dilations.data(), input1_id, YNN_INVALID_VALUE_ID, &stencil_id,
      /*flags=*/0);
  if (status != ynn_status_success) {
    return status;
  }

  status = ynn_define_dot(subgraph, num_k_dims, stencil_id, input2_id,
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

  return status;
}

static absl::StatusOr<YnnSubgraph> EmitYnnDotSubgraph(
    const HloDotInstruction* dot, Literals& literals,
    absl::Span<const se::DeviceAddressBase> arguments_buffers,
    bool capture_rhs) {
  // TODO(b/468895209): Use the fusion emitter above instead of replicating the
  // logic here.
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

static absl::StatusOr<YnnSubgraph> EmitYnnConvolutionSubgraph(
    const HloConvolutionInstruction* conv, Literals& literals,
    absl::Span<const se::DeviceAddressBase> arguments_buffers) {
  TF_ASSIGN_OR_RETURN(
      YnnSubgraph subgraph, CreateYnnSubgraph([&](ynn_subgraph_t* subgraph) {
        return ynn_create_subgraph(
            /*external_value_ids=*/3,
            YnnFlags(conv->GetModule()->config().debug_options()), subgraph);
      }));

  uint32_t lhs_id = 0;
  uint32_t rhs_id = 1;
  uint32_t out_id = 2;

  const HloInstruction* lhs = conv->operand(0);
  const HloInstruction* rhs = conv->operand(1);

  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();
  const Shape& out_shape = conv->shape();

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
      /*data=*/nullptr,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*scale_id=*/YNN_INVALID_VALUE_ID, input_tensor_flags, &rhs_id));

  const uint32_t output_tensor_flags = YNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  YNN_RETURN_IF_ERROR(ynn_define_tensor_value(
      subgraph.get(), ynn_out_type, out_dims.size(), out_dims.data(),
      /*data=*/nullptr,
      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
      /*scale_id=*/YNN_INVALID_VALUE_ID, output_tensor_flags, &out_id));

  Window conv_window = conv->window();
  int conv_window_dims_size = conv_window.dimensions_size();

  ConvolutionDimensionNumbers conv_dimensions =
      conv->convolution_dimension_numbers();

  std::vector<int32_t> stencil_axes(conv_window_dims_size);
  std::vector<size_t> stencil_dims(conv_window_dims_size);
  std::vector<size_t> stencil_strides(conv_window_dims_size);
  std::vector<size_t> stencil_dilations(conv_window_dims_size);
  std::vector<int64_t> padding_lows(conv_window_dims_size);
  std::vector<int64_t> padding_highs(conv_window_dims_size);

  for (size_t i = 0; i < conv_window.dimensions_size(); ++i) {
    stencil_axes[i] = conv_dimensions.input_spatial_dimensions(i);
    stencil_dims[i] = conv_window.dimensions(i).size();
    stencil_strides[i] = conv_window.dimensions(i).stride();
    stencil_dilations[i] = conv_window.dimensions(i).window_dilation();
    padding_lows[i] = conv_window.dimensions(i).padding_low();
    padding_highs[i] = conv_window.dimensions(i).padding_high();
  }

  YNN_RETURN_IF_ERROR(DefineConvolution(
      subgraph.get(), ynn_lhs_type, ynn_out_type, lhs_id, rhs_id, out_id,
      rhs_dims, out_dims, conv->feature_group_count(),
      conv->operand(0)->shape().dimensions(
          conv_dimensions.input_feature_dimension()),
      conv->operand(1)->shape().dimensions(
          conv_dimensions.kernel_output_feature_dimension()),
      std::move(stencil_axes), std::move(stencil_dims),
      std::move(stencil_strides), std::move(stencil_dilations), padding_lows,
      padding_highs));

  ynn_status status = ynn_optimize_subgraph(
      subgraph.get(), /*threadpool=*/nullptr, /*flags=*/0);
  TF_RETURN_IF_ERROR(YnnStatusToStatus(status));

  return subgraph;
}

absl::StatusOr<absl::AnyInvocable<absl::StatusOr<YnnSubgraph>(
    absl::Span<const se::DeviceAddressBase> arguments_buffers)>>
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

  return
      [computation, literals = Literals()](
          absl::Span<const se::DeviceAddressBase> arguments_buffers) mutable {
        return EmitYnnSubgraph(computation, literals);
      };
}

absl::StatusOr<absl::AnyInvocable<absl::StatusOr<YnnSubgraph>(
    absl::Span<const se::DeviceAddressBase> arguments_buffers)>>
EmitYnnDotBuilder(const HloDotInstruction* dot, bool capture_rhs) {
  return
      [dot, capture_rhs, literals = Literals()](
          absl::Span<const se::DeviceAddressBase> arguments_buffers) mutable {
        return EmitYnnDotSubgraph(dot, literals, arguments_buffers,
                                  capture_rhs);
      };
}

absl::StatusOr<absl::AnyInvocable<absl::StatusOr<YnnSubgraph>(
    absl::Span<const se::DeviceAddressBase> arguments_buffers)>>
EmitYnnConvolutionBuilder(const HloConvolutionInstruction* conv) {
  return
      [conv, literals = Literals()](
          absl::Span<const se::DeviceAddressBase> arguments_buffers) mutable {
        return EmitYnnConvolutionSubgraph(conv, literals, arguments_buffers);
      };
}

}  // namespace xla::cpu
