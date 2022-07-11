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

#include "tensorflow/compiler/xla/service/hlo_verifier.h"

#include <memory>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {

namespace {

bool IsCallerInstruction(HloInstruction* hlo) {
  switch (hlo->opcode()) {
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kWhile:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSort:
    case HloOpcode::kFusion:
    case HloOpcode::kCustomCall:
      return true;
    default:
      return false;
  }
}

Status CheckOperandCount(const HloInstruction* hlo, int expected) {
  if (hlo->operand_count() != expected) {
    return InternalError("Expected %d operands for %s instruction: %s",
                         expected, HloOpcodeString(hlo->opcode()),
                         hlo->ToString());
  }
  return OkStatus();
}

Status CheckParameterCount(const HloInstruction* calling_instruction,
                           const HloComputation* computation, int expected) {
  if (computation->num_parameters() != expected) {
    return InternalError(
        "Expected computation %s called from %s to have %d parameters, has %d",
        computation->name(), calling_instruction->name(), expected,
        computation->num_parameters());
  }
  return OkStatus();
}

int64_t GetSubgroupSize(HloCollectiveInstruction* hlo,
                        CollectiveOpGroupMode group_mode) {
  const HloModuleConfig& config = hlo->GetModule()->config();
  // empty replica groups imply all replicas form a single group.
  int64_t replica_subgroup_size =
      hlo->replica_groups().empty()
          ? 0
          : hlo->replica_groups()[0].replica_ids_size();
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica:
    case CollectiveOpGroupMode::kCrossReplicaAndPartition: {
      int64_t replica_subgroup_size =
          hlo->replica_groups().empty()
              ? config.replica_count()
              : hlo->replica_groups()[0].replica_ids_size();
      if (group_mode == CollectiveOpGroupMode::kCrossReplicaAndPartition) {
        // Replicas from all partitions participate.
        replica_subgroup_size *= config.num_partitions();
      }
      return replica_subgroup_size;
    }
    case CollectiveOpGroupMode::kFlattenedID:
      return replica_subgroup_size;
    case CollectiveOpGroupMode::kCrossPartition:
      return hlo->replica_groups().empty()
                 ? config.num_partitions()
                 : hlo->replica_groups()[0].replica_ids_size();
  }
}

Status CheckNestedComputationThreadNameEqual(const HloComputation* comp,
                                             bool skip_nested_async_op_check) {
  std::optional<absl::string_view> thread_name = comp->thread_name();
  for (const HloInstruction* instr : comp->instructions()) {
    if (skip_nested_async_op_check && instr->IsAsynchronous()) {
      continue;
    }
    for (const HloComputation* cmp : instr->called_computations()) {
      if (cmp->thread_name() != thread_name) {
        return InternalError(
            "Nested computations expects same computation's thread name (%s vs "
            "%s).",
            thread_name ? absl::StrCat(*thread_name) : "none",
            cmp->thread_name() ? absl::StrCat(*cmp->thread_name()) : "none");
      }
      TF_RETURN_IF_ERROR(CheckNestedComputationThreadNameEqual(
          cmp, skip_nested_async_op_check));
    }
  }
  return Status::OK();
}
}  // namespace

Status ShapeVerifier::Preprocess(HloInstruction* hlo) {
  if (!hlo->called_computations().empty() && !IsCallerInstruction(hlo)) {
    return InternalError(
        "Called computations specified for non-caller instruction  %s",
        hlo->ToString());
  }
  std::optional<int> arity = HloOpcodeArity(hlo->opcode());
  if (arity) {
    TF_RETURN_IF_ERROR(CheckOperandCount(hlo, *arity));
  }
  return OkStatus();
}

Status ShapeVerifier::HandleElementwiseUnary(HloInstruction* hlo) {
  return CheckUnaryShape(hlo);
}

Status ShapeVerifier::HandleElementwiseBinary(HloInstruction* hlo) {
  return CheckBinaryShape(hlo);
}

Status ShapeVerifier::HandleClamp(HloInstruction* clamp) {
  return CheckTernaryShape(clamp);
}

Status ShapeVerifier::HandleSelect(HloInstruction* select) {
  return CheckTernaryShape(select);
}

Status ShapeVerifier::HandleConcatenate(HloInstruction* concatenate) {
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : concatenate->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(concatenate,
                    ShapeInference::InferConcatOpShape(
                        operand_shapes, concatenate->concatenate_dimension()));
}

Status ShapeVerifier::HandleConvert(HloInstruction* convert) {
  return CheckShape(convert, ShapeInference::InferConvertShape(
                                 convert->operand(0)->shape(),
                                 convert->shape().element_type()));
}

Status ShapeVerifier::HandleBitcastConvert(HloInstruction* convert) {
  return CheckShape(convert, ShapeInference::InferBitcastConvertShape(
                                 convert->operand(0)->shape(),
                                 convert->shape().element_type()));
}

Status ShapeVerifier::HandleCopy(HloInstruction* copy) {
  return CheckUnaryShape(copy);
}

Status ShapeVerifier::HandleDot(HloInstruction* dot) {
  TF_ASSIGN_OR_RETURN(
      const Shape expected,
      ShapeInference::InferDotOpShape(
          dot->operand(0)->shape(), dot->operand(1)->shape(),
          dot->dot_dimension_numbers(),
          /*preferred_element_type=*/dot->shape().element_type()));
  return CheckShape(dot, expected);
}

Status ShapeVerifier::HandleConvolution(HloInstruction* convolution) {
  TF_ASSIGN_OR_RETURN(
      Shape expected,
      ShapeInference::InferConvolveShape(
          convolution->operand(0)->shape(), convolution->operand(1)->shape(),
          convolution->feature_group_count(), convolution->batch_group_count(),
          convolution->window(), convolution->convolution_dimension_numbers(),
          /*preferred_element_type=*/convolution->shape().element_type()));
  return CheckShape(convolution, expected);
}

Status ShapeVerifier::HandleFft(HloInstruction* fft) {
  TF_ASSIGN_OR_RETURN(
      const Shape expected,
      ShapeInference::InferFftShape(fft->operand(0)->shape(), fft->fft_type(),
                                    fft->fft_length()));
  return CheckShape(fft, expected);
}

Status ShapeVerifier::HandleTriangularSolve(HloInstruction* hlo) {
  TF_ASSIGN_OR_RETURN(const Shape expected,
                      ShapeInference::InferTriangularSolveShape(
                          hlo->operand(0)->shape(), hlo->operand(1)->shape(),
                          hlo->triangular_solve_options()));
  return CheckShape(hlo, expected);
}

Status ShapeVerifier::HandleCholesky(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(CheckOperandCount(hlo, 1));
  TF_ASSIGN_OR_RETURN(const Shape expected, ShapeInference::InferCholeskyShape(
                                                hlo->operand(0)->shape()));
  return CheckShape(hlo, expected);
}

Status ShapeVerifier::HandleOptimizationBarrier(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(CheckOperandCount(hlo, 1));
  return CheckShape(hlo, hlo->operand(0)->shape());
}

// Checks that `hlo`'s set of ReplicaGroups:
//
//  - names each replica 0 through n-1 exactly once (where n is either number of
//    replicas, or number of partitions, or their product)
//  - does not contain any empty ReplicaGroups.
//
// Note that although none of the groups may be empty, `hlo` is allowed to have
// empty groups when group mode is not kFlattenedID. That just means it has one
// big group.
//
// In general, if replica groups is not empty, all replica groups should be of
// the same size. The exception is all-reduce, where non-uniform replica groups
// are allowed. This is controlled by `uniform_replica_group_size`.
static Status CheckReplicaGroups(HloInstruction* hlo,
                                 CollectiveOpGroupMode group_mode,
                                 bool uniform_replica_group_size = true) {
  if (!hlo->replica_groups().empty()) {
    absl::flat_hash_set<int64_t> replicas_seen;
    for (const ReplicaGroup& g : hlo->replica_groups()) {
      if (g.replica_ids().empty()) {
        return InternalError(
            "Instruction cannot have an empty replica group: %s",
            hlo->ToString());
      }
      for (int64_t i : g.replica_ids()) {
        if (!replicas_seen.insert(i).second) {
          return InternalError(
              "Replica %d is repeated in instruction's replica-groups: %s", i,
              hlo->ToString());
        }
      }
    }
    size_t n = replicas_seen.size();
    for (int64_t i = 0; i < n; ++i) {
      if (!replicas_seen.count(i)) {
        return InternalError(
            "Replica %d is not named in instruction's replica-groups: %s", i,
            hlo->ToString());
      }
    }

    // replica-groups have numbers [0, n). This n should be either replica or
    // partition count, or their product. In some cases, replica and/or
    // partition count is not set in the HloModule config and has a default
    // value of 1. For those cases, skip this part of the verification.
    int64_t replica_count = hlo->GetModule()->config().replica_count();
    int64_t num_partitions = hlo->GetModule()->config().num_partitions();
    switch (group_mode) {
      case CollectiveOpGroupMode::kCrossReplica:
      case CollectiveOpGroupMode::kCrossReplicaAndPartition: {
        TF_RET_CHECK(replica_count == 1 || n == replica_count)
            << "In " << CollectiveOpGroupModeToString(group_mode)
            << " mode, replica groups should contain " << replica_count
            << " replicas, but found " << n << ": " << hlo->ToString();
        break;
      }
      case CollectiveOpGroupMode::kCrossPartition: {
        TF_RET_CHECK(num_partitions == 1 || n == num_partitions)
            << "In " << CollectiveOpGroupModeToString(group_mode)
            << " mode, replica groups should contain " << num_partitions
            << " partitions, but found " << n << ": " << hlo->ToString();
        break;
      }
      case CollectiveOpGroupMode::kFlattenedID: {
        const int64_t num_flattened_ids = replica_count * num_partitions;
        TF_RET_CHECK(num_flattened_ids == 1 || n == num_flattened_ids)
            << "In " << CollectiveOpGroupModeToString(group_mode)
            << " mode, replica groups should contain " << num_flattened_ids
            << " flattened IDs, but found " << n << ": " << hlo->ToString();
        break;
      }
    }

    if (uniform_replica_group_size) {
      int64_t size = hlo->replica_groups()[0].replica_ids_size();
      for (const ReplicaGroup& g : hlo->replica_groups()) {
        TF_RET_CHECK(size == g.replica_ids_size())
            << "Replica groups expected to be of uniform size";
      }
    }
  } else {
    TF_RET_CHECK(group_mode != CollectiveOpGroupMode::kFlattenedID)
        << "Replica groups must be specified in flattened-id mode";
  }

  return OkStatus();
}

static Status CheckCommonAllGatherInvariants(HloInstruction* hlo,
                                             int64_t* computed_shard_count) {
  auto ag = Cast<HloAllGatherInstruction>(hlo);
  CHECK_NE(computed_shard_count, nullptr) << "Expected a shard count as input";
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                      GetCollectiveOpGroupMode(ag->channel_id().has_value(),
                                               ag->use_global_device_ids()));
  TF_RETURN_IF_ERROR(CheckReplicaGroups(ag, group_mode));
  TF_RET_CHECK(ag->all_gather_dimension() >= 0);

  int64_t shard_count;
  for (int64_t i = 0; i < ag->operand_count(); ++i) {
    TF_RET_CHECK(ag->all_gather_dimension() < ag->operand(i)->shape().rank());

    Shape output_shape;
    if (hlo->opcode() == HloOpcode::kAllGather) {
      output_shape = (ag->operand_count() == 1) ? ag->shape()
                                                : ag->shape().tuple_shapes(i);
    } else {
      TF_RET_CHECK(hlo->opcode() == HloOpcode::kAllGatherStart);
      output_shape = (ag->operand_count() == 1)
                         ? ag->shape().tuple_shapes(1)
                         : ag->shape().tuple_shapes(1).tuple_shapes(i);
    }
    TF_RET_CHECK(ag->all_gather_dimension() < output_shape.rank());
    if (i == 0) {
      shard_count = CeilOfRatio(
          output_shape.dimensions(ag->all_gather_dimension()),
          ag->operand(i)->shape().dimensions(ag->all_gather_dimension()));
    }
  }

  int64_t subgroup_size = GetSubgroupSize(ag, group_mode);
  // If replica and partition count is not explicitly set, it will have a
  // default value of 1, in which case the subgroup_size will be 1 as well. Skip
  // these verification checks in that case.
  TF_RET_CHECK(subgroup_size == 1 || shard_count == subgroup_size)
      << "shard_count = " << shard_count
      << ", subgroup_size = " << subgroup_size << ", " << hlo->ToString();
  *computed_shard_count = shard_count;
  return OkStatus();
}

Status ShapeVerifier::HandleAllGather(HloInstruction* hlo) {
  auto ag = Cast<HloAllGatherInstruction>(hlo);
  int64_t shard_count;
  TF_RETURN_IF_ERROR(CheckCommonAllGatherInvariants(hlo, &shard_count));
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(
      ag, ShapeInference::InferAllGatherShape(
              operand_shapes, ag->all_gather_dimension(), shard_count));
}

Status ShapeVerifier::HandleAllGatherStart(HloInstruction* hlo) {
  auto ag = Cast<HloAllGatherInstruction>(hlo);
  int64_t shard_count;
  TF_RETURN_IF_ERROR(CheckCommonAllGatherInvariants(hlo, &shard_count));
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(
      ag, ShapeInference::InferAllGatherStartShape(
              operand_shapes, ag->all_gather_dimension(), shard_count));
}

Status ShapeVerifier::HandleAllGatherDone(HloInstruction* hlo) {
  return CheckShape(
      hlo, ShapeInference::InferAllGatherDoneShape(hlo->operand(0)->shape()));
}

Status ShapeVerifier::HandleAllReduce(HloInstruction* hlo) {
  auto ar = Cast<HloAllReduceInstruction>(hlo);
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                      GetCollectiveOpGroupMode(ar->channel_id().has_value(),
                                               ar->use_global_device_ids()));
  TF_RETURN_IF_ERROR(
      CheckReplicaGroups(ar, group_mode, /*uniform_replica_group_size=*/false));

  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(hlo, ShapeInference::InferAllReduceShape(operand_shapes));
}

Status ShapeVerifier::HandleReduceScatter(HloInstruction* hlo) {
  auto ars = Cast<HloReduceScatterInstruction>(hlo);
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                      GetCollectiveOpGroupMode(ars->channel_id().has_value(),
                                               ars->use_global_device_ids()));
  TF_RETURN_IF_ERROR(CheckReplicaGroups(ars, group_mode));
  TF_RET_CHECK(ars->scatter_dimension() >= 0);

  for (int64_t i = 0; i < ars->operand_count(); ++i) {
    TF_RET_CHECK(ars->scatter_dimension() < ars->operand(i)->shape().rank());

    const Shape& output_shape = (ars->operand_count() == 1)
                                    ? ars->shape()
                                    : ars->shape().tuple_shapes(i);
    TF_RET_CHECK(ars->scatter_dimension() < output_shape.rank());
  }

  const Shape& output0_shape =
      (ars->operand_count() == 1) ? ars->shape() : ars->shape().tuple_shapes(0);
  int64_t shard_count =
      CeilOfRatio(ars->operand(0)->shape().dimensions(ars->scatter_dimension()),
                  output0_shape.dimensions(ars->scatter_dimension()));
  int64_t subgroup_size = GetSubgroupSize(ars, group_mode);
  // If replica and partition count is not explicitly set, it will have a
  // default value of 1, in which case the subgroup_size will be 1 as well. Skip
  // these verification checks in that case.
  TF_RET_CHECK(subgroup_size == 1 || shard_count == subgroup_size)
      << "shard_count = " << shard_count
      << ", subgroup_size = " << subgroup_size << ", " << hlo->ToString();

  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(ars,
                    ShapeInference::InferReduceScatterShape(
                        operand_shapes, ars->scatter_dimension(), shard_count));
}

Status ShapeVerifier::HandleAllReduceStart(HloInstruction* hlo) {
  auto ar = Cast<HloAllReduceInstruction>(hlo);
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                      GetCollectiveOpGroupMode(ar->channel_id().has_value(),
                                               ar->use_global_device_ids()));
  TF_RETURN_IF_ERROR(
      CheckReplicaGroups(ar, group_mode, /*uniform_replica_group_size=*/false));

  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(hlo,
                    ShapeInference::InferAllReduceStartShape(operand_shapes));
}

Status ShapeVerifier::HandleAllReduceDone(HloInstruction* hlo) {
  return CheckShape(
      hlo, ShapeInference::InferAllReduceDoneShape(hlo->operand(0)->shape()));
}

Status ShapeVerifier::HandleAllToAll(HloInstruction* hlo) {
  auto* all_to_all = Cast<HloAllToAllInstruction>(hlo);
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                      GetCollectiveOpGroupMode(
                          all_to_all->channel_id().has_value(), std::nullopt));

  TF_RETURN_IF_ERROR(CheckReplicaGroups(hlo, group_mode));

  TF_RET_CHECK(all_to_all != nullptr);

  if (all_to_all->split_dimension()) {
    int64_t split_count = GetSubgroupSize(all_to_all, group_mode);
    TF_RET_CHECK(hlo->operand_count() == 1);
    return CheckShape(
        hlo, ShapeInference::InferAllToAllShape(
                 hlo->operand(0)->shape(), *all_to_all->split_dimension(),
                 *all_to_all->split_dimension(), split_count));
  } else {
    std::vector<const Shape*> operand_shapes;
    for (const HloInstruction* operand : hlo->operands()) {
      operand_shapes.push_back(&operand->shape());
    }
    return CheckShape(hlo,
                      ShapeInference::InferAllToAllTupleShape(operand_shapes));
  }
}

Status ShapeVerifier::HandlePartitionId(HloInstruction* hlo) {
  return CheckShape(hlo, ShapeUtil::MakeShape(U32, {}));
}

Status ShapeVerifier::HandleReplicaId(HloInstruction* hlo) {
  return CheckShape(hlo, ShapeUtil::MakeShape(U32, {}));
}

namespace {

Status CheckBufferOffset(const Shape& buffer_shape,
                         const Shape& buffer_offset_shape) {
  if (!buffer_offset_shape.IsTuple()) {
    return InternalError("Buffer offset is not tuple.");
  }
  bool all_is_array =
      absl::c_all_of(buffer_offset_shape.tuple_shapes(),
                     [](const Shape& shape) { return shape.IsArray(); });
  bool all_is_tuple =
      absl::c_all_of(buffer_offset_shape.tuple_shapes(),
                     [](const Shape& shape) { return shape.IsTuple(); });
  if (!all_is_array && !all_is_tuple) {
    return InternalError(
        "Buffer offset should either be a tuple of arrays or "
        " a tuple of tuples.");
  }

  if (all_is_tuple) {
    if (absl::c_any_of(buffer_offset_shape.tuple_shapes(),
                       [&buffer_shape](const Shape& shape) {
                         return ShapeUtil::TupleElementCount(shape) !=
                                buffer_shape.rank();
                       })) {
      return InternalError(
          "Buffer offset index should have the same number of "
          "elements as the buffer's rank.");
    }
  } else {
    if (buffer_offset_shape.tuple_shapes_size() != buffer_shape.rank()) {
      return InternalError(
          "Buffer offset index should have the same number of "
          "elements as the buffer's rank.");
    }
  }
  return OkStatus();
}

Status CheckInplaceCollectivePermute(HloInstruction* collective_permute) {
  if (collective_permute->operand_count() == 1) {
    return OkStatus();
  }
  if (collective_permute->operand_count() != 4) {
    return InternalError("Unexpected number of operands: %d.",
                         collective_permute->operand_count());
  }

  const Shape& input_buffer_shape = collective_permute->operand(0)->shape();
  const Shape& output_buffer_shape = collective_permute->operand(1)->shape();
  const Shape& input_offset_shape = collective_permute->operand(2)->shape();
  const Shape& output_offset_shape = collective_permute->operand(3)->shape();

  if (input_buffer_shape.IsArray() && output_buffer_shape.IsArray()) {
    Status check_input_buffer_offset =
        CheckBufferOffset(input_buffer_shape, input_offset_shape);
    if (!check_input_buffer_offset.ok()) {
      return check_input_buffer_offset;
    }
    Status check_output_buffer_offset =
        CheckBufferOffset(output_buffer_shape, output_offset_shape);
    if (!check_output_buffer_offset.ok()) {
      return check_output_buffer_offset;
    }
  } else if (input_buffer_shape.IsTuple() && output_buffer_shape.IsTuple()) {
    if (ShapeUtil::TupleElementCount(input_buffer_shape) !=
        ShapeUtil::TupleElementCount(output_buffer_shape)) {
      return InternalError("Unmatching input buffers and output buffers.");
    }
    if (!input_offset_shape.IsTuple() ||
        ShapeUtil::TupleElementCount(input_offset_shape) !=
            ShapeUtil::TupleElementCount(input_buffer_shape)) {
      return InternalError("Unmatching input buffers and input offset.");
    }
    for (int i = 0; i < input_buffer_shape.tuple_shapes_size(); ++i) {
      Status check_input_buffer_offset =
          CheckBufferOffset(input_buffer_shape.tuple_shapes(i),
                            input_offset_shape.tuple_shapes(i));
      if (!check_input_buffer_offset.ok()) {
        return check_input_buffer_offset;
      }
    }
    if (!output_offset_shape.IsTuple() ||
        ShapeUtil::TupleElementCount(output_offset_shape) !=
            ShapeUtil::TupleElementCount(output_buffer_shape)) {
      return InternalError("Unmatching output buffers and output offset.");
    }
    for (int i = 0; i < output_buffer_shape.tuple_shapes_size(); ++i) {
      Status check_output_buffer_offset =
          CheckBufferOffset(output_buffer_shape.tuple_shapes(i),
                            output_offset_shape.tuple_shapes(i));
      if (!check_output_buffer_offset.ok()) {
        return check_output_buffer_offset;
      }
    }
  } else {
    return InternalError("Unmatching input buffers and output buffers.");
  }
  return OkStatus();
}

Status CheckDuplicatedSourceOrTarget(HloInstruction* hlo,
                                     CollectiveOpGroupMode group_mode) {
  // A source or target cannot appear twice in the collective-permute's
  // source-target pairs. Also, based on the group formation mode, check if the
  // source and target IDs are within expected range.

  // Note: for collective-permute, only kCrossReplica and kCrossPartition modes
  // are valid.
  const HloModuleConfig& config = hlo->GetModule()->config();
  const int64_t limit = group_mode == CollectiveOpGroupMode::kCrossReplica
                            ? config.replica_count()
                            : config.num_partitions();
  absl::flat_hash_map<int64_t, std::vector<int64_t>> seen_source_to_targets;
  absl::flat_hash_map<int64_t, std::vector<int64_t>> seen_target_to_sources;
  int allowed_seen_count = 1;
  if (hlo->operand_count() == 4) {
    if (hlo->operand(0)->shape().IsArray()) {
      allowed_seen_count = hlo->operand(2)->shape().tuple_shapes_size();
    } else {
      allowed_seen_count =
          hlo->operand(2)->shape().tuple_shapes(0).tuple_shapes_size();
    }
  }

  for (const auto& p : hlo->source_target_pairs()) {
    TF_RET_CHECK(p.first >= 0)
        << "Source " << p.first
        << " in the instruction's source-target pair must be >= 0 : "
        << hlo->ToString();
    TF_RET_CHECK(limit == 1 || p.first < limit)
        << "Source " << p.first
        << " in the instruction's source-target pair must be < " << limit
        << " : " << hlo->ToString();
    if (seen_source_to_targets.contains(p.first) &&
        seen_source_to_targets[p.first].size() == allowed_seen_count) {
      if (allowed_seen_count == 1) {
        return InternalError(
            "Source %d appears more than once in instruction's source-target "
            "pairs: %s",
            p.first, hlo->ToString());
      } else {
        return InternalError(
            "Source %d appears more than %d times in instruction's "
            "source-target "
            "pairs: %s",
            p.first, allowed_seen_count, hlo->ToString());
      }
    } else {
      seen_source_to_targets[p.first].push_back(p.second);
    }
    TF_RET_CHECK(p.second >= 0)
        << "Target " << p.second
        << " in the instruction's source-target pair must be >= 0 : "
        << hlo->ToString();
    TF_RET_CHECK(limit == 1 || p.second < limit)
        << "Target " << p.second
        << " in the instruction's source-target pair must be < " << limit
        << " : " << hlo->ToString();
    if (seen_target_to_sources.contains(p.second) &&
        seen_target_to_sources[p.second].size() == allowed_seen_count) {
      if (allowed_seen_count == 1) {
        return InternalError(
            "Target %d appears more than once in instruction's source-target "
            "pairs: %s",
            p.second, hlo->ToString());
      } else {
        return InternalError(
            "Target %d appears more than %d times in instruction's "
            "source-target "
            "pairs: %s",
            p.second, allowed_seen_count, hlo->ToString());
      }
    } else {
      seen_target_to_sources[p.second].push_back(p.first);
    }
  }
  return OkStatus();
}

}  // namespace

Status ShapeVerifier::HandleCollectivePermute(HloInstruction* hlo) {
  TF_ASSIGN_OR_RETURN(
      CollectiveOpGroupMode group_mode,
      GetCollectiveOpGroupMode(hlo->channel_id().has_value(),
                               /*use_global_device_ids=*/std::nullopt));
  TF_RETURN_IF_ERROR(CheckInplaceCollectivePermute(hlo));
  TF_RETURN_IF_ERROR(CheckDuplicatedSourceOrTarget(hlo, group_mode));
  std::vector<const Shape*> operand_shapes;
  absl::c_transform(
      hlo->operands(), std::back_inserter(operand_shapes),
      [](const HloInstruction* operand) { return &(operand->shape()); });
  return CheckShape(
      hlo, ShapeInference::InferCollectivePermuteShape(operand_shapes));
}

Status ShapeVerifier::HandleCollectivePermuteStart(HloInstruction* hlo) {
  TF_ASSIGN_OR_RETURN(
      CollectiveOpGroupMode group_mode,
      GetCollectiveOpGroupMode(hlo->channel_id().has_value(),
                               /*use_global_device_ids=*/std::nullopt));
  TF_RETURN_IF_ERROR(CheckInplaceCollectivePermute(hlo));
  TF_RETURN_IF_ERROR(CheckDuplicatedSourceOrTarget(hlo, group_mode));
  std::vector<const Shape*> operand_shapes;
  absl::c_transform(
      hlo->operands(), std::back_inserter(operand_shapes),
      [](const HloInstruction* operand) { return &(operand->shape()); });
  return CheckShape(
      hlo, ShapeInference::InferCollectivePermuteStartShape(operand_shapes));
}

Status ShapeVerifier::HandleCollectivePermuteDone(HloInstruction* hlo) {
  return CheckShape(hlo, ShapeInference::InferCollectivePermuteDoneShape(
                             hlo->operand(0)->shape()));
}

Status ShapeVerifier::HandleReducePrecision(HloInstruction* reduce_precision) {
  return CheckShape(reduce_precision, ShapeInference::InferReducePrecisionShape(
                                          reduce_precision->operand(0)->shape(),
                                          reduce_precision->exponent_bits(),
                                          reduce_precision->mantissa_bits()));
}

Status ShapeVerifier::CheckIsTokenOperand(const HloInstruction* instruction,
                                          int64_t operand_no) {
  const HloInstruction* token = instruction->operand(operand_no);
  if (!ShapeUtil::Equal(token->shape(), ShapeUtil::MakeTokenShape())) {
    return InternalError(
        "Expected operand %d to be token-shaped, actual shape is "
        "%s:\n%s",
        operand_no, StringifyShape(token->shape()), instruction->ToString());
  }
  return OkStatus();
}

Status ShapeVerifier::CheckOperandAndParameter(
    const HloInstruction* instruction, int64_t operand_number,
    const HloComputation* computation, int64_t parameter_number) {
  const HloInstruction* operand = instruction->operand(operand_number);
  const HloInstruction* parameter =
      computation->parameter_instruction(parameter_number);
  if (!ShapesSame(operand->shape(), parameter->shape())) {
    return InternalError("Operand %s shape does not match parameter's %s in %s",
                         operand->ToString(), parameter->ToString(),
                         instruction->ToString());
  }
  return OkStatus();
}

Status ShapeVerifier::HandleInfeed(HloInstruction* instruction) {
  HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(instruction);
  TF_RETURN_IF_ERROR(CheckIsTokenOperand(instruction, 0));

  // The output of infeed is a tuple containing the data value and a token.
  return CheckShape(infeed,
                    ShapeUtil::MakeTupleShape(
                        {infeed->infeed_shape(), ShapeUtil::MakeTokenShape()}));
}

Status ShapeVerifier::HandleOutfeed(HloInstruction* instruction) {
  HloOutfeedInstruction* outfeed = Cast<HloOutfeedInstruction>(instruction);
  TF_RETURN_IF_ERROR(CheckIsTokenOperand(instruction, 1));

  // Outfeed has a separate shape field for the value which is outfed to the
  // host. The shape of the instruction itself is always a token.
  if (!ShapesSame(outfeed->outfeed_shape(), outfeed->operand(0)->shape())) {
    return InternalError(
        "Expected outfeed shape to be equal to operand's shape %s, "
        "actual shape is %s:\n%s",
        StringifyShape(outfeed->operand(0)->shape()),
        StringifyShape(outfeed->outfeed_shape()), outfeed->ToString());
  }
  return CheckShape(outfeed, ShapeUtil::MakeTokenShape());
}

bool ShapeVerifier::HasCompatibleElementTypes(const Shape& shape_0,
                                              const Shape& shape_1,
                                              const Shape& result_shape) {
  return ShapeUtil::SameElementType(shape_0, shape_1) &&
         (ShapeUtil::SameElementType(shape_0, result_shape) ||
          (opts_.allow_mixed_precision &&
           ShapeUtil::SameElementTypeIgnoringFpPrecision(shape_0,
                                                         result_shape)));
}

Status ShapeVerifier::HandleRng(HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(CheckOperandCount(instruction, 2));

  const Shape& shape_0 = instruction->operand(0)->shape();
  const Shape& shape_1 = instruction->operand(1)->shape();
  if (!ShapeUtil::IsScalar(shape_0) || !ShapeUtil::IsScalar(shape_1)) {
    return InternalError(
        "Expected scalar types for the two operands of Rng instruction: %s",
        instruction->ToString());
  }

  if (!HasCompatibleElementTypes(shape_0, shape_1, instruction->shape())) {
    return InternalError(
        "Expected compatible element types for the result and the two operands"
        " of Rng instruction: %s",
        instruction->ToString());
  }

  PrimitiveType element_type = shape_0.element_type();
  switch (instruction->random_distribution()) {
    case RNG_UNIFORM:
      if (!primitive_util::IsFloatingPointType(element_type) &&
          !primitive_util::IsIntegralType(element_type) &&
          element_type != PRED) {
        return InternalError(
            "Element type not supported."
            " Expected element to be of floating point type, integral type or"
            " predicate type for RngUniform: %s",
            instruction->ToString());
      }
      break;

    case RNG_NORMAL:
      if (!primitive_util::IsFloatingPointType(element_type)) {
        return InternalError(
            "Element type not supported."
            " Expected element to be FloatingPointType for RngNormal: %s",
            instruction->ToString());
      }
      break;
    default:
      return InternalError(
          "Invalid Rng distribution %s",
          RandomDistribution_Name(instruction->random_distribution()));
  }

  return OkStatus();
}

Status ShapeVerifier::HandleRngBitGenerator(HloInstruction* hlo) {
  if (!hlo->shape().IsTuple()) {
    return OkStatus();
  }
  if (hlo->shape().IsTuple() && hlo->shape().tuple_shapes_size() != 2) {
    return InternalError(
        "Expected tuple shape with 2 elements for RngBitGenerator. Got: %s",
        hlo->shape().ToString());
  }
  if (!ShapeUtil::Compatible(hlo->operand(0)->shape(),
                             hlo->shape().tuple_shapes(0))) {
    return InternalError(
        "Expected state shape to match between input and output for "
        "RngBitGenerator. Got %s vs. %s",
        hlo->operand(0)->shape().ToString(),
        hlo->shape().tuple_shapes(0).ToString());
  }
  return OkStatus();
}

Status ShapeVerifier::HandleRngGetAndUpdateState(HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(CheckOperandCount(instruction, 0));
  const Shape& result_shape = instruction->shape();
  const Shape expected_shape = ShapeUtil::MakeShape(U64, {2});
  if (!ShapeUtil::Compatible(result_shape, expected_shape)) {
    return InternalError(
        "Invalid RngGetAndUpdateState, expect result to have shape %s, got %s ",
        StringifyShape(expected_shape), StringifyShape(result_shape));
  }

  return OkStatus();
}

Status ShapeVerifier::HandleReverse(HloInstruction* reverse) {
  return CheckShape(
      reverse, ShapeInference::InferReverseShape(reverse->operand(0)->shape(),
                                                 reverse->dimensions()));
}

Status ShapeVerifier::HandleSort(HloInstruction* sort) {
  if (sort->operand_count() < 1) {
    return InternalError("Expected at least 1 operand for %s instruction: %s",
                         HloOpcodeString(sort->opcode()), sort->ToString());
  }
  HloComputation* compare = sort->to_apply();

  // Check that the 'compare' computation returns a PRED.
  Shape compare_shape = compare->root_instruction()->shape();
  if (!ShapeUtil::Compatible(compare_shape, ShapeUtil::MakeShape(PRED, {}))) {
    return InternalError(
        "The Sort compare computation shape does not lead to a scalar "
        "predicate shape: %s",
        StringifyShape(compare_shape));
  }

  // Check that the number of parameters of the 'compare' computation is
  // correct.
  TF_RETURN_IF_ERROR(
      CheckParameterCount(sort, compare, sort->operand_count() * 2));

  // Verify that the operands of the compare computation have the correct scalar
  // shapes.
  for (int64_t parameter_idx = 0; parameter_idx < compare->num_parameters();
       ++parameter_idx) {
    int64_t operand_idx = parameter_idx / 2;
    Shape expected_scalar_shape = ShapeUtil::MakeShape(
        sort->operand(operand_idx)->shape().element_type(), {});
    Shape actual_parameter_shape =
        compare->parameter_instruction(parameter_idx)->shape();
    if (!ShapeUtil::CompatibleIgnoringFpPrecision(expected_scalar_shape,
                                                  actual_parameter_shape)) {
      return InternalError(
          "Expected the %lld-th parameter of the compare computation of sort "
          "to have shape %s, but got %s",
          parameter_idx, StringifyShape(expected_scalar_shape),
          StringifyShape(actual_parameter_shape));
    }
  }

  // Verify that all operand shapes have the same dimensions.
  for (int64_t operand = 1; operand < sort->operand_count(); ++operand) {
    if (!ShapeUtil::SameDimensions(sort->operand(0)->shape(),
                                   sort->operand(operand)->shape())) {
      return InternalError(
          "Expected sort to have to have the same dimensions for all operands. "
          "First operand shape is: %s\n, shape (operand index %lld) is: %s",
          StringifyShape(sort->operand(0)->shape()), operand,
          StringifyShape(sort->operand(operand)->shape()));
    }
  }
  return CheckVariadicShape(sort);
}

Status ShapeVerifier::HandleConstant(HloInstruction* constant) {
  if (!Cast<HloConstantInstruction>(constant)->HasLiteral()) {
    return InternalError("Constant is required to have a valid literal: %s",
                         constant->ToString());
  }
  return CheckShape(constant, constant->literal().shape(),
                    /*only_compare_minor_to_major_in_layout=*/true);
}

Status ShapeVerifier::HandleIota(HloInstruction* hlo) {
  auto* iota = Cast<HloIotaInstruction>(hlo);
  if (!iota->shape().IsArray()) {
    return InternalError("Iota does not support non-array result.");
  }
  const int64_t rank = iota->shape().rank();
  if (rank == 0) {
    return InternalError("Iota does not support scalars.");
  }
  int64_t iota_dimension = iota->iota_dimension();
  if (iota_dimension >= rank || iota_dimension < 0) {
    return InternalError(
        "The iota dimension cannot go beyond the operation rank or be "
        "negative.");
  }

  PrimitiveType primitive_type = iota->shape().element_type();
  if (!primitive_util::IsIntegralType(primitive_type) &&
      !primitive_util::IsFloatingPointType(primitive_type) &&
      !primitive_util::IsComplexType(primitive_type)) {
    return InvalidArgument(
        "Only support iota of integral, floating point or complex primitive "
        "types, got %s",
        PrimitiveType_Name(primitive_type));
  }

  return OkStatus();
}

Status ShapeVerifier::HandleGetTupleElement(HloInstruction* get_tuple_element) {
  return CheckShape(get_tuple_element,
                    ShapeInference::InferGetTupleElementShape(
                        get_tuple_element->operand(0)->shape(),
                        get_tuple_element->tuple_index()));
}

namespace {
Status SameElementTypesForOperandsAndToApplyParameters(
    const HloInstruction& instruction, int64_t num_operands_to_check) {
  const ProgramShape& to_apply = instruction.to_apply()->ComputeProgramShape();
  for (int i = 0; i < num_operands_to_check; ++i) {
    const Shape& parameter_shape = to_apply.parameters(i);
    const Shape& operand_shape = instruction.operands()[i]->shape();
    if (!ShapeUtil::SameElementType(parameter_shape, operand_shape)) {
      return InvalidArgument(
          "Shape mismatch between to_apply computation"
          " parameter and operand %d in %s.",
          i, instruction.ToString().c_str());
    }
  }
  return OkStatus();
}
}  // namespace

Status ShapeVerifier::HandleReduce(HloInstruction* reduce) {
  if (reduce->operand_count() % 2 != 0) {
    return InternalError(
        "Expected an even number of operands for %s instruction: %s",
        HloOpcodeString(reduce->opcode()), reduce->ToString());
  }

  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : reduce->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  TF_RETURN_IF_ERROR(
      CheckShape(reduce, ShapeInference::InferReduceShape(
                             operand_shapes, reduce->dimensions(),
                             reduce->to_apply()->ComputeProgramShape())));

  return opts_.allow_mixed_precision
             ? OkStatus()
             : SameElementTypesForOperandsAndToApplyParameters(
                   *reduce, reduce->operand_count());
}

Status ShapeVerifier::HandleBitcast(HloInstruction* bitcast) {
  if (opts_.layout_sensitive &&
      opts_.shape_size(bitcast->shape()) !=
          opts_.shape_size(bitcast->operand(0)->shape())) {
    return InternalError(
        "Bitcast cannot have different shape sizes of output (%d) and operand "
        "(%d) (%s) (%s)",
        opts_.shape_size(bitcast->shape()),
        opts_.shape_size(bitcast->operand(0)->shape()),
        bitcast->shape().ToString(true),
        bitcast->operand(0)->shape().ToString(true));
  }
  return OkStatus();
}

Status ShapeVerifier::HandleBroadcast(HloInstruction* broadcast) {
  // HLO broadcast has no exact analog at the client level so there is no
  // ShapeInference method. Check the output shape explicitly.
  const Shape& operand_shape = broadcast->operand(0)->shape();
  // Check for mixed precision.
  TF_RET_CHECK(SameElementType(broadcast->shape(), operand_shape));
  TF_RET_CHECK(operand_shape.rank() == broadcast->dimensions().size());
  for (int64_t operand_dimension = 0; operand_dimension < operand_shape.rank();
       ++operand_dimension) {
    int64_t output_dimension = broadcast->dimensions()[operand_dimension];
    TF_RET_CHECK((output_dimension < broadcast->shape().rank()) &&
                 output_dimension >= 0 &&
                 (broadcast->shape().dimensions(output_dimension) ==
                  operand_shape.dimensions(operand_dimension)))
        << broadcast->ToString() << " operand shape " << operand_shape;
  }
  return OkStatus();
}

Status ShapeVerifier::HandleDynamicReshape(HloInstruction* dynamic_reshape) {
  // Check for mixed precision.
  const Shape& operand_shape = dynamic_reshape->operand(0)->shape();
  TF_RET_CHECK(SameElementType(dynamic_reshape->shape(), operand_shape));
  TF_RET_CHECK(ShapeUtil::ElementsIn(dynamic_reshape->shape()) ==
               ShapeUtil::ElementsIn(operand_shape));
  TF_RET_CHECK(dynamic_reshape->shape().rank() + 1 ==
               dynamic_reshape->operand_count());
  for (int64_t i = 1; i < dynamic_reshape->operand_count(); ++i) {
    TF_RET_CHECK(dynamic_reshape->operand(i)->shape().element_type() == S32);
  }
  return OkStatus();
}

Status ShapeVerifier::HandleReshape(HloInstruction* reshape) {
  // Check for mixed precision.
  const Shape& operand_shape = reshape->operand(0)->shape();
  TF_RET_CHECK(SameElementType(reshape->shape(), operand_shape));
  TF_RET_CHECK(ShapeUtil::ElementsIn(reshape->shape()) ==
               ShapeUtil::ElementsIn(operand_shape));
  return OkStatus();
}

Status ShapeVerifier::HandleTranspose(HloInstruction* transpose) {
  return CheckShape(
      transpose, ShapeInference::InferTransposeShape(
                     transpose->operand(0)->shape(), transpose->dimensions()));
}

Status ShapeVerifier::HandleParameter(HloInstruction* hlo) {
  return OkStatus();
}

Status ShapeVerifier::HandleFusion(HloInstruction* fusion) {
  if (fusion->called_computations().size() != 1) {
    return InternalError(
        "Fusion has a non-unary number of called computations (%s)",
        fusion->ToString().c_str());
  }
  const Shape& root_computation_shape =
      fusion->called_computations()[0]->root_instruction()->shape();
  if (!ShapesSame(fusion->shape(), root_computation_shape)) {
    return InternalError(
        "Fused computation shape (%s) is not equal to the fusion shape (%s)",
        root_computation_shape.ToString(true), fusion->shape().ToString(true));
  }

  auto& fused_parameters = fusion->fused_parameters();
  if (fused_parameters.size() != fusion->operand_count()) {
    return InternalError(
        "Fused parameter count (%d) does not match the number of operands (%d)"
        " passed to the fusion instruction in: %s.",
        fused_parameters.size(), fusion->operand_count(),
        fusion->ToString().c_str());
  }
  for (HloInstruction* fused_param : fused_parameters) {
    int64_t param_no = fused_param->parameter_number();
    if (!ShapesSame(fused_param->shape(), fusion->operand(param_no)->shape())) {
      return InternalError(
          "Shape mismatch between parameter number %d and its operand in "
          "%s.",
          param_no, fusion->ToString().c_str());
    }
  }
  return OkStatus();
}

Status ShapeVerifier::HandleCall(HloInstruction* call) {
  TF_RETURN_IF_ERROR(
      CheckParameterCount(call, call->to_apply(), call->operand_count()));
  for (int64_t i = 0; i < call->to_apply()->num_parameters(); ++i) {
    TF_RETURN_IF_ERROR(CheckOperandAndParameter(call, i, call->to_apply(), i));
  }
  // The shape of kCall should match the shape of the computation it calls.
  return CheckShape(call, call->to_apply()->root_instruction()->shape());
}

Status ShapeVerifier::HandleCustomCall(HloInstruction* instruction) {
  const HloCustomCallInstruction* custom_call =
      DynCast<const HloCustomCallInstruction>(instruction);
  TF_RET_CHECK(custom_call != nullptr);
  if (custom_call->layout_constrained()) {
    // If the layout is constrained, verify all the respective shapes have
    // layouts and that the constrained operand shapes match the shapes of the
    // operands.
    TF_RET_CHECK(LayoutUtil::HasLayout(custom_call->shape()));
    TF_RET_CHECK(custom_call->operand_count() ==
                 custom_call->operand_shapes_with_layout().size());
    for (int64_t i = 0; i < custom_call->operand_count(); ++i) {
      const Shape& operand_shape_with_layout =
          custom_call->operand_shapes_with_layout()[i];
      TF_RET_CHECK(ShapeUtil::Compatible(custom_call->operand(i)->shape(),
                                         operand_shape_with_layout))
          << custom_call->operand(i)->shape().ToString() << " operand "
          << operand_shape_with_layout.ToString();
      TF_RET_CHECK(LayoutUtil::HasLayout(operand_shape_with_layout));
    }
  }
  for (const auto& pair : custom_call->output_to_operand_aliasing()) {
    TF_RET_CHECK(pair.second.first < custom_call->operand_count())
        << "Invalid aliasing operand index.";
    TF_RET_CHECK(ShapeUtil::IndexIsValid(
        custom_call->operand(pair.second.first)->shape(), pair.second.second))
        << "Invalid aliasing operand shape index.";
    TF_RET_CHECK(ShapeUtil::IndexIsValid(custom_call->shape(), pair.first))
        << "Invalid aliasing output shape index.";
    const Shape& output_subshape =
        ShapeUtil::GetSubshape(custom_call->shape(), pair.first);
    const Shape& operand_subshape = ShapeUtil::GetSubshape(
        custom_call->operand(pair.second.first)->shape(), pair.second.second);
    if (opts_.layout_sensitive) {
      TF_RET_CHECK(operand_subshape == output_subshape)
          << "Different aliasing shapes: " << operand_subshape.ToString()
          << " vs " << output_subshape.ToString();
    } else {
      TF_RET_CHECK(ShapeUtil::Compatible(output_subshape, operand_subshape))
          << "Different aliasing shapes: " << operand_subshape.ToString()
          << " vs " << output_subshape.ToString();
    }
  }
  return OkStatus();
}

Status ShapeVerifier::HandleSlice(HloInstruction* slice) {
  return CheckShape(slice,
                    ShapeInference::InferSliceShape(
                        slice->operand(0)->shape(), slice->slice_starts(),
                        slice->slice_limits(), slice->slice_strides()));
}

Status ShapeVerifier::HandleDynamicSlice(HloInstruction* dynamic_slice) {
  return CheckShape(
      dynamic_slice,
      ShapeInference::InferDynamicSliceShape(
          dynamic_slice->operand(0)->shape(),
          Cast<HloDynamicSliceInstruction>(dynamic_slice)->index_shapes(),
          dynamic_slice->dynamic_slice_sizes()));
}

Status ShapeVerifier::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  return CheckShape(
      dynamic_update_slice,
      ShapeInference::InferDynamicUpdateSliceShape(
          dynamic_update_slice->operand(0)->shape(),
          dynamic_update_slice->operand(1)->shape(),
          Cast<HloDynamicUpdateSliceInstruction>(dynamic_update_slice)
              ->index_shapes()));
}

Status ShapeVerifier::HandleTuple(HloInstruction* tuple) {
  return CheckVariadicShape(tuple);
}

Status ShapeVerifier::HandleMap(HloInstruction* map) {
  std::vector<const Shape*> operand_shapes;
  int64_t max_operand_rank = 0;
  for (const HloInstruction* operand : map->operands()) {
    operand_shapes.push_back(&operand->shape());
    max_operand_rank = std::max(max_operand_rank, operand->shape().rank());
  }
  // TODO(b/65689298) Remove code below once Map is generalized to accept
  // arbitrary map dimensions.
  std::vector<int64_t> map_dims(max_operand_rank);
  std::iota(map_dims.begin(), map_dims.end(), 0);

  TF_RETURN_IF_ERROR(CheckShape(
      map,
      ShapeInference::InferMapShape(
          operand_shapes, map->to_apply()->ComputeProgramShape(), map_dims)));

  return opts_.allow_mixed_precision
             ? OkStatus()
             : SameElementTypesForOperandsAndToApplyParameters(
                   *map, map->operand_count());
}

Status ShapeVerifier::HandleReduceWindow(HloInstruction* reduce_window) {
  VLOG(2) << "Verify reduce window:" << reduce_window->ToString() << "\n";
  auto reduce_window_instr = Cast<HloReduceWindowInstruction>(reduce_window);
  auto input_shapes = reduce_window_instr->input_shapes();
  VLOG(2) << "reduce window input shape count: " << input_shapes.size() << "\n";
  auto init_shapes = reduce_window_instr->init_value_shapes();
  VLOG(2) << "reduce instruction is :" << reduce_window->ToString() << "\n";
  TF_RETURN_IF_ERROR(CheckShape(
      reduce_window, ShapeInference::InferReduceWindowShape(
                         input_shapes, init_shapes, reduce_window->window(),
                         reduce_window->to_apply()->ComputeProgramShape())));

  return opts_.allow_mixed_precision
             ? OkStatus()
             : SameElementTypesForOperandsAndToApplyParameters(
                   *reduce_window, reduce_window->operand_count());
}

Status ShapeVerifier::HandleSelectAndScatter(HloInstruction* instruction) {
  return CheckShape(
      instruction,
      ShapeInference::InferSelectAndScatterShape(
          instruction->operand(0)->shape(),
          instruction->select()->ComputeProgramShape(), instruction->window(),
          instruction->operand(1)->shape(), instruction->operand(2)->shape(),
          instruction->scatter()->ComputeProgramShape()));
}

Status ShapeVerifier::HandleWhile(HloInstruction* xla_while) {
  TF_RETURN_IF_ERROR(
      CheckParameterCount(xla_while, xla_while->while_body(), 1));
  TF_RETURN_IF_ERROR(
      CheckParameterCount(xla_while, xla_while->while_condition(), 1));
  TF_RETURN_IF_ERROR(
      CheckOperandAndParameter(xla_while, 0, xla_while->while_body(), 0));
  TF_RETURN_IF_ERROR(
      CheckOperandAndParameter(xla_while, 0, xla_while->while_condition(), 0));
  const Shape& conditional_shape =
      xla_while->while_condition()->root_instruction()->shape();
  if (!ShapeUtil::Compatible(conditional_shape,
                             ShapeUtil::MakeShape(PRED, {}))) {
    return InternalError(
        "Conditional computation shape does not lead to a scalar predicate "
        "shape: %s",
        StringifyShape(conditional_shape));
  }
  // The shape of kWhile should match the shape of the body computation it
  // calls.
  return CheckShape(xla_while,
                    xla_while->while_body()->root_instruction()->shape());
}

Status ShapeVerifier::HandleConditional(HloInstruction* conditional) {
  if (!ShapeUtil::IsScalar(conditional->operand(0)->shape())) {
    return InvalidArgument(
        "The first operand of conditional must be a scalar. Got %s",
        conditional->operand(0)->shape().DebugString());
  }
  const int num_branches = conditional->branch_count();
  PrimitiveType operand0_type = conditional->operand(0)->shape().element_type();
  if (operand0_type == PRED) {
    TF_RET_CHECK(num_branches == 2);
  } else {
    if (operand0_type != S32) {
      return InvalidArgument(
          "The first operand of indexed conditional must be a scalar of S32. "
          "Got"
          " type %s.",
          PrimitiveType_Name(operand0_type));
    }
    TF_RET_CHECK(num_branches >= 1);
  }
  TF_RETURN_IF_ERROR(CheckOperandCount(conditional, num_branches + 1));
  for (int j = 0; j < num_branches; ++j) {
    TF_RETURN_IF_ERROR(CheckParameterCount(
        conditional, conditional->branch_computation(j), 1));
    TF_RETURN_IF_ERROR(CheckOperandAndParameter(
        conditional, j + 1, conditional->branch_computation(j), 0));
    TF_RETURN_IF_ERROR(CheckShape(
        conditional,
        conditional->branch_computation(j)->root_instruction()->shape()));
  }
  return OkStatus();
}

Status ShapeVerifier::HandlePad(HloInstruction* pad) {
  return CheckShape(pad, ShapeInference::InferPadShape(pad->operand(0)->shape(),
                                                       pad->operand(1)->shape(),
                                                       pad->padding_config()));
}

namespace {
Status CheckAsyncOpOperand(const HloInstruction* async_op) {
  const HloInstruction* operand = async_op->operand(0);
  if (operand->opcode() != HloOpcode::kAsyncStart &&
      operand->opcode() != HloOpcode::kAsyncUpdate) {
    return InternalError(
        "%s expects operand to be async-update or async-done, found "
        "%s.",
        HloOpcodeString(async_op->opcode()),
        HloOpcodeString(operand->opcode()));
  }
  if (*async_op->async_wrapped_computation() !=
      *operand->async_wrapped_computation()) {
    return InternalError(
        "The %s expects its wrapped async computation to be identical to its "
        "operand's wrapped async computation (%s vs %s), thread name (%s vs "
        "%s).",
        HloOpcodeString(async_op->opcode()),
        async_op->async_wrapped_instruction()->ToString(),
        operand->async_wrapped_instruction()->ToString(),
        async_op->async_wrapped_computation()->thread_name()
            ? absl::StrCat(
                  *async_op->async_wrapped_computation()->thread_name())
            : "none",
        operand->async_wrapped_computation()->thread_name()
            ? absl::StrCat(*operand->async_wrapped_computation()->thread_name())
            : "none");
  }
  if (async_op->async_group_id() != operand->async_group_id()) {
    return InternalError(
        "%s expects its operand to have the same group id (%s vs %s).",
        HloOpcodeString(async_op->opcode()),
        async_op->async_group_id() ? absl::StrCat(*async_op->async_group_id())
                                   : "none",
        operand->async_group_id() ? absl::StrCat(*operand->async_group_id())
                                  : "none");
  }
  return OkStatus();
}

Status CheckAsyncOpComputationShapes(const HloInstruction* async_op,
                                     const Shape& async_shape) {
  if (!async_shape.IsTuple() || async_shape.tuple_shapes_size() < 2) {
    return InternalError(
        "The %s expects the async shape to be a tuple of at least two "
        "elements, found %s.",
        HloOpcodeString(async_op->opcode()), async_shape.ToString());
  }
  ProgramShape computation_shape =
      async_op->async_wrapped_computation()->ComputeProgramShape();
  Shape param_shape = ShapeUtil::MakeTupleShape(computation_shape.parameters());
  if (async_shape.tuple_shapes(0) != param_shape) {
    return InternalError(
        "The %s expects the async shape at index {0} to match async "
        "computation parameter shape (%s vs %s).",
        HloOpcodeString(async_op->opcode()),
        async_shape.tuple_shapes(0).ToString(/*print_layout=*/true),
        param_shape.ToString(/*print_layout=*/true));
  }
  if (async_shape.tuple_shapes(1) != computation_shape.result()) {
    return InternalError(
        "The %s expects the async shape at index {1} to match the async "
        "computation root shape (%s vs %s).",
        HloOpcodeString(async_op->opcode()),
        async_shape.tuple_shapes(1).ToString(/*print_layout=*/true),
        computation_shape.result().ToString(/*print_layout=*/true));
  }
  return Status::OK();
}

Status CheckAsyncOpComputationThreadName(const HloInstruction* async_op) {
  std::optional<absl::string_view> async_thread_name =
      async_op->async_thread_name();
  if (async_thread_name !=
      async_op->async_wrapped_computation()->thread_name()) {
    return InternalError(
        "async-start expects same async thread name as wrapped computation's "
        "thread name (%s vs %s).",
        async_thread_name ? absl::StrCat(*async_thread_name) : "none",
        async_op->async_wrapped_computation()->thread_name()
            ? absl::StrCat(
                  *async_op->async_wrapped_computation()->thread_name())
            : "none");
  }
  return CheckNestedComputationThreadNameEqual(
      async_op->async_wrapped_computation(),
      /*skip_nested_async_op_check=*/false);
}

// TODO(b/229887502): apply CheckCallableInstructionThreadName to all
// CallableInstructions verifier.
Status CheckCallableInstructionThreadName(const HloInstruction* instruction,
                                          bool skip_nested_async_op_check) {
  for (const HloComputation* computation : instruction->called_computations()) {
    if (instruction->parent() != nullptr) {
      if (instruction->parent()->thread_name() != computation->thread_name()) {
        return InternalError(
            "callable instruction %s expects parent computation thread name "
            "same as called computation's thread name (%s vs %s).",
            instruction->ToString(),
            instruction->parent()->thread_name()
                ? absl::StrCat(*instruction->parent()->thread_name())
                : "none",
            computation->thread_name()
                ? absl::StrCat(*computation->thread_name())
                : "none");
      }
    }
    TF_RETURN_IF_ERROR(CheckNestedComputationThreadNameEqual(
        computation, skip_nested_async_op_check));
  }
  return Status::OK();
}
}  // namespace

Status ShapeVerifier::HandleAsyncStart(HloInstruction* async_start) {
  TF_RETURN_IF_ERROR(
      CheckAsyncOpComputationShapes(async_start, async_start->shape()));
  TF_RETURN_IF_ERROR(CheckAsyncOpComputationThreadName(async_start));
  const Shape& param_shape = async_start->shape().tuple_shapes(0);
  for (int i = 0; i < async_start->operand_count(); ++i) {
    if (param_shape.tuple_shapes(i) != async_start->operand(i)->shape()) {
      return InternalError(
          "The %s expects the shape of operand %d to match the async shape at "
          "index {0} (%s vs %s).",
          HloOpcodeString(async_start->opcode()), i,
          async_start->operand(i)->shape().ToString(/*print_layout=*/true),
          param_shape.tuple_shapes(i).ToString(/*print_layout=*/true));
    }
  }
  return Status::OK();
}

Status ShapeVerifier::HandleAsyncUpdate(HloInstruction* async_update) {
  TF_RETURN_IF_ERROR(CheckAsyncOpComputationThreadName(async_update));
  if (async_update->operand(0)->shape() != async_update->shape()) {
    return InternalError(
        "The %s expects the shape of operand and output to match (%s vs %s).",
        HloOpcodeString(async_update->opcode()),
        async_update->operand(0)->shape().ToString(),
        async_update->shape().ToString());
  }
  TF_RETURN_IF_ERROR(
      CheckAsyncOpComputationShapes(async_update, async_update->shape()));
  return CheckAsyncOpOperand(async_update);
}

Status ShapeVerifier::HandleAsyncDone(HloInstruction* async_done) {
  TF_RETURN_IF_ERROR(CheckAsyncOpComputationThreadName(async_done));
  TF_RETURN_IF_ERROR(CheckAsyncOpComputationShapes(
      async_done, async_done->operand(0)->shape()));
  const Shape& root_shape = async_done->operand(0)->shape().tuple_shapes(1);
  if (root_shape != async_done->shape()) {
    return InternalError(
        "The %s expects the shape of output to match the async shape at index "
        "{1} (%s vs %s).",
        HloOpcodeString(async_done->opcode()), async_done->shape().ToString(),
        root_shape.ToString());
  }
  return CheckAsyncOpOperand(async_done);
}

Status ShapeVerifier::HandleCopyStart(HloInstruction* copy_start) {
  return CheckShape(copy_start,
                    ShapeUtil::MakeTupleShape({copy_start->operand(0)->shape(),
                                               copy_start->operand(0)->shape(),
                                               ShapeUtil::MakeShape(U32, {})}),
                    /*only_compare_minor_to_major_in_layout=*/true);
}

Status ShapeVerifier::HandleCopyDone(HloInstruction* copy_done) {
  const Shape& operand_shape = copy_done->operand(0)->shape();
  const Shape& dest_shape = ShapeUtil::GetTupleElementShape(operand_shape, 0);
  const Shape& src_shape = ShapeUtil::GetTupleElementShape(operand_shape, 1);
  if (!ShapesSame(dest_shape, src_shape,
                  /*minor_to_major_only=*/false,
                  /*ignore_memory_space=*/true)) {
    return InternalError(
        "Source and destination buffers in CopyDone arguments need to be the "
        "same shape found %s and %s\n%s",
        StringifyShape(dest_shape), StringifyShape(src_shape),
        copy_done->ToString());
  }
  return CheckShape(copy_done, ShapeUtil::GetTupleElementShape(
                                   copy_done->operand(0)->shape(), 0));
}

Status ShapeVerifier::HandleSend(HloInstruction* send) {
  return CheckShape(send,
                    ShapeUtil::MakeTupleShape({send->operand(0)->shape(),
                                               ShapeUtil::MakeShape(U32, {}),
                                               ShapeUtil::MakeTokenShape()}),
                    /*only_compare_minor_to_major_in_layout=*/true);
}

Status ShapeVerifier::HandleSendDone(HloInstruction* send_done) {
  return CheckShape(send_done, ShapeUtil::MakeTokenShape());
}

Status ShapeVerifier::HandleRecv(HloInstruction* recv) {
  return CheckShape(
      recv,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::GetTupleElementShape(recv->shape(), 0),
           ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeTokenShape()}),
      /*only_compare_minor_to_major_in_layout=*/true);
}

Status ShapeVerifier::HandleRecvDone(HloInstruction* recv_done) {
  return CheckShape(
      recv_done,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::GetTupleElementShape(recv_done->operand(0)->shape(), 0),
           ShapeUtil::MakeTokenShape()}));
}

Status ShapeVerifier::HandleBatchNormTraining(
    HloInstruction* batch_norm_training) {
  return CheckShape(batch_norm_training,
                    ShapeInference::InferBatchNormTrainingShape(
                        batch_norm_training->operand(0)->shape(),
                        batch_norm_training->operand(1)->shape(),
                        batch_norm_training->operand(2)->shape(),
                        batch_norm_training->feature_index()));
}

Status ShapeVerifier::HandleBatchNormInference(
    HloInstruction* batch_norm_inference) {
  return CheckShape(batch_norm_inference,
                    ShapeInference::InferBatchNormInferenceShape(
                        batch_norm_inference->operand(0)->shape(),
                        batch_norm_inference->operand(1)->shape(),
                        batch_norm_inference->operand(2)->shape(),
                        batch_norm_inference->operand(3)->shape(),
                        batch_norm_inference->operand(4)->shape(),
                        batch_norm_inference->feature_index()));
}

Status ShapeVerifier::HandleBatchNormGrad(HloInstruction* batch_norm_grad) {
  return CheckShape(batch_norm_grad, ShapeInference::InferBatchNormGradShape(
                                         batch_norm_grad->operand(0)->shape(),
                                         batch_norm_grad->operand(1)->shape(),
                                         batch_norm_grad->operand(2)->shape(),
                                         batch_norm_grad->operand(3)->shape(),
                                         batch_norm_grad->operand(4)->shape(),
                                         batch_norm_grad->feature_index()));
}

namespace {

// Checks that the instruction does not have mixed precision floating point
// inputs.
Status CheckMixedPrecisionOperands(const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    // Allow-list the following opcodes for mixed-precision check, because
    // they involve data pass through or grouping via tuples, where the
    // precisions of buffers can be different.
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kConstant:
    case HloOpcode::kConvolution:
    case HloOpcode::kDot:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kFusion:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kParameter:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kSort:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
      break;
    default: {
      PrimitiveType fp_type = PRIMITIVE_TYPE_INVALID;
      for (auto operand : instruction->operands()) {
        TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
            operand->shape(),
            [&](const Shape& subshape, const ShapeIndex& index) {
              if (!ShapeUtil::ElementIsFloating(subshape)) {
                return OkStatus();
              }
              if (fp_type == PRIMITIVE_TYPE_INVALID) {
                fp_type = subshape.element_type();
              } else if (fp_type != subshape.element_type()) {
                return InternalError(
                    "Seen floating point types of different precisions in "
                    "%s, but mixed precision is disallowed.",
                    instruction->ToString());
              }
              return OkStatus();
            }));
      }
    }
  }
  return OkStatus();
}

}  // namespace

Status ShapeVerifier::HandleGather(HloInstruction* gather) {
  return CheckShape(
      gather,
      ShapeInference::InferGatherShape(
          gather->operand(0)->shape(), gather->operand(1)->shape(),
          gather->gather_dimension_numbers(), gather->gather_slice_sizes()));
}

Status ShapeVerifier::HandleScatter(HloInstruction* scatter) {
  absl::InlinedVector<const Shape*, 3> arg_shapes;
  arg_shapes.reserve(scatter->operand_count());
  for (const HloInstruction* operand : scatter->operands()) {
    arg_shapes.push_back(&operand->shape());
  }
  return CheckShape(scatter,
                    ShapeInference::InferScatterShape(
                        arg_shapes, scatter->to_apply()->ComputeProgramShape(),
                        scatter->scatter_dimension_numbers()));
}

Status ShapeVerifier::HandleAfterAll(HloInstruction* token) {
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : token->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(token, ShapeUtil::MakeTokenShape());
}

Status ShapeVerifier::HandleAddDependency(HloInstruction* add_dependency) {
  TF_RETURN_IF_ERROR(CheckIsTokenOperand(add_dependency, 1));
  return CheckShape(add_dependency, add_dependency->operand(0)->shape());
}

Status ShapeVerifier::HandleGetDimensionSize(HloInstruction* get_size) {
  return CheckShape(get_size,
                    ShapeInference::InferGetDimensionSizeShape(
                        get_size->operand(0)->shape(), get_size->dimension()));
}

Status ShapeVerifier::HandleSetDimensionSize(HloInstruction* set_size) {
  return CheckShape(set_size,
                    ShapeInference::InferSetDimensionSizeShape(
                        set_size->operand(0)->shape(),
                        set_size->operand(1)->shape(), set_size->dimension()));
}

Status ShapeVerifier::CheckShape(const HloInstruction* instruction,
                                 const Shape& inferred_shape,
                                 bool only_compare_minor_to_major_in_layout) {
  // If allow_mixed_precision_ is false, check if there are operands with
  // different precisions. We need this check because ShapeInference allows
  // mixed precision inputs.
  if (!opts_.allow_mixed_precision) {
    TF_RETURN_IF_ERROR(CheckMixedPrecisionOperands(instruction));
  }

  // Check if the output shape matches the expected shape.
  //
  // We treat BF16 and F32 as compatible types if mixed precision is allowed,
  // but only when the instruction defines the BF16/F32 buffer.
  bool equal = [&] {
    switch (instruction->opcode()) {
      // The opcodes below can't have implicit layout conversions, nor can they
      // implicitly transform f32 -> bf16.  Fundamentally these are either
      // reinterpreting existing data (e.g. kBitcast) or shuffling data around
      // without modifying it (e.g. kGetTupleElement).
      case HloOpcode::kBitcast:
      case HloOpcode::kCall:
      case HloOpcode::kConditional:
      case HloOpcode::kConstant:
      case HloOpcode::kCopyDone:
      case HloOpcode::kCopyStart:
      case HloOpcode::kCustomCall:
      case HloOpcode::kGetTupleElement:
      case HloOpcode::kInfeed:
      case HloOpcode::kOutfeed:
      case HloOpcode::kOptimizationBarrier:
      case HloOpcode::kParameter:
      case HloOpcode::kRecv:
      case HloOpcode::kRecvDone:
      case HloOpcode::kSend:
      case HloOpcode::kSendDone:
      case HloOpcode::kTuple:
      case HloOpcode::kWhile:
        return ShapesSame(instruction->shape(), inferred_shape,
                          only_compare_minor_to_major_in_layout);
      case HloOpcode::kDynamicUpdateSlice:
        // For DynamicUpdateSlice it has an "in-place" update semantics, but
        // inside of fusions memory space propagation doesn't propagate the
        // memory spaces all the way, causing possible mismatches. Relax the
        // constraint in that condition.
        return ShapesSame(instruction->shape(), inferred_shape,
                          only_compare_minor_to_major_in_layout,
                          /*ignore_memory_space=*/
                          instruction->parent()->IsFusionComputation());

      // We allow arbitrary layout and f32->bf16 transformations on all other
      // instructions, although this may be made more strict pending discussion
      // in b/112709536.
      default:
        if (opts_.allow_mixed_precision) {
          return ShapeUtil::CompatibleIgnoringFpPrecision(instruction->shape(),
                                                          inferred_shape);
        } else {
          return ShapeUtil::Compatible(instruction->shape(), inferred_shape);
        }
    }
  }();
  if (!equal) {
    return InternalError(
        "Expected instruction to have shape equal to %s, actual "
        "shape is %s:\n%s",
        StringifyShape(inferred_shape), StringifyShape(instruction->shape()),
        instruction->ToString());
  }
  return OkStatus();
}

Status ShapeVerifier::CheckShape(const HloInstruction* instruction,
                                 const StatusOr<Shape>& inferred_shape_status) {
  if (!inferred_shape_status.ok()) {
    Status s = inferred_shape_status.status();
    tensorflow::errors::AppendToMessage(&s, ", for instruction ",
                                        instruction->ToString());
    return s;
  }
  return CheckShape(instruction, inferred_shape_status.ValueOrDie());
}

Status ShapeVerifier::CheckUnaryShape(const HloInstruction* instruction) {
  return CheckShape(instruction,
                    ShapeInference::InferUnaryOpShape(instruction->opcode(),
                                                      instruction->operand(0)));
}

Status ShapeVerifier::CheckBinaryShape(const HloInstruction* instruction) {
  return CheckShape(
      instruction, ShapeInference::InferBinaryOpShape(instruction->opcode(),
                                                      instruction->operand(0),
                                                      instruction->operand(1)));
}

Status ShapeVerifier::CheckTernaryShape(const HloInstruction* instruction) {
  return CheckShape(instruction,
                    ShapeInference::InferTernaryOpShape(
                        instruction->opcode(), instruction->operand(0),
                        instruction->operand(1), instruction->operand(2)));
}

Status ShapeVerifier::CheckVariadicShape(const HloInstruction* instruction) {
  return CheckShape(instruction,
                    ShapeInference::InferVariadicOpShape(
                        instruction->opcode(), instruction->operands()));
}

Status ShapeVerifier::VerifyEntryComputationLayout(const HloModule& module) {
  const HloComputation* computation = module.entry_computation();
  const auto& layout = module.entry_computation_layout();
  const ShapeLayout& result_layout = layout.result_layout();

  TF_RETURN_IF_ERROR(
      ShapeUtil::ValidateShapeWithOptionalLayout(result_layout.shape()));

  if (!ShapeUtil::Compatible(computation->root_instruction()->shape(),
                             result_layout.shape())) {
    return InternalError(
        "Shape of the root instruction of entry computation (%s) should be "
        "compatible to one specified in module's entry computation layout (%s)",
        ShapeUtil::HumanString(computation->root_instruction()->shape()),
        ShapeUtil::HumanString(result_layout.shape()));
  }

  if (computation->num_parameters() != layout.parameter_count()) {
    return InternalError(
        "Number of parameters in entry computation layout (%d) must be same "
        "as number of parameters of entry computation (%d)",
        layout.parameter_count(), computation->num_parameters());
  }

  for (int i = 0; i < computation->num_parameters(); ++i) {
    const HloInstruction* parameter = computation->parameter_instruction(i);
    TF_RETURN_IF_ERROR(
        ShapeUtil::ValidateShapeWithOptionalLayout(layout.parameter_shape(i)));
    if (!ShapeUtil::Compatible(parameter->shape(), layout.parameter_shape(i))) {
      return InternalError(
          "Shape of the entry computation parameter %d is %s should be "
          "compatible to the one specified in module's entry computation "
          "layout %s",
          i, ShapeUtil::HumanString(parameter->shape()),
          ShapeUtil::HumanString(layout.parameter_shape(i)));
    }
  }

  return OkStatus();
}

std::string ComputationsToString(
    absl::Span<HloComputation* const> computations) {
  return absl::StrJoin(computations, ",",
                       [](std::string* s, const HloComputation* computation) {
                         s->append(computation->name());
                       });
}

// Verifies various invariants about the structure of the HLO:
//
// (1) each instruction has a non-null parent() set to the HloComputation
// which
//     contains it.
//
// (2) each computation has a non-null parent() set to the HloModule which
//     contains it.
//
// (3) the operands of each instruction are in the same computation as the
//     instruction.
Status VerifyHloStructure(HloModule* module) {
  for (const HloComputation* computation : module->computations()) {
    if (computation->parent() == nullptr) {
      return InternalError("Computation %s has a null parent pointer",
                           computation->name());
    }
    if (computation->parent() != module) {
      return InternalError(
          "Computation %s parent() does not point to parent module",
          computation->name());
    }

    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->parent() == nullptr) {
        return InternalError("Instruction %s has a null parent pointer",
                             instruction->name());
      }
      if (instruction->parent() != computation) {
        return InternalError(
            "Instruction %s parent() does not point to parent computation",
            instruction->name());
      }
    }
  }

  // Check that operands are in the same computation separately from verifying
  // parent() correctness so conditions like a null HloInstruction::parent()
  // are identified and reported explicitly above rather than reporting a
  // mismatched operand.
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      for (int i = 0; i < instruction->operand_count(); ++i) {
        const HloInstruction* operand = instruction->operand(i);
        if (operand->parent() != instruction->parent()) {
          return InternalError(
              "Operand %d (%s) of instruction %s is in a different "
              "computation: %s vs %s",
              i, operand->name(), instruction->name(),
              operand->parent() ? operand->parent()->name() : "(null)",
              instruction->parent()->name());
        }
      }
    }
  }
  return OkStatus();
}

namespace {

// Returns true if the given Shape has a TOKEN shape as any subshape.
bool ShapeContainsToken(const Shape& shape) {
  bool contains_token = false;
  ShapeUtil::ForEachSubshape(
      shape, [&contains_token](const Shape& subshape, const ShapeIndex&) {
        if (subshape.IsToken()) {
          contains_token = true;
        }
      });
  return contains_token;
}

// Verifies that all types entering and exiting the entry computation are
// legal.
Status VerifyEntryAndExitShapes(const HloModule& module) {
  // Tokens cannot be passed as entry parameters.
  // TODO(b/80000000): Remove this constraint.
  for (int i = 0; i < module.entry_computation()->num_parameters(); ++i) {
    HloInstruction* param =
        module.entry_computation()->parameter_instruction(i);
    if (ShapeContainsToken(param->shape())) {
      return InternalError(
          "Entry parameter %d is or contains a token shape: %s", i,
          ShapeUtil::HumanString(param->shape()));
    }
  }
  return OkStatus();
}

// Checks if the given two instructions share the same channel id.
Status CheckSameChannel(const HloInstruction* instr1,
                        const HloInstruction* instr2) {
  if (instr1->channel_id() != instr2->channel_id()) {
    return InternalError(
        "Expected to have the same channel id, actual channel ids are: %s "
        "(%d), %s (%d)",
        instr1->ToString(), *instr1->channel_id(), instr2->ToString(),
        *instr2->channel_id());
  }
  return OkStatus();
}

// Checks if the given two instructions have the same is_host_transfer
// attribute value. Intsructions must be send/recv instructions or their
// 'done' variant.
Status CheckSameIsHostTransfer(const HloInstruction* instr1,
                               const HloInstruction* instr2) {
  const HloSendRecvInstruction* send_recv1 =
      DynCast<const HloSendRecvInstruction>(instr1);
  const HloSendRecvInstruction* send_recv2 =
      DynCast<const HloSendRecvInstruction>(instr2);
  TF_RET_CHECK(send_recv1 != nullptr);
  TF_RET_CHECK(send_recv2 != nullptr);
  if (send_recv1->is_host_transfer() != send_recv2->is_host_transfer()) {
    return InternalError(
        "Expected instructions to have the same is-host-transfer property: "
        "%s, "
        "%s ",
        instr1->ToString(), instr2->ToString());
  }
  return OkStatus();
}

Status VerifySingleUser(const HloInstruction* instruction,
                        const absl::flat_hash_set<HloOpcode>& expected_users) {
  TF_RET_CHECK(instruction->users().size() == 1)
      << "The " << HloOpcodeString(instruction->opcode())
      << " instruction requires one consumer, found "
      << instruction->users().size();

  const HloInstruction* user = instruction->users().front();
  TF_RET_CHECK(expected_users.contains(user->opcode()))
      << "The consumer of a " << HloOpcodeString(instruction->opcode())
      << " instruction needs to be one of ("
      << absl::StrJoin(expected_users, ", ",
                       [](std::string* out, HloOpcode opcode) {
                         out->append(HloOpcodeString(opcode));
                       })
      << "), found " << HloOpcodeString(user->opcode());
  return OkStatus();
}

Status VerifySingleOperand(const HloInstruction* instruction,
                           const std::vector<HloOpcode>& expected_operands) {
  TF_RET_CHECK(instruction->operands().size() == 1)
      << "The " << HloOpcodeString(instruction->opcode())
      << " instruction requires one consumer, found "
      << instruction->users().size();

  const HloInstruction* operand = instruction->operand(0);
  TF_RET_CHECK(absl::c_find(expected_operands, operand->opcode()) !=
               expected_operands.end())
      << "The operand of a " << HloOpcodeString(instruction->opcode())
      << " instruction needs to be "
      << absl::StrJoin(expected_operands, " or ",
                       [](std::string* out, HloOpcode opcode) {
                         out->append(HloOpcodeString(opcode));
                       })
      << ", found " << HloOpcodeString(operand->opcode());
  return OkStatus();
}

// Checks asynchronous instruction pairs.
Status VerifyAsynchronousInstructionPairs(const HloModule& module) {
  // CopyStart must have a single CopyDone user.
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      switch (instruction->opcode()) {
        case HloOpcode::kAsyncStart: {
          TF_RETURN_IF_ERROR(VerifySingleUser(
              instruction, {HloOpcode::kAsyncUpdate, HloOpcode::kAsyncDone}));
          break;
        }
        case HloOpcode::kAsyncUpdate: {
          TF_RETURN_IF_ERROR(VerifySingleOperand(
              instruction, {HloOpcode::kAsyncStart, HloOpcode::kAsyncUpdate}));
          TF_RETURN_IF_ERROR(VerifySingleUser(
              instruction, {HloOpcode::kAsyncUpdate, HloOpcode::kAsyncDone}));
          break;
        }
        case HloOpcode::kAsyncDone: {
          TF_RETURN_IF_ERROR(VerifySingleOperand(
              instruction, {HloOpcode::kAsyncStart, HloOpcode::kAsyncUpdate}));
          break;
        }
        case HloOpcode::kAllReduceStart: {
          TF_RETURN_IF_ERROR(
              VerifySingleUser(instruction, {HloOpcode::kAllReduceDone}));
          break;
        }
        case HloOpcode::kAllReduceDone: {
          TF_RETURN_IF_ERROR(
              VerifySingleOperand(instruction, {HloOpcode::kAllReduceStart}));
          break;
        }
        case HloOpcode::kCopyStart: {
          TF_RETURN_IF_ERROR(
              VerifySingleUser(instruction, {HloOpcode::kCopyDone}));
          break;
        }
        case HloOpcode::kCopyDone: {
          TF_RETURN_IF_ERROR(
              VerifySingleOperand(instruction, {HloOpcode::kCopyStart}));
          break;
        }
        case HloOpcode::kCollectivePermuteStart: {
          TF_RETURN_IF_ERROR(VerifySingleUser(
              instruction, {HloOpcode::kCollectivePermuteDone}));
          break;
        }
        case HloOpcode::kCollectivePermuteDone: {
          TF_RETURN_IF_ERROR(VerifySingleOperand(
              instruction, {HloOpcode::kCollectivePermuteStart}));
          break;
        }
        default:
          break;
      }
    }
  }
  return OkStatus();
}

// Checks that AllReduce instructions in the module are either all layout
// constrained or all unconstrained.
Status VerifyLayoutConstrainedAllReduce(const HloModule& module) {
  const HloAllReduceInstruction* reference = nullptr;
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if ((instruction->opcode() != HloOpcode::kAllReduce) &&
          (instruction->opcode() != HloOpcode::kAllReduceStart)) {
        continue;
      }
      auto all_reduce = DynCast<HloAllReduceInstruction>(instruction);
      if (!reference) {
        reference = all_reduce;
      }
      if (reference->constrain_layout() != all_reduce->constrain_layout()) {
        return FailedPrecondition(
            "HloModule has a mix of layout constrained and unconstrained "
            "AllReduce instructions.");
      }
    }
  }
  return OkStatus();
}

// Checks various invariants of channel instructions (send/recv and
// collectives).
Status VerifyChannels(const HloModule& module) {
  absl::flat_hash_map<int64_t, std::vector<const HloInstruction*>>
      channel_instructions;

  // Send/Recv instruction must have a single user: the corresponding
  // SendDone/RecvDone. with matching channel.
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      auto channel_instr = DynCast<HloChannelInstruction>(instruction);
      if (!channel_instr || !channel_instr->channel_id()) {
        continue;
      }
      channel_instructions[*channel_instr->channel_id()].push_back(instruction);

      switch (instruction->opcode()) {
        case HloOpcode::kSend: {
          TF_RET_CHECK(instruction->users().size() == 1);
          const HloInstruction* send_done = instruction->users().front();
          TF_RET_CHECK(send_done->opcode() == HloOpcode::kSendDone);
          TF_RETURN_IF_ERROR(CheckSameChannel(instruction, send_done));
          TF_RETURN_IF_ERROR(CheckSameIsHostTransfer(instruction, send_done));
          break;
        }
        case HloOpcode::kRecv: {
          TF_RET_CHECK(instruction->users().size() == 1);
          const HloInstruction* recv_done = instruction->users().front();
          TF_RET_CHECK(recv_done->opcode() == HloOpcode::kRecvDone);
          TF_RETURN_IF_ERROR(CheckSameChannel(instruction, recv_done));
          TF_RETURN_IF_ERROR(CheckSameIsHostTransfer(instruction, recv_done));
          break;
        }
        case HloOpcode::kSendDone:
          TF_RET_CHECK(instruction->operands().size() == 1);
          TF_RET_CHECK(instruction->operand(0)->opcode() == HloOpcode::kSend);
          break;
        case HloOpcode::kRecvDone:
          TF_RET_CHECK(instruction->operands().size() == 1);
          TF_RET_CHECK(instruction->operand(0)->opcode() == HloOpcode::kRecv);
          break;
        default:
          break;
      }
    }
  }

  // Iterate over each channel to check invariants.
  for (auto& pair : channel_instructions) {
    auto& instructions = pair.second;
    const HloInstruction* first = instructions[0];
    auto sendrecv = DynCast<HloSendRecvInstruction>(first);
    if (sendrecv) {
      absl::flat_hash_set<HloOpcode> opcodes;
      for (const HloInstruction* instr : instructions) {
        opcodes.insert(instr->opcode());
        auto cast = DynCast<HloSendRecvInstruction>(instr);
        TF_RET_CHECK(cast != nullptr)
            << "channel " << pair.first
            << " is used for different types of channel instructions";
      }
      if (sendrecv->is_host_transfer()) {
        TF_RET_CHECK(instructions.size() == 2)
            << "channel " << pair.first
            << " is used for multiple host send/recv instructions";
      } else {
        TF_RET_CHECK(instructions.size() == opcodes.size())
            << "channel " << pair.first
            << " is used for multiple send/recv instructions";
      }
    } else {
      for (const HloInstruction* instr : instructions) {
        TF_RET_CHECK(first->opcode() == instr->opcode())
            << "channel " << pair.first
            << " is used for different types of channel instructions";
      }
    }
  }

  return OkStatus();
}

// CHECKs various invariants of a fusion instruction.
Status CheckFusionInstruction(HloInstruction* fusion) {
  // The parent fusion instruction of the fusion computation must be 'fusion'.
  HloComputation* fused_computation = fusion->fused_instructions_computation();
  if (fusion != fused_computation->FusionInstruction()) {
    return InternalError(
        "Instruction of fused computation does not match expected "
        "instruction "
        "%s.",
        fusion->ToString());
  }

  // Fused root instruction and fused parameters must all be owned by the
  // fusion computation.
  bool root_owned = false;
  const std::vector<HloInstruction*>& fused_parameters =
      fusion->fused_parameters();
  const HloInstruction* fused_root = fusion->fused_expression_root();
  std::vector<bool> parameter_owned(fused_parameters.size(), false);
  for (auto* instruction : fused_computation->instructions()) {
    if (fused_root == instruction) {
      if (root_owned) {
        return InternalError("Root appears more than once in %s.",
                             fusion->ToString());
      }
      root_owned = true;
    }
    for (int i = 0; i < fused_parameters.size(); ++i) {
      if (fused_parameters[i] == instruction) {
        if (parameter_owned[i]) {
          return InternalError("Parameter appears more than once in %s.",
                               fusion->ToString());
        }
        parameter_owned[i] = true;
      }
    }
  }
  if (!root_owned) {
    return InternalError("Root not found in computation of %s.",
                         fusion->ToString());
  }
  // Make sure all the parameter_owned entries are set
  for (int i = 0; i < parameter_owned.size(); i++) {
    if (!parameter_owned[i]) {
      return InternalError("Parameter %d not found in computation of %s.", i,
                           fusion->ToString());
    }
  }

  // Fused root must have no users.
  if (fused_root->user_count() != 0) {
    return InternalError("Root of %s may not have users.", fusion->ToString());
  }

  // All uses of fused instructions must be in the fusion computation, and
  // every non-root instruction must have at least one use.
  for (auto* instruction :
       fusion->fused_instructions_computation()->instructions()) {
    if (instruction != fused_root) {
      if (instruction->user_count() == 0) {
        return InternalError("Non-root instruction %s in %s must have users.",
                             instruction->ToString(), fusion->ToString());
      }
      for (auto& user : instruction->users()) {
        if (fused_computation != user->parent()) {
          return InternalError(
              "Non-root instruction %s in %s may not have external users.",
              instruction->ToString(), fusion->ToString());
        }
      }
    }
  }

  // Fused parameter instructions must be numbered contiguously and match up
  // (shapes equal) with their respective operand.
  CHECK_EQ(fusion->operands().size(), fused_parameters.size());
  std::vector<bool> parameter_numbers(fused_parameters.size(), false);
  for (auto fused_param : fused_parameters) {
    int64_t param_no = fused_param->parameter_number();
    if (param_no < 0) {
      return InternalError("Unexpected negative parameter number %d in %s.",
                           param_no, fusion->ToString());
    }
    if (param_no >= fused_parameters.size()) {
      return InternalError(
          "Unexpected parameter number %d in %s: higher then number of "
          "parameters %lu.",
          param_no, fusion->ToString(), fused_parameters.size());
    }
    if (parameter_numbers[param_no]) {
      return InternalError(
          "Did not expect parameter number %d more than once in %s.", param_no,
          fusion->ToString());
    }
    parameter_numbers[param_no] = true;
  }
  // Make sure all the parameter_numbers entries were seen.
  for (int i = 0; i < parameter_numbers.size(); i++) {
    if (!parameter_numbers[i]) {
      return InternalError("Did not see parameter number %d in %s.", i,
                           fusion->ToString());
    }
  }

  TF_RET_CHECK(fusion->called_computations() ==
               absl::Span<HloComputation* const>(
                   {fusion->fused_instructions_computation()}))
      << "Fusion HLO calls computations other than the "
         "fused_instructions_computation: "
      << fusion->ToString() << " fusion->fused_instructions_computation(): "
      << fusion->fused_instructions_computation()->ToString()
      << " fusion->called_computations(): "
      << ComputationsToString(fusion->called_computations());

  for (const auto& fused : fusion->fused_instructions()) {
    TF_RET_CHECK(fused->parent() == fusion->fused_instructions_computation())
        << "Fused HLO was missing a parent: " << fused->ToString()
        << " parent: " << fused->parent()
        << " computation: " << fusion->parent();
  }

  // TODO(b/65423525): We'd like to check that all operands are distinct.
  // This is currently disabled due to the invariant being violated by
  // multi-output fusion.
  return OkStatus();
}

// Checks that the operand shapes are compatible to the output shape, i.e.,
// that there are no implicit broadcasts.
Status CheckElementwiseInstruction(HloInstruction* instruction) {
  const Shape& out_shape = instruction->shape();
  for (HloInstruction* operand : instruction->operands()) {
    const Shape& operand_shape = operand->shape();
    if (!ShapeUtil::CompatibleIgnoringElementType(operand_shape, out_shape)) {
      return FailedPrecondition(
          "Implicit broadcast is not allowed in HLO."
          "Found different shapes for instruction %s.\n"
          "output: %s\noperand: %s\n",
          HloOpcodeString(instruction->opcode()),
          ShapeUtil::HumanString(out_shape),
          ShapeUtil::HumanString(operand_shape));
    }
  }
  if (auto* comparison = DynCast<HloCompareInstruction>(instruction)) {
    const Shape& operand_shape = comparison->operand(1)->shape();
    PrimitiveType operand_element_type = operand_shape.element_type();
    Comparison::Type default_comparison_type =
        Comparison::DefaultComparisonType(operand_element_type);
    if (primitive_util::IsFloatingPointType(operand_element_type)) {
      if (comparison->type() != Comparison::Type::kFloat &&
          comparison->type() != Comparison::Type::kFloatTotalOrder) {
        return FailedPrecondition(
            "Expected comparison type %s or %s.\n"
            "actual: %s\noperand: %s\n",
            ComparisonTypeToString(Comparison::Type::kFloat),
            ComparisonTypeToString(Comparison::Type::kFloatTotalOrder),
            ComparisonTypeToString(comparison->type()),
            ShapeUtil::HumanString(operand_shape));
      }
    } else if (comparison->type() != default_comparison_type) {
      return FailedPrecondition(
          "Expected comparison type %s.\n"
          "actual: %s\noperand: %s\n",
          ComparisonTypeToString(default_comparison_type),
          ComparisonTypeToString(comparison->type()),
          ShapeUtil::HumanString(operand_shape));
    }
  }
  return OkStatus();
}

// Visitor which verifies various fields on the HLO instruction. This class does
// not check result shape as that is checked in the ShapeVerifier.
class InstructionVerifier : public DfsHloVisitorWithDefault {
 public:
  explicit InstructionVerifier(const HloVerifierOpts& opts) : opts_(opts) {}

  Status DefaultAction(HloInstruction*) override { return OkStatus(); }

  Status HandleFusion(HloInstruction* fusion) override {
    TF_RETURN_IF_ERROR(CheckCallableInstructionThreadName(
        fusion, /*skip_nested_async_op_check*/ false));
    return CheckFusionInstruction(fusion);
  }

  Status HandleBroadcast(HloInstruction* broadcast) override {
    // If you see this failure then someone has confused the difference
    // between the HLO broadcast op, and the UserComputation broadcast
    // op. See https://groups.google.com/forum/#!topic/xla-dev/9LqijHmTt_I
    // or ComputationLowerer::Visit()
    TF_RET_CHECK(broadcast->dimensions().size() ==
                 broadcast->operand(0)->shape().rank())
        << "Broadcast HLO (" << broadcast->ToShortString()
        << ") has invalid number of dimensions: "
        << broadcast->dimensions().size()
        << " != " << broadcast->operand(0)->shape().rank();
    return OkStatus();
  }

  Status HandleBitcastConvert(HloInstruction* c) override {
    // Shape verifier will check all we need.
    return OkStatus();
  }

  Status HandleWhile(HloInstruction* xla_while) override {
    auto* while_cond = xla_while->while_condition();
    auto* while_body = xla_while->while_body();
    if (while_cond->num_parameters() != 1) {
      return FailedPrecondition(
          "While condition must have exactly 1 parameter; had %d : %s",
          while_cond->num_parameters(), while_cond->ToString());
    }
    if (while_body->num_parameters() != 1) {
      return FailedPrecondition(
          "While body must have exactly 1 parameter; had %d : %s",
          while_body->num_parameters(), while_body->ToString());
    }
    if (xla_while->operand_count() != 1) {
      return FailedPrecondition(
          "While loop must have exactly one operand; had %d : %s",
          xla_while->operand_count(), xla_while->ToString());
    }
    return OkStatus();
  }

  Status HandleConditional(HloInstruction* conditional) override {
    for (int b = 0; b < conditional->branch_count(); ++b) {
      if (conditional->branch_computation(b)->num_parameters() != 1) {
        return FailedPrecondition(
            "Branch computation %s of %s must have 1 parameter instead of %d",
            conditional->branch_computation(b)->name(), conditional->ToString(),
            conditional->branch_computation(b)->num_parameters());
      }
    }
    return OkStatus();
  }

  Status HandleElementwiseUnary(HloInstruction* instruction) override {
    return CheckElementwiseInstruction(instruction);
  }

  Status HandleElementwiseBinary(HloInstruction* instruction) override {
    return CheckElementwiseInstruction(instruction);
  }

  Status HandleGetTupleElement(HloInstruction* gte) override {
    TF_RET_CHECK(gte->operand(0)->shape().IsTuple());
    return OkStatus();
  }

  Status HandleTranspose(HloInstruction* transpose) override {
    const Shape& shape = transpose->shape();
    const HloInstruction* operand = transpose->operand(0);
    TF_RET_CHECK(shape.dimensions().size() == transpose->dimensions().size());
    TF_RET_CHECK(shape.dimensions().size() ==
                 transpose->operand(0)->shape().dimensions().size());
    TF_RET_CHECK(std::equal(
        shape.dimensions().begin(), shape.dimensions().end(),
        Permute(operand->shape().dimensions(), transpose->dimensions())
            .begin()))
        << "shape: " << shape << ", operand->shape(): " << shape
        << ", dimensions: {" << absl::StrJoin(transpose->dimensions(), ", ")
        << "}";
    return OkStatus();
  }

  Status HandleAllReduce(HloInstruction* crs) override {
    if (crs->channel_id().has_value()) {
      TF_RET_CHECK(crs->channel_id().value() > 0)
          << "All reduce channel id must be greater than 0 for "
          << crs->ToShortString();
    }
    return OkStatus();
  }

  Status Preprocess(HloInstruction* instruction) override {
    auto previous = instructions_by_name_.find(instruction->name());
    TF_RET_CHECK(previous == instructions_by_name_.end())
        << "HLO has name that is not unique within module:\n"
        << instruction->ToString()
        << " in computation: " << instruction->parent()->name()
        << "\nPrevious HLO with same name:\n"
        << previous->second->ToString()
        << " in computation: " << previous->second->parent()->name();
    instructions_by_name_[instruction->name()] = instruction;
    return OkStatus();
  }

  Status Postprocess(HloInstruction* instruction) override {
    if (!opts_.InstructionCanChangeLayout(instruction) &&
        LayoutUtil::IsDenseArray(instruction->shape())) {
      const Shape& result_shape = instruction->shape();
      const Layout& result_layout = result_shape.layout();
      for (HloInstruction* operand : instruction->operands()) {
        const Shape& operand_shape = operand->shape();
        if (LayoutUtil::IsDenseArray(operand_shape) &&
            operand_shape.rank() == result_shape.rank()) {
          const Layout& operand_layout = operand_shape.layout();
          TF_RET_CHECK(LayoutUtil::Equal(result_layout, operand_layout))
              << "Instruction shouldn't change layouts "
              << instruction->ToString() << " From " << result_shape << " To "
              << operand_shape;
        }
      }
    }

    return OkStatus();
  }

 private:
  absl::flat_hash_map<std::string, const HloInstruction*> instructions_by_name_;
  const HloVerifierOpts& opts_;
};

}  // namespace

StatusOr<bool> HloVerifier::Run(HloModule* module) {
  auto disabled = module->config().debug_options().xla_disable_hlo_passes();
  if (std::find(disabled.begin(), disabled.end(), name()) != disabled.end()) {
    return false;
  }
  auto status_or_changed = [&]() -> StatusOr<bool> {
    TF_RET_CHECK(!module->name().empty());

    if (module->entry_computation()->IsFusionComputation()) {
      return InvalidArgument(
          "Module entry computation cannot be a fusion computation");
    }

    TF_RETURN_IF_ERROR(VerifyHloStructure(module));
    TF_RETURN_IF_ERROR(VerifyAsynchronousInstructionPairs(*module));
    TF_RETURN_IF_ERROR(VerifyChannels(*module));

    std::unique_ptr<ShapeVerifier> shape_verifier =
        target_metadata_->GetVerifier();
    InstructionVerifier instruction_verifier(
        target_metadata_->GetVerifierOpts());
    for (auto* computation : module->computations()) {
      TF_RETURN_IF_ERROR(computation->Accept(shape_verifier.get()));
      TF_RETURN_IF_ERROR(computation->Accept(&instruction_verifier));
    }

    TF_RETURN_IF_ERROR(shape_verifier->VerifyEntryComputationLayout(*module));
    TF_RETURN_IF_ERROR(VerifyEntryAndExitShapes(*module));

    // If the module has a schedule, it must be valid.
    if (module->has_schedule()) {
      TF_RETURN_IF_ERROR(module->schedule().Verify());
    }

    TF_RETURN_IF_ERROR(module->input_output_alias_config().Verify(
        *module, [this](const Shape& shape) -> int64_t {
          if (target_metadata_->GetVerifierOpts().IsLayoutSensitive()) {
            return target_metadata_->GetVerifierOpts().ShapeSize(shape);
          } else {
            return 0;
          }
        }));

    TF_RETURN_IF_ERROR(module->dynamic_parameter_binding().Verify(*module));
    TF_RETURN_IF_ERROR(VerifyLayoutConstrainedAllReduce(*module));
    return false;
  }();
  if (status_or_changed.ok()) {
    return status_or_changed.ValueOrDie();
  }
  return Status(status_or_changed.status().code(),
                absl::StrCat("during context [", context_, "]: ",
                             status_or_changed.status().error_message()));
}

}  // namespace xla
