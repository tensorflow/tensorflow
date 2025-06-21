/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/hlo_verifier.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool IsCallerInstruction(HloInstruction* hlo) {
  return HloInstruction::MightHaveCalledComputations(hlo->opcode());
}

absl::Status CheckOperandCount(const HloInstruction* hlo, int expected) {
  if (hlo->operand_count() != expected) {
    return Internal("Expected %d operands for %s instruction: %s", expected,
                    HloOpcodeString(hlo->opcode()), hlo->ToString());
  }
  return absl::OkStatus();
}

int64_t GetSubgroupSize(HloCollectiveInstruction* hlo,
                        CollectiveOpGroupMode group_mode) {
  const HloModuleConfig& config = hlo->GetModule()->config();
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
      // Empty replica groups not allowed in this mode.
      return hlo->replica_groups()[0].replica_ids_size();
    case CollectiveOpGroupMode::kCrossPartition:
      return hlo->replica_groups().empty()
                 ? config.num_partitions()
                 : hlo->replica_groups()[0].replica_ids_size();
  }
}

absl::Status CheckUnaryOpWithResultAccuracy(HloInstruction* unary) {
  HloOpcode opcode = unary->opcode();
  if (unary->has_result_accuracy()) {
    if (IsUnaryOpWithResultAccuracy(unary->opcode())) {
      return absl::OkStatus();
    } else {
      return Internal("Unary op with result accuracy is not supported for %s",
                      HloOpcodeString(opcode));
    }
  }
  return absl::OkStatus();
}
}  // namespace

/*static*/ absl::Status ShapeVerifier::CheckParameterCount(
    const HloInstruction* calling_instruction,
    const HloComputation* computation, int expected) {
  if (computation->num_parameters() != expected) {
    return Internal(
        "Expected computation %s called from %s to have %d parameters, has %d",
        computation->name(), calling_instruction->name(), expected,
        computation->num_parameters());
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::Preprocess(HloInstruction* hlo) {
  if (!hlo->called_computations().empty() && !IsCallerInstruction(hlo)) {
    return Internal(
        "Called computations specified for non-caller instruction %s",
        hlo->ToString());
  }
  std::optional<int> arity = HloOpcodeArity(hlo->opcode());
  if (arity) {
    TF_RETURN_IF_ERROR(CheckOperandCount(hlo, *arity));
  }
  if (!opts_.allow_unbounded_dynamism && hlo->shape().is_unbounded_dynamic()) {
    return InvalidArgument("Unbounded dynamism is disabled for instruction: %s",
                           hlo->ToString());
  }
  if (hlo->shape().has_layout()) {
    if (hlo->shape().layout().minor_to_major().size() !=
        hlo->shape().dimensions().size()) {
      return InvalidArgument(
          "Instruction has mismatched minor-to-major size and dimension size: "
          "%s",
          hlo->ToString());
    }
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleElementwiseUnary(HloInstruction* hlo) {
  return CheckUnaryShape(hlo);
}

absl::Status ShapeVerifier::HandleElementwiseBinary(HloInstruction* hlo) {
  return CheckBinaryShape(hlo);
}

absl::Status ShapeVerifier::HandleClamp(HloInstruction* clamp) {
  return CheckTernaryShape(clamp);
}

absl::Status ShapeVerifier::HandleSelect(HloInstruction* select) {
  return CheckTernaryShape(select);
}

absl::Status ShapeVerifier::HandleConcatenate(HloInstruction* concatenate) {
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : concatenate->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(concatenate,
                    ShapeInference::InferConcatOpShape(
                        operand_shapes, concatenate->concatenate_dimension()));
}

absl::Status ShapeVerifier::HandleConvert(HloInstruction* convert) {
  return CheckShape(convert, ShapeInference::InferConvertShape(
                                 convert->operand(0)->shape(),
                                 convert->shape().element_type()));
}

absl::Status ShapeVerifier::HandleBitcastConvert(HloInstruction* convert) {
  return CheckShape(convert, ShapeInference::InferBitcastConvertShape(
                                 convert->operand(0)->shape(),
                                 convert->shape().element_type()));
}

absl::Status ShapeVerifier::HandleStochasticConvert(HloInstruction* convert) {
  return CheckShape(
      convert, ShapeInference::InferStochasticConvertShape(
                   convert->operand(0)->shape(), convert->operand(1)->shape(),
                   convert->shape().element_type()));
}

absl::Status ShapeVerifier::HandleCopy(HloInstruction* copy) {
  return CheckUnaryShape(copy);
}

absl::Status ShapeVerifier::HandleDot(HloInstruction* dot) {
  auto sparsity = Cast<HloDotInstruction>(dot)->sparsity();
  TF_RETURN_IF_ERROR(
      CheckOperandCount(dot, HloDotInstruction::kOperands + sparsity.size()));
  TF_ASSIGN_OR_RETURN(
      const Shape expected,
      ShapeInference::InferDotOpShape(
          dot->operand(0)->shape(), dot->operand(1)->shape(),
          dot->dot_dimension_numbers(),
          /*preferred_element_type=*/dot->shape().element_type(), sparsity));

  for (int i = 0; i < sparsity.size(); ++i) {
    const SparsityDescriptor& descriptor = sparsity[i];
    TF_RET_CHECK(descriptor.index() == 0 || descriptor.index() == 1);
    TF_ASSIGN_OR_RETURN(const Shape expected_metadata_shape,
                        ShapeInference::InferSparseDotMetadataShape(
                            dot->operand(descriptor.index())->shape(),
                            dot->dot_dimension_numbers(), descriptor));
    const Shape actual_metadata_shape =
        dot->operand(HloDotInstruction::kOperands + i)->shape();
    if (!ShapeUtil::Compatible(actual_metadata_shape,
                               expected_metadata_shape)) {
      return Internal(
          "Expected sparse dot metadata to have shape equal to %s, actual "
          "shape is %s:\n%s",
          StringifyShape(expected_metadata_shape),
          StringifyShape(actual_metadata_shape), dot->ToString());
    }
  }
  return CheckShape(dot, expected);
}

absl::Status ShapeVerifier::HandleRaggedDot(HloInstruction* ragged_dot) {
  TF_RETURN_IF_ERROR(
      CheckOperandCount(ragged_dot, HloRaggedDotInstruction::kOperands));
  TF_ASSIGN_OR_RETURN(
      const Shape expected,
      ShapeInference::InferRaggedDotOpShape(
          ragged_dot->operand(0)->shape(), ragged_dot->operand(1)->shape(),
          ragged_dot->operand(2)->shape(),
          ragged_dot->ragged_dot_dimension_numbers(),
          /*preferred_element_type=*/ragged_dot->shape().element_type()));
  return CheckShape(ragged_dot, expected);
}

absl::Status ShapeVerifier::HandleConvolution(HloInstruction* convolution) {
  TF_ASSIGN_OR_RETURN(
      Shape expected,
      ShapeInference::InferConvolveShape(
          convolution->operand(0)->shape(), convolution->operand(1)->shape(),
          convolution->feature_group_count(), convolution->batch_group_count(),
          convolution->window(), convolution->convolution_dimension_numbers(),
          /*preferred_element_type=*/convolution->shape().element_type()));

  return CheckShape(convolution, expected);
}

absl::Status ShapeVerifier::HandleFft(HloInstruction* fft) {
  TF_ASSIGN_OR_RETURN(
      const Shape expected,
      ShapeInference::InferFftShape(fft->operand(0)->shape(), fft->fft_type(),
                                    fft->fft_length()));
  return CheckShape(fft, expected);
}

absl::Status ShapeVerifier::HandleTriangularSolve(HloInstruction* hlo) {
  TF_ASSIGN_OR_RETURN(const Shape expected,
                      ShapeInference::InferTriangularSolveShape(
                          hlo->operand(0)->shape(), hlo->operand(1)->shape(),
                          hlo->triangular_solve_options()));
  return CheckShape(hlo, expected);
}

absl::Status ShapeVerifier::HandleCholesky(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(CheckOperandCount(hlo, 1));
  TF_ASSIGN_OR_RETURN(const Shape expected, ShapeInference::InferCholeskyShape(
                                                hlo->operand(0)->shape()));
  return CheckShape(hlo, expected);
}

absl::Status ShapeVerifier::HandleOptimizationBarrier(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(CheckOperandCount(hlo, 1));
  return CheckShape(hlo, hlo->operand(0)->shape());
}

bool ShapeVerifier::ShapesSame(const Shape& a, const Shape& b,
                               Shape::Equal equal) {
  if (!opts_.layout_sensitive) {
    return ShapeUtil::Compatible(a, b);
  }
  return equal(a, b);
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
static absl::Status CheckReplicaGroups(HloInstruction* hlo,
                                       CollectiveOpGroupMode group_mode,
                                       bool uniform_replica_group_size = true) {
  if (!hlo->replica_groups().empty()) {
    absl::flat_hash_set<int64_t> replicas_seen;
    for (const ReplicaGroup& g : hlo->replica_groups()) {
      if (g.replica_ids().empty()) {
        return Internal("Instruction cannot have an empty replica group: %s",
                        hlo->ToString());
      }
      for (int64_t i : g.replica_ids()) {
        if (!replicas_seen.insert(i).second) {
          return Internal(
              "Replica %d is repeated in instruction's replica-groups: %s", i,
              hlo->ToString());
        }
      }
    }
    size_t n = replicas_seen.size();
    for (int64_t i = 0; i < n; ++i) {
      if (!replicas_seen.count(i)) {
        return Internal(
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

  return absl::OkStatus();
}

static absl::Status CheckCommonAllGatherInvariants(
    HloInstruction* hlo, int64_t* computed_shard_count) {
  auto ag = Cast<HloAllGatherInstruction>(hlo);
  CHECK_NE(computed_shard_count, nullptr) << "Expected a shard count as input";
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                      GetCollectiveOpGroupMode(ag->channel_id().has_value(),
                                               ag->use_global_device_ids()));
  TF_RETURN_IF_ERROR(CheckReplicaGroups(ag, group_mode));
  TF_RET_CHECK(ag->all_gather_dimension() >= 0);
  TF_RET_CHECK(ag->operand_count() >= 1);

  int64_t shard_count;
  for (int64_t i = 0; i < ag->operand_count(); ++i) {
    TF_RET_CHECK(
        ag->all_gather_dimension() <
        static_cast<int64_t>(ag->operand(i)->shape().dimensions().size()));

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
    TF_RET_CHECK(ag->all_gather_dimension() <
                 static_cast<int64_t>(output_shape.dimensions().size()));
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
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleAllGather(HloInstruction* hlo) {
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

absl::Status ShapeVerifier::HandleAllGatherStart(HloInstruction* hlo) {
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

absl::Status ShapeVerifier::HandleAllGatherDone(HloInstruction* hlo) {
  return CheckShape(
      hlo, ShapeInference::InferAllGatherDoneShape(hlo->operand(0)->shape()));
}

absl::Status ShapeVerifier::HandleAllReduce(HloInstruction* hlo) {
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

absl::Status ShapeVerifier::HandleReduceScatter(HloInstruction* hlo) {
  auto ars = Cast<HloReduceScatterInstruction>(hlo);
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                      GetCollectiveOpGroupMode(ars->channel_id().has_value(),
                                               ars->use_global_device_ids()));
  TF_RETURN_IF_ERROR(CheckReplicaGroups(ars, group_mode));
  TF_RET_CHECK(ars->scatter_dimension() >= 0);
  TF_RET_CHECK(ars->operand_count() >= 1);

  for (int64_t i = 0; i < ars->operand_count(); ++i) {
    TF_RET_CHECK(
        ars->scatter_dimension() <
        static_cast<int64_t>(ars->operand(i)->shape().dimensions().size()));

    const Shape& output_shape = (ars->operand_count() == 1)
                                    ? ars->shape()
                                    : ars->shape().tuple_shapes(i);
    TF_RET_CHECK(ars->scatter_dimension() <
                 static_cast<int64_t>(output_shape.dimensions().size()));
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

absl::Status ShapeVerifier::HandleAllReduceStart(HloInstruction* hlo) {
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

absl::Status ShapeVerifier::HandleAllReduceDone(HloInstruction* hlo) {
  return CheckShape(
      hlo, ShapeInference::InferAllReduceDoneShape(hlo->operand(0)->shape()));
}

absl::Status ShapeVerifier::HandleAllToAll(HloInstruction* hlo) {
  auto* all_to_all = Cast<HloAllToAllInstruction>(hlo);
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                      GetCollectiveOpGroupMode(
                          all_to_all->channel_id().has_value(), std::nullopt));

  TF_RETURN_IF_ERROR(CheckReplicaGroups(hlo, group_mode));

  TF_RET_CHECK(all_to_all != nullptr);
  const int64_t split_count = GetSubgroupSize(all_to_all, group_mode);
  if (all_to_all->split_dimension()) {
    TF_RET_CHECK(hlo->operand_count() == 1);
    return CheckShape(
        hlo, ShapeInference::InferAllToAllShape(
                 hlo->operand(0)->shape(), *all_to_all->split_dimension(),
                 *all_to_all->split_dimension(), split_count));
  } else {
    TF_RET_CHECK(hlo->operand_count() == split_count);
    std::vector<const Shape*> operand_shapes;
    for (const HloInstruction* operand : hlo->operands()) {
      operand_shapes.push_back(&operand->shape());
    }
    return CheckShape(hlo,
                      ShapeInference::InferAllToAllTupleShape(operand_shapes));
  }
}

absl::Status ShapeVerifier::HandleRaggedAllToAll(HloInstruction* hlo) {
  auto* all_to_all = Cast<HloRaggedAllToAllInstruction>(hlo);
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                      GetCollectiveOpGroupMode(
                          all_to_all->channel_id().has_value(), std::nullopt));

  TF_RETURN_IF_ERROR(CheckReplicaGroups(hlo, group_mode));

  const int64_t kNumRaggedOperands = 6;
  TF_RET_CHECK(all_to_all != nullptr);
  TF_RET_CHECK(hlo->operand_count() == kNumRaggedOperands);
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_shapes.push_back(&operand->shape());
  }

  // Check that *_offsets/*_sizes operands all have the same shape and
  // are rank 1 or rank 2.
  const int64_t kOffsetsSizesOperandsStart = 2;
  for (int64_t i = kOffsetsSizesOperandsStart + 1; i < kNumRaggedOperands;
       ++i) {
    if (operand_shapes[i - 1]->dimensions().size() != 1 &&
        operand_shapes[i - 1]->dimensions().size() != 2) {
      return Internal("RaggedAllToAll operand %d must be rank 1 or 2: %s",
                      i - 1, hlo->ToString());
    }
    if (!ShapeUtil::Equal(*operand_shapes[i - 1], *operand_shapes[i])) {
      return Internal(
          "RaggedAllToAll operands have different shapes (%d, %d): %s", i - 1,
          i, hlo->ToString());
    }
  }

  return CheckShape(hlo,
                    ShapeInference::InferRaggedAllToAllShape(operand_shapes));
}

absl::Status ShapeVerifier::HandlePartitionId(HloInstruction* hlo) {
  return CheckShape(hlo, ShapeUtil::MakeShape(U32, {}));
}

absl::Status ShapeVerifier::HandleReplicaId(HloInstruction* hlo) {
  return CheckShape(hlo, ShapeUtil::MakeShape(U32, {}));
}

namespace {

absl::Status CheckBufferOffset(const Shape& buffer_shape,
                               const Shape& buffer_offset_shape) {
  if (!buffer_offset_shape.IsTuple()) {
    return Internal("Buffer offset is not tuple.");
  }
  bool all_is_array =
      absl::c_all_of(buffer_offset_shape.tuple_shapes(),
                     [](const Shape& shape) { return shape.IsArray(); });
  bool all_is_tuple =
      absl::c_all_of(buffer_offset_shape.tuple_shapes(),
                     [](const Shape& shape) { return shape.IsTuple(); });
  if (!all_is_array && !all_is_tuple) {
    return Internal(
        "Buffer offset should either be a tuple of arrays or "
        " a tuple of tuples.");
  }

  if (all_is_tuple) {
    if (absl::c_any_of(buffer_offset_shape.tuple_shapes(),
                       [&buffer_shape](const Shape& shape) {
                         return ShapeUtil::TupleElementCount(shape) !=
                                buffer_shape.dimensions().size();
                       })) {
      return Internal(
          "Buffer offset index should have the same number of "
          "elements as the buffer's rank.");
    }
  } else {
    if (buffer_offset_shape.tuple_shapes().size() !=
        buffer_shape.dimensions().size()) {
      return Internal(
          "Buffer offset index should have the same number of "
          "elements as the buffer's rank.");
    }
  }
  return absl::OkStatus();
}

absl::Status CheckInplaceCollectivePermute(
    HloCollectivePermuteInstruction* collective_permute) {
  if (!collective_permute->inplace()) {
    return absl::OkStatus();
  }
  // TODO support grouped partial collective permute
  if (collective_permute->operand_count() != 4) {
    return Internal("Unexpected number of operands: %d.",
                    collective_permute->operand_count());
  }

  const Shape& input_buffer_shape = collective_permute->operand(0)->shape();
  const Shape& output_buffer_shape = collective_permute->operand(1)->shape();
  const Shape& input_offset_shape = collective_permute->operand(2)->shape();
  const Shape& output_offset_shape = collective_permute->operand(3)->shape();

  if (input_buffer_shape.IsArray() && output_buffer_shape.IsArray()) {
    TF_RETURN_IF_ERROR(
        CheckBufferOffset(input_buffer_shape, input_offset_shape));
    TF_RETURN_IF_ERROR(
        CheckBufferOffset(output_buffer_shape, output_offset_shape));
  } else if (input_buffer_shape.IsTuple() && output_buffer_shape.IsTuple()) {
    if (ShapeUtil::TupleElementCount(input_buffer_shape) !=
        ShapeUtil::TupleElementCount(output_buffer_shape)) {
      return Internal("Unmatching input buffers and output buffers.");
    }
    if (!input_offset_shape.IsTuple() ||
        ShapeUtil::TupleElementCount(input_offset_shape) !=
            ShapeUtil::TupleElementCount(input_buffer_shape)) {
      return Internal("Unmatching input buffers and input offset.");
    }

    for (int i = 0; i < input_buffer_shape.tuple_shapes().size(); ++i) {
      TF_RETURN_IF_ERROR(CheckBufferOffset(input_buffer_shape.tuple_shapes(i),
                                           input_offset_shape.tuple_shapes(i)));
    }
    if (!output_offset_shape.IsTuple() ||
        ShapeUtil::TupleElementCount(output_offset_shape) !=
            ShapeUtil::TupleElementCount(output_buffer_shape)) {
      return Internal("Unmatching output buffers and output offset.");
    }
    for (int i = 0; i < output_buffer_shape.tuple_shapes().size(); ++i) {
      TF_RETURN_IF_ERROR(
          CheckBufferOffset(output_buffer_shape.tuple_shapes(i),
                            output_offset_shape.tuple_shapes(i)));
    }
  } else {
    return Internal("Unmatching input buffers and output buffers.");
  }
  return absl::OkStatus();
}

absl::Status CheckDuplicatedSourceOrTarget(
    HloCollectivePermuteInstruction* collective_permute) {
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                      GetCollectiveOpGroupMode(collective_permute));

  // A source or target cannot appear twice in the collective-permute's
  // source-target pairs. Also, based on the group formation mode, check if the
  // source and target IDs are within expected range.

  // Note: for collective-permute, only kCrossReplica and kCrossPartition modes
  // are valid.
  const HloModuleConfig& config = collective_permute->GetModule()->config();
  const int64_t limit = group_mode == CollectiveOpGroupMode::kCrossReplica
                            ? config.replica_count()
                            : config.num_partitions();
  absl::flat_hash_map<int64_t, std::vector<int64_t>> seen_source_to_targets;
  absl::flat_hash_map<int64_t, std::vector<int64_t>> seen_target_to_sources;
  int allowed_seen_count = 1;
  if (collective_permute->inplace()) {
    if (collective_permute->operand(0)->shape().IsArray()) {
      allowed_seen_count =
          collective_permute->operand(2)->shape().tuple_shapes().size();
    } else {
      allowed_seen_count = collective_permute->operand(2)
                               ->shape()
                               .tuple_shapes(0)
                               .tuple_shapes()
                               .size();
    }
  }

  for (const auto& p : collective_permute->source_target_pairs()) {
    TF_RET_CHECK(p.first >= 0)
        << "Source " << p.first
        << " in the instruction's source-target pair must be >= 0 : "
        << collective_permute->ToString();
    TF_RET_CHECK(limit == 1 || p.first < limit)
        << "Source " << p.first
        << " in the instruction's source-target pair must be < " << limit
        << " : " << collective_permute->ToString();
    if (seen_source_to_targets.contains(p.first) &&
        seen_source_to_targets[p.first].size() == allowed_seen_count) {
      if (allowed_seen_count == 1) {
        return Internal(
            "Source %d appears more than once in instruction's source-target "
            "pairs: %s",
            p.first, collective_permute->ToString());
      } else {
        return Internal(
            "Source %d appears more than %d times in instruction's "
            "source-target "
            "pairs: %s",
            p.first, allowed_seen_count, collective_permute->ToString());
      }
    } else {
      seen_source_to_targets[p.first].push_back(p.second);
    }
    TF_RET_CHECK(p.second >= 0)
        << "Target " << p.second
        << " in the instruction's source-target pair must be >= 0 : "
        << collective_permute->ToString();
    TF_RET_CHECK(limit == 1 || p.second < limit)
        << "Target " << p.second
        << " in the instruction's source-target pair must be < " << limit
        << " : " << collective_permute->ToString();
    if (seen_target_to_sources.contains(p.second) &&
        seen_target_to_sources[p.second].size() == allowed_seen_count) {
      if (allowed_seen_count == 1) {
        return Internal(
            "Target %d appears more than once in instruction's source-target "
            "pairs: %s",
            p.second, collective_permute->ToString());
      } else {
        return Internal(
            "Target %d appears more than %d times in instruction's "
            "source-target "
            "pairs: %s",
            p.second, allowed_seen_count, collective_permute->ToString());
      }
    } else {
      seen_target_to_sources[p.second].push_back(p.first);
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ShapeVerifier::HandleCollectiveBroadcast(HloInstruction* hlo) {
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(
      hlo, ShapeInference::InferCollectiveBroadcastShape(operand_shapes));
}

absl::Status ShapeVerifier::HandleCollectivePermute(HloInstruction* hlo) {
  HloCollectivePermuteInstruction* collective_permute =
      Cast<HloCollectivePermuteInstruction>(hlo);
  TF_RETURN_IF_ERROR(CheckInplaceCollectivePermute(collective_permute));
  TF_RETURN_IF_ERROR(CheckDuplicatedSourceOrTarget(collective_permute));
  std::vector<const Shape*> operand_shapes;
  absl::c_transform(
      collective_permute->operands(), std::back_inserter(operand_shapes),
      [](const HloInstruction* operand) { return &(operand->shape()); });
  return CheckShape(hlo, ShapeInference::InferCollectivePermuteShape(
                             operand_shapes, collective_permute->inplace()));
}

absl::Status ShapeVerifier::HandleCollectivePermuteStart(HloInstruction* hlo) {
  HloCollectivePermuteInstruction* collective_permute_start =
      Cast<HloCollectivePermuteInstruction>(hlo);

  TF_RETURN_IF_ERROR(CheckInplaceCollectivePermute(collective_permute_start));
  TF_RETURN_IF_ERROR(CheckDuplicatedSourceOrTarget(collective_permute_start));
  std::vector<const Shape*> operand_shapes;
  absl::c_transform(
      collective_permute_start->operands(), std::back_inserter(operand_shapes),
      [](const HloInstruction* operand) { return &(operand->shape()); });
  std::vector<Shape> context_shapes;
  if (collective_permute_start->shape().IsTuple() &&
      collective_permute_start->shape().tuple_shapes().size() > 2) {
    context_shapes = std::vector<Shape>(
        collective_permute_start->shape().tuple_shapes().begin() + 2,
        collective_permute_start->shape().tuple_shapes().end());
  }
  return CheckShape(
      collective_permute_start,
      ShapeInference::InferCollectivePermuteStartShape(
          operand_shapes, context_shapes, collective_permute_start->inplace()));
}

absl::Status ShapeVerifier::HandleCollectivePermuteDone(HloInstruction* hlo) {
  return CheckShape(hlo, ShapeInference::InferCollectivePermuteDoneShape(
                             hlo->operand(0)->shape()));
}

absl::Status ShapeVerifier::HandleReducePrecision(
    HloInstruction* reduce_precision) {
  return CheckShape(reduce_precision, ShapeInference::InferReducePrecisionShape(
                                          reduce_precision->operand(0)->shape(),
                                          reduce_precision->exponent_bits(),
                                          reduce_precision->mantissa_bits()));
}

absl::Status ShapeVerifier::CheckIsTokenOperand(
    const HloInstruction* instruction, int64_t operand_no) {
  const HloInstruction* token = instruction->operand(operand_no);
  if (!ShapeUtil::Equal(token->shape(), ShapeUtil::MakeTokenShape())) {
    return Internal(
        "Expected operand %d to be token-shaped, actual shape is "
        "%s:\n%s",
        operand_no, StringifyShape(token->shape()), instruction->ToString());
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::CheckOperandAndParameter(
    const HloInstruction* instruction, int64_t operand_number,
    const HloComputation* computation, int64_t parameter_number) {
  const HloInstruction* operand = instruction->operand(operand_number);
  const HloInstruction* parameter =
      computation->parameter_instruction(parameter_number);
  if (!ShapesSame(operand->shape(), parameter->shape())) {
    return Internal("Operand %s shape does not match parameter's %s in %s",
                    operand->ToString(), parameter->ToString(),
                    instruction->ToString());
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleInfeed(HloInstruction* instruction) {
  HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(instruction);
  TF_RETURN_IF_ERROR(CheckIsTokenOperand(instruction, 0));

  // The output of infeed is a tuple containing the data value and a token.
  return CheckShape(infeed,
                    ShapeUtil::MakeTupleShape(
                        {infeed->infeed_shape(), ShapeUtil::MakeTokenShape()}),
                    /*only_compare_minor_to_major_in_layout=*/true);
}

absl::Status ShapeVerifier::HandleOutfeed(HloInstruction* instruction) {
  HloOutfeedInstruction* outfeed = Cast<HloOutfeedInstruction>(instruction);
  TF_RETURN_IF_ERROR(CheckIsTokenOperand(instruction, 1));

  // Outfeed has a separate shape field for the value which is outfed to the
  // host. The shape of the instruction itself is always a token.
  if (!ShapesSame(outfeed->outfeed_shape(), outfeed->operand(0)->shape())) {
    return Internal(
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

absl::Status ShapeVerifier::HandleRng(HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(CheckOperandCount(instruction, 2));

  const Shape& shape_0 = instruction->operand(0)->shape();
  const Shape& shape_1 = instruction->operand(1)->shape();
  if (!ShapeUtil::IsScalar(shape_0) || !ShapeUtil::IsScalar(shape_1)) {
    return Internal(
        "Expected scalar types for the two operands of Rng instruction: %s",
        instruction->ToString());
  }

  if (!HasCompatibleElementTypes(shape_0, shape_1, instruction->shape())) {
    return Internal(
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
        return Internal(
            "Element type not supported."
            " Expected element to be of floating point type, integral type or"
            " predicate type for RngUniform: %s",
            instruction->ToString());
      }
      break;

    case RNG_NORMAL:
      if (!primitive_util::IsFloatingPointType(element_type)) {
        return Internal(
            "Element type not supported."
            " Expected element to be FloatingPointType for RngNormal: %s",
            instruction->ToString());
      }
      break;
    default:
      return Internal(
          "Invalid Rng distribution %s",
          RandomDistribution_Name(instruction->random_distribution()));
  }

  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleRngBitGenerator(HloInstruction* hlo) {
  if (!hlo->shape().IsTuple()) {
    return absl::OkStatus();
  }
  if (hlo->shape().IsTuple() && hlo->shape().tuple_shapes().size() != 2) {
    return Internal(
        "Expected tuple shape with 2 elements for RngBitGenerator. Got: %s",
        hlo->shape().ToString(true));
  }
  if (!ShapeUtil::Compatible(hlo->operand(0)->shape(),
                             hlo->shape().tuple_shapes(0))) {
    return Internal(
        "Expected state shape to match between input and output for "
        "RngBitGenerator. Got %s vs. %s",
        hlo->operand(0)->shape().ToString(true),
        hlo->shape().tuple_shapes(0).ToString());
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleRngGetAndUpdateState(
    HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(CheckOperandCount(instruction, 0));
  const Shape& result_shape = instruction->shape();
  const Shape expected_shape = ShapeUtil::MakeShape(U64, {2});
  if (!ShapeUtil::Compatible(result_shape, expected_shape)) {
    return Internal(
        "Invalid RngGetAndUpdateState, expect result to have shape %s, got %s ",
        StringifyShape(expected_shape), StringifyShape(result_shape));
  }

  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleReverse(HloInstruction* reverse) {
  return CheckShape(
      reverse, ShapeInference::InferReverseShape(reverse->operand(0)->shape(),
                                                 reverse->dimensions()));
}

absl::Status ShapeVerifier::HandleTopK(HloInstruction* hlo) {
  return CheckShape(
      hlo, ShapeInference::InferTopKShape(hlo->operand(0)->shape(),
                                          Cast<HloTopKInstruction>(hlo)->k()));
}

absl::Status ShapeVerifier::HandleSort(HloInstruction* hlo) {
  HloSortInstruction* sort = Cast<HloSortInstruction>(hlo);
  if (sort->operand_count() < 1) {
    return Internal("Expected at least 1 operand for %s instruction: %s",
                    HloOpcodeString(sort->opcode()), sort->ToString());
  }
  HloComputation* compare = sort->to_apply();

  // Check that the 'compare' computation returns a PRED.
  Shape compare_shape = compare->root_instruction()->shape();
  if (!ShapeUtil::Compatible(compare_shape, ShapeUtil::MakeShape(PRED, {}))) {
    return Internal(
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
      return Internal(
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
      return Internal(
          "Expected sort to have to have the same dimensions for all operands. "
          "First operand shape is: %s\n, shape (operand index %lld) is: %s",
          StringifyShape(sort->operand(0)->shape()), operand,
          StringifyShape(sort->operand(operand)->shape()));
    }
  }

  // Verify the sort_dimension.
  if (sort->sort_dimension() >=
      static_cast<int64_t>(sort->operand(0)->shape().dimensions().size())) {
    return Internal(
        "Expected the sort_dimension %d of sort to be smaller than the rank %d "
        "of the operand(s).",
        sort->sort_dimension(), sort->shape().dimensions().size());
  }

  return CheckVariadicShape(sort);
}

absl::Status ShapeVerifier::HandleConstant(HloInstruction* constant) {
  if (!Cast<HloConstantInstruction>(constant)->HasLiteral()) {
    return Internal("Constant is required to have a valid literal: %s",
                    constant->ToString());
  }
  return CheckShape(constant, constant->literal().shape(),
                    /*only_compare_minor_to_major_in_layout=*/true);
}

absl::Status ShapeVerifier::HandleIota(HloInstruction* hlo) {
  auto* iota = Cast<HloIotaInstruction>(hlo);
  if (!iota->shape().IsArray()) {
    return Internal("Iota does not support non-array result.");
  }
  const int64_t rank = iota->shape().dimensions().size();
  if (rank == 0) {
    return Internal("Iota does not support scalars.");
  }
  int64_t iota_dimension = iota->iota_dimension();
  if (iota_dimension >= rank || iota_dimension < 0) {
    return Internal(
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

  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  return CheckShape(get_tuple_element,
                    ShapeInference::InferGetTupleElementShape(
                        get_tuple_element->operand(0)->shape(),
                        get_tuple_element->tuple_index()));
}

namespace {
absl::Status SameElementTypesForOperandsAndToApplyParameters(
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
  return absl::OkStatus();
}
}  // namespace

absl::Status ShapeVerifier::HandleReduce(HloInstruction* reduce) {
  if (reduce->operand_count() % 2 != 0) {
    return Internal(
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
             ? absl::OkStatus()
             : SameElementTypesForOperandsAndToApplyParameters(
                   *reduce, reduce->operand_count());
}

absl::Status ShapeVerifier::HandleBitcast(HloInstruction* bitcast) {
  const Shape& output_shape = bitcast->shape();
  const Shape& operand_shape = bitcast->operand(0)->shape();
  if (opts_.layout_sensitive &&
      opts_.shape_size(output_shape) != opts_.shape_size(operand_shape)) {
    // Allow bitcast that has the same data size but different trailing
    // paddings.
    if (!opts_.allow_bitcast_to_have_different_size ||
        !(output_shape.is_static() && operand_shape.is_static() &&
          (ShapeUtil::ArrayDataSize(output_shape) ==
           ShapeUtil::ArrayDataSize(operand_shape)))) {
      return Internal(
          "%s: Bitcast cannot have different shape sizes of output (%d) and "
          "operand "
          "(%d) (%s) (%s)",
          bitcast->ToString(), opts_.shape_size(output_shape),
          opts_.shape_size(operand_shape), output_shape.ToString(true),
          operand_shape.ToString(true));
    }
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleBroadcast(HloInstruction* broadcast) {
  // HLO broadcast has no exact analog at the client level so there is no
  // ShapeInference method. Check the output shape explicitly.
  const Shape& operand_shape = broadcast->operand(0)->shape();
  // Check for mixed precision.
  TF_RET_CHECK(SameElementType(broadcast->shape(), operand_shape))
      << broadcast->ToString();
  TF_RET_CHECK(operand_shape.dimensions().size() ==
               broadcast->dimensions().size())
      << broadcast->ToString();
  for (int64_t operand_dimension = 0;
       operand_dimension < operand_shape.dimensions().size();
       ++operand_dimension) {
    int64_t output_dimension = broadcast->dimensions()[operand_dimension];
    TF_RET_CHECK(output_dimension >= 0 &&
                 output_dimension < broadcast->shape().dimensions().size() &&
                 (broadcast->shape().dimensions(output_dimension) ==
                  operand_shape.dimensions(operand_dimension)))
        << broadcast->ToString() << " operand shape " << operand_shape;
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleDynamicReshape(
    HloInstruction* dynamic_reshape) {
  // Check for mixed precision.
  const Shape& operand_shape = dynamic_reshape->operand(0)->shape();
  TF_RET_CHECK(SameElementType(dynamic_reshape->shape(), operand_shape));
  TF_RET_CHECK(ShapeUtil::ElementsIn(dynamic_reshape->shape()) ==
               ShapeUtil::ElementsIn(operand_shape));
  TF_RET_CHECK(dynamic_reshape->shape().dimensions().size() + 1 ==
               dynamic_reshape->operand_count());
  for (int64_t i = 1; i < dynamic_reshape->operand_count(); ++i) {
    TF_RET_CHECK(dynamic_reshape->operand(i)->shape().element_type() == S32);
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleReshape(HloInstruction* reshape) {
  // Check for mixed precision.
  const Shape& operand_shape = reshape->operand(0)->shape();
  TF_RET_CHECK(SameElementType(reshape->shape(), operand_shape));
  TF_RET_CHECK(ShapeUtil::ElementsIn(reshape->shape()) ==
               ShapeUtil::ElementsIn(operand_shape));
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleTranspose(HloInstruction* transpose) {
  return CheckShape(
      transpose, ShapeInference::InferTransposeShape(
                     transpose->operand(0)->shape(), transpose->dimensions()));
}

absl::Status ShapeVerifier::HandleParameter(HloInstruction* hlo) {
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleFusion(HloInstruction* fusion) {
  if (fusion->called_computations().size() != 1) {
    return Internal("Fusion has a non-unary number of called computations (%s)",
                    fusion->ToString().c_str());
  }
  const Shape& root_computation_shape =
      fusion->called_computations()[0]->root_instruction()->shape();
  if (!ShapesSame(fusion->shape(), root_computation_shape)) {
    return Internal(
        "Fused computation shape (%s) is not equal to the fusion shape (%s)",
        root_computation_shape.ToString(true), fusion->shape().ToString(true));
  }

  auto& fused_parameters = fusion->fused_parameters();
  if (fused_parameters.size() > fusion->operand_count()) {
    return Internal(
        "Fused parameter count (%d) is greater than the number of operands (%d)"
        " passed to the fusion instruction in: %s.",
        fused_parameters.size(), fusion->operand_count(),
        fusion->ToString().c_str());
  }
  for (HloInstruction* fused_param : fused_parameters) {
    int64_t param_no = fused_param->parameter_number();
    if (!ShapesSame(fused_param->shape(), fusion->operand(param_no)->shape())) {
      return Internal(
          "Shape mismatch between parameter number %d and its operand in "
          "%s.",
          param_no, fusion->ToString().c_str());
    }
  }
  const HloFusionInstruction* casted_fusion =
      DynCast<const HloFusionInstruction>(fusion);
  for (const auto& pair : casted_fusion->output_to_operand_aliasing()) {
    TF_RET_CHECK(pair.second.first < casted_fusion->operand_count())
        << "Invalid aliasing operand index.";
    TF_RET_CHECK(ShapeUtil::IndexIsValid(
        casted_fusion->operand(pair.second.first)->shape(), pair.second.second))
        << "Invalid aliasing operand shape index.";
    TF_RET_CHECK(ShapeUtil::IndexIsValid(casted_fusion->shape(), pair.first))
        << "Invalid aliasing output shape index.";
    const Shape& output_subshape =
        ShapeUtil::GetSubshape(casted_fusion->shape(), pair.first);
    const Shape& operand_subshape = ShapeUtil::GetSubshape(
        casted_fusion->operand(pair.second.first)->shape(), pair.second.second);
    if (opts_.layout_sensitive) {
      if (casted_fusion->IsFused()) {
        // Nested fusions can have aliasing that does not require the
        // tiling/memory space assignment to be the same in order to alias.
        TF_RET_CHECK(
            Shape::Equal().IgnoreTilesInLayout().IgnoreMemorySpaceInLayout()(
                operand_subshape, output_subshape))
            << "Different aliasing shapes: "
            << operand_subshape.ToString(/*print_layout=*/true) << " vs "
            << output_subshape.ToString(/*print_layout=*/true);
      } else {
        TF_RET_CHECK(Shape::Equal()(operand_subshape, output_subshape))
            << "Different aliasing shapes: "
            << operand_subshape.ToString(/*print_layout=*/true) << " vs "
            << output_subshape.ToString(/*print_layout=*/true);
      }
    } else {
      TF_RET_CHECK(ShapeUtil::Compatible(output_subshape, operand_subshape))
          << "Different aliasing shapes: "
          << operand_subshape.ToString(/*print_layout=*/true) << " vs "
          << output_subshape.ToString(/*print_layout=*/true);
    }
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleCall(HloInstruction* call) {
  TF_RETURN_IF_ERROR(
      CheckParameterCount(call, call->to_apply(), call->operand_count()));
  for (int64_t i = 0; i < call->to_apply()->num_parameters(); ++i) {
    TF_RETURN_IF_ERROR(CheckOperandAndParameter(call, i, call->to_apply(), i));
  }
  if (call->is_composite()) {
    TF_RET_CHECK(call->has_frontend_attributes())
        << "A composite call op must have frontend attributes";
    auto map = call->frontend_attributes().map();
    if (auto name = map.find("composite.name");
        name == map.end() || name->second.empty()) {
      return InvalidArgument(
          "A composite call op must have frontend attributes with key "
          "composite.name whose value is non-empty");
    }
    if (auto attributes = map.find("composite.attributes");
        attributes != map.end() && attributes->second.empty()) {
      return InvalidArgument(
          "A composite call op must have frontend attributes with key "
          "composite.attributes whose value is default: {} or non-empty");
    }
    if (auto version_str = map.find("composite.version");
        version_str != map.end()) {
      int64_t version = 0;
      if (!absl::SimpleAtoi(version_str->second, &version) || version < 0) {
        return InvalidArgument(
            "A composite call op must have frontend attributes with a "
            "composite.version whose value is a non-negative integer but got: "
            "%s",
            version_str->second);
      }
    }
  }
  // The shape of kCall should match the shape of the computation it calls.
  return CheckShape(call, call->to_apply()->root_instruction()->shape());
}

absl::Status ShapeVerifier::HandleCustomCall(HloInstruction* instruction) {
  const HloCustomCallInstruction* custom_call =
      DynCast<const HloCustomCallInstruction>(instruction);
  TF_RET_CHECK(custom_call != nullptr);
  if (custom_call->layout_constrained() &&
      !custom_call->IsCustomCall("LayoutConstraint")) {
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
          << custom_call->operand(i)->shape().ToString(true) << " operand "
          << operand_shape_with_layout.ToString();
      TF_RET_CHECK(LayoutUtil::HasLayout(operand_shape_with_layout));
    }
  }
  bool ignore_buffer = custom_call->IsCustomCall(kPinCustomCallTarget) ||
                       custom_call->IsCustomCall(kUnpinCustomCallTarget);
  for (const auto& pair : custom_call->output_to_operand_aliasing()) {
    TF_RET_CHECK(pair.second.first < custom_call->operand_count())
        << "Invalid aliasing operand index.";
    TF_RET_CHECK(ShapeUtil::IndexIsValid(
        custom_call->operand(pair.second.first)->shape(), pair.second.second))
        << "Invalid aliasing operand shape index.";
    TF_RET_CHECK(ShapeUtil::IndexIsValid(custom_call->shape(), pair.first))
        << "Invalid aliasing output shape index.";
    if (custom_call->frontend_attributes().map().contains(
            "xla_skip_custom_call_alias_shape_check")) {
      return absl::OkStatus();
    }
    const Shape& output_subshape =
        ShapeUtil::GetSubshape(custom_call->shape(), pair.first);
    const Shape& operand_subshape = ShapeUtil::GetSubshape(
        custom_call->operand(pair.second.first)->shape(), pair.second.second);
    if (opts_.layout_sensitive) {
      TF_RET_CHECK(Shape::Equal().IgnoreBuffer(ignore_buffer)(operand_subshape,
                                                              output_subshape))
          << "Different aliasing shapes: "
          << operand_subshape.ToString(/*print_layout=*/true) << " vs "
          << output_subshape.ToString(/*print_layout=*/true);
    } else {
      TF_RET_CHECK(
          Shape::Equal().IgnoreDynamicDimension().IgnoreLayout().IgnoreBuffer(
              ignore_buffer)(output_subshape, operand_subshape))
          << "Different aliasing shapes: "
          << operand_subshape.ToString(/*print_layout=*/true) << " vs "
          << output_subshape.ToString(/*print_layout=*/true);
    }
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleSlice(HloInstruction* slice) {
  return CheckShape(slice,
                    ShapeInference::InferSliceShape(
                        slice->operand(0)->shape(), slice->slice_starts(),
                        slice->slice_limits(), slice->slice_strides()));
}

absl::Status ShapeVerifier::HandleDynamicSlice(HloInstruction* dynamic_slice) {
  return CheckShape(
      dynamic_slice,
      ShapeInference::InferDynamicSliceShape(
          dynamic_slice->operand(0)->shape(),
          Cast<HloDynamicSliceInstruction>(dynamic_slice)->index_shapes(),
          dynamic_slice->dynamic_slice_sizes()));
}

absl::Status ShapeVerifier::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  return CheckShape(
      dynamic_update_slice,
      ShapeInference::InferDynamicUpdateSliceShape(
          dynamic_update_slice->operand(0)->shape(),
          dynamic_update_slice->operand(1)->shape(),
          Cast<HloDynamicUpdateSliceInstruction>(dynamic_update_slice)
              ->index_shapes()));
}

absl::Status ShapeVerifier::HandleTuple(HloInstruction* tuple) {
  return CheckVariadicShape(tuple);
}

absl::Status ShapeVerifier::HandleMap(HloInstruction* map) {
  std::vector<const Shape*> operand_shapes;
  int64_t max_operand_rank = 0;
  for (const HloInstruction* operand : map->operands()) {
    operand_shapes.push_back(&operand->shape());
    max_operand_rank =
        std::max(max_operand_rank,
                 static_cast<int64_t>(operand->shape().dimensions().size()));
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
             ? absl::OkStatus()
             : SameElementTypesForOperandsAndToApplyParameters(
                   *map, map->operand_count());
}

absl::Status ShapeVerifier::HandleReduceWindow(HloInstruction* reduce_window) {
  auto reduce_window_instr = Cast<HloReduceWindowInstruction>(reduce_window);
  auto input_shapes = reduce_window_instr->input_shapes();
  auto init_shapes = reduce_window_instr->init_value_shapes();
  TF_RETURN_IF_ERROR(CheckShape(
      reduce_window, ShapeInference::InferReduceWindowShape(
                         input_shapes, init_shapes, reduce_window->window(),
                         reduce_window->to_apply()->ComputeProgramShape())));

  return opts_.allow_mixed_precision
             ? absl::OkStatus()
             : SameElementTypesForOperandsAndToApplyParameters(
                   *reduce_window, reduce_window->operand_count());
}

absl::Status ShapeVerifier::HandleSelectAndScatter(
    HloInstruction* instruction) {
  return CheckShape(
      instruction,
      ShapeInference::InferSelectAndScatterShape(
          instruction->operand(0)->shape(),
          instruction->select()->ComputeProgramShape(), instruction->window(),
          instruction->operand(1)->shape(), instruction->operand(2)->shape(),
          instruction->scatter()->ComputeProgramShape()));
}

absl::Status ShapeVerifier::HandleWhile(HloInstruction* xla_while) {
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
    return Internal(
        "Conditional computation shape does not lead to a scalar predicate "
        "shape: %s",
        StringifyShape(conditional_shape));
  }
  // The shape of kWhile should match the shape of the body computation it
  // calls.
  return CheckShape(xla_while,
                    xla_while->while_body()->root_instruction()->shape());
}

absl::Status ShapeVerifier::HandleConditional(HloInstruction* conditional) {
  if (!ShapeUtil::IsScalar(conditional->operand(0)->shape())) {
    return InvalidArgument(
        "The first operand of conditional must be a scalar. Got %s",
        conditional->operand(0)->shape().ToString());
  }
  const int num_branches = conditional->branch_count();
  PrimitiveType operand0_type = conditional->operand(0)->shape().element_type();
  if (operand0_type == PRED) {
    TF_RET_CHECK(num_branches == 2);
  } else {
    if (operand0_type != S32) {
      return InvalidArgument(
          "The first operand of indexed conditional must be a scalar of S32. "
          "Got type %s.",
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
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandlePad(HloInstruction* pad) {
  return CheckShape(pad, ShapeInference::InferPadShape(pad->operand(0)->shape(),
                                                       pad->operand(1)->shape(),
                                                       pad->padding_config()));
}

namespace {
absl::Status CheckAsyncOpOperand(const HloInstruction* async_op) {
  const HloInstruction* operand = async_op->operand(0);
  if (operand->opcode() != HloOpcode::kAsyncStart &&
      operand->opcode() != HloOpcode::kAsyncUpdate) {
    return Internal(
        "%s expects operand to be async-update or async-done, found "
        "%s.",
        HloOpcodeString(async_op->opcode()),
        HloOpcodeString(operand->opcode()));
  }
  if (*async_op->async_wrapped_computation() !=
      *operand->async_wrapped_computation()) {
    return Internal(
        "The %s expects its wrapped async computation to be identical to its "
        "operand's wrapped async computation (%s vs %s), thread name (%s vs "
        "%s).",
        HloOpcodeString(async_op->opcode()),
        async_op->async_wrapped_instruction()->ToString(),
        operand->async_wrapped_instruction()->ToString(),
        async_op->async_wrapped_computation()->execution_thread(),
        operand->async_wrapped_computation()->execution_thread());
  }
  return absl::OkStatus();
}

absl::Status CheckAsyncOpComputationThreadName(const HloInstruction* async_op) {
  absl::string_view async_execution_thread = async_op->async_execution_thread();
  if (async_execution_thread !=
      async_op->async_wrapped_computation()->execution_thread()) {
    return Internal(
        "%s expects same async thread name as wrapped computation's "
        "thread name (%s vs %s).",
        HloOpcodeString(async_op->opcode()), async_execution_thread,
        async_op->async_wrapped_computation()->execution_thread());
  }
  return absl::OkStatus();
}

absl::Status CheckCallableInstructionThreadName(
    const HloInstruction* instruction) {
  for (const HloComputation* computation : instruction->called_computations()) {
    if (instruction->parent() != nullptr) {
      if (instruction->parent()->execution_thread() !=
          computation->execution_thread()) {
        return Internal(
            "callable instruction %s expects parent computation thread name "
            "same as called computation's thread name (%s vs %s).",
            instruction->ToString(), instruction->parent()->execution_thread(),
            computation->execution_thread());
      }
    }
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status ShapeVerifier::CheckAsyncOpComputationShapes(
    const HloInstruction* async_op, const Shape& async_shape) {
  if (!async_shape.IsTuple() || async_shape.tuple_shapes().size() < 2) {
    return Internal(
        "The %s expects the async shape to be a tuple of at least two "
        "elements, found %s.",
        HloOpcodeString(async_op->opcode()),
        async_shape.ToString(/*print_layout=*/true));
  }

  ProgramShape computation_shape =
      async_op->async_wrapped_computation()->ComputeProgramShape();
  Shape param_shape = ShapeUtil::MakeTupleShape(computation_shape.parameters());
  if (!ShapesSame(async_shape.tuple_shapes(0), param_shape)) {
    return Internal(
        "The %s expects the async shape at index {0} to match async "
        "computation parameter shape (%s vs %s).",
        HloOpcodeString(async_op->opcode()),
        async_shape.tuple_shapes(0).ToString(/*print_layout=*/true),
        param_shape.ToString(/*print_layout=*/true));
  }
  if (!ShapesSame(async_shape.tuple_shapes(1), computation_shape.result())) {
    return Internal(
        "The %s expects the async shape at index {1} to match the async "
        "computation root shape (%s vs %s).",
        HloOpcodeString(async_op->opcode()),
        async_shape.tuple_shapes(1).ToString(/*print_layout=*/true),
        computation_shape.result().ToString(/*print_layout=*/true));
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleAsyncStart(HloInstruction* async_start) {
  TF_RETURN_IF_ERROR(
      CheckAsyncOpComputationShapes(async_start, async_start->shape()));
  TF_RETURN_IF_ERROR(CheckAsyncOpComputationThreadName(async_start));
  const Shape& param_shape = async_start->shape().tuple_shapes(0);
  for (int i = 0; i < async_start->operand_count(); ++i) {
    if (!ShapesSame(param_shape.tuple_shapes(i),
                    async_start->operand(i)->shape())) {
      return Internal(
          "The %s expects the shape of operand %d to match the async shape at "
          "index {0} (%s vs %s).",
          HloOpcodeString(async_start->opcode()), i,
          async_start->operand(i)->shape().ToString(/*print_layout=*/true),
          param_shape.tuple_shapes(i).ToString(/*print_layout=*/true));
    }
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::HandleAsyncUpdate(HloInstruction* async_update) {
  TF_RETURN_IF_ERROR(CheckAsyncOpComputationThreadName(async_update));
  if (!ShapesSame(async_update->operand(0)->shape(), async_update->shape())) {
    return Internal(
        "The %s expects the shape of operand and output to match (%s vs %s).",
        HloOpcodeString(async_update->opcode()),
        async_update->operand(0)->shape().ToString(true),
        async_update->shape().ToString(true));
  }
  TF_RETURN_IF_ERROR(
      CheckAsyncOpComputationShapes(async_update, async_update->shape()));
  return CheckAsyncOpOperand(async_update);
}

absl::Status ShapeVerifier::HandleAsyncDone(HloInstruction* async_done) {
  TF_RETURN_IF_ERROR(CheckAsyncOpComputationThreadName(async_done));
  TF_RETURN_IF_ERROR(CheckAsyncOpComputationShapes(
      async_done, async_done->operand(0)->shape()));
  const Shape& root_shape = async_done->operand(0)->shape().tuple_shapes(1);
  if (!ShapesSame(root_shape, async_done->shape())) {
    return Internal(
        "The %s expects the shape of output to match the async shape at index "
        "{1} (%s vs %s).",
        HloOpcodeString(async_done->opcode()),
        async_done->shape().ToString(true), root_shape.ToString(true));
  }
  return CheckAsyncOpOperand(async_done);
}

absl::Status ShapeVerifier::HandleCopyStart(HloInstruction* copy_start) {
  return CheckShape(copy_start,
                    ShapeUtil::MakeTupleShape({copy_start->operand(0)->shape(),
                                               copy_start->operand(0)->shape(),
                                               ShapeUtil::MakeShape(U32, {})}),
                    /*only_compare_minor_to_major_in_layout=*/true);
}

absl::Status ShapeVerifier::HandleCopyDone(HloInstruction* copy_done) {
  const Shape& operand_shape = copy_done->operand(0)->shape();
  const Shape& dest_shape = ShapeUtil::GetTupleElementShape(operand_shape, 0);
  const Shape& src_shape = ShapeUtil::GetTupleElementShape(operand_shape, 1);
  if (!ShapesSame(dest_shape, src_shape,
                  Shape::Equal()
                      .IgnoreMemorySpaceInLayout()
                      .IgnoreSplitConfigInLayout())) {
    return Internal(
        "Source and destination buffers in CopyDone arguments need to be the "
        "same shape found %s and %s\n%s",
        StringifyShape(dest_shape), StringifyShape(src_shape),
        copy_done->ToString());
  }
  return CheckShape(copy_done, ShapeUtil::GetTupleElementShape(
                                   copy_done->operand(0)->shape(), 0));
}

absl::Status ShapeVerifier::HandleSend(HloInstruction* send) {
  return CheckShape(send,
                    ShapeUtil::MakeTupleShape({send->operand(0)->shape(),
                                               ShapeUtil::MakeShape(U32, {}),
                                               ShapeUtil::MakeTokenShape()}),
                    /*only_compare_minor_to_major_in_layout=*/true);
}

absl::Status ShapeVerifier::HandleSendDone(HloInstruction* send_done) {
  return CheckShape(send_done, ShapeUtil::MakeTokenShape());
}

absl::Status ShapeVerifier::HandleRecv(HloInstruction* recv) {
  return CheckShape(
      recv,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::GetTupleElementShape(recv->shape(), 0),
           ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeTokenShape()}),
      /*only_compare_minor_to_major_in_layout=*/true);
}

absl::Status ShapeVerifier::HandleRecvDone(HloInstruction* recv_done) {
  return CheckShape(
      recv_done,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::GetTupleElementShape(recv_done->operand(0)->shape(), 0),
           ShapeUtil::MakeTokenShape()}));
}

absl::Status ShapeVerifier::HandleBatchNormTraining(
    HloInstruction* batch_norm_training) {
  return CheckShape(batch_norm_training,
                    ShapeInference::InferBatchNormTrainingShape(
                        batch_norm_training->operand(0)->shape(),
                        batch_norm_training->operand(1)->shape(),
                        batch_norm_training->operand(2)->shape(),
                        batch_norm_training->feature_index()));
}

absl::Status ShapeVerifier::HandleBatchNormInference(
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

absl::Status ShapeVerifier::HandleBatchNormGrad(
    HloInstruction* batch_norm_grad) {
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
absl::Status CheckMixedPrecisionOperands(const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    // Allow-list the following opcodes for mixed-precision check, because
    // they involve data pass through or grouping via tuples, where the
    // precisions of buffers can be different.
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kConstant:
    case HloOpcode::kConvolution:
    case HloOpcode::kDot:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
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
            [&](const Shape& subshape,
                const ShapeIndex& index) -> absl::Status {
              if (!ShapeUtil::ElementIsFloating(subshape)) {
                return absl::OkStatus();
              }
              if (fp_type == PRIMITIVE_TYPE_INVALID) {
                fp_type = subshape.element_type();
              } else if (fp_type != subshape.element_type()) {
                return Internal(
                    "Seen floating point types of different precisions in "
                    "%s, but mixed precision is disallowed.",
                    instruction->ToString());
              }
              return absl::OkStatus();
            }));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ShapeVerifier::HandleGather(HloInstruction* gather) {
  return CheckShape(
      gather,
      ShapeInference::InferGatherShape(
          gather->operand(0)->shape(), gather->operand(1)->shape(),
          gather->gather_dimension_numbers(), gather->gather_slice_sizes()));
}

absl::Status ShapeVerifier::HandleScatter(HloInstruction* scatter) {
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

absl::Status ShapeVerifier::HandleAfterAll(HloInstruction* token) {
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : token->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(token, ShapeUtil::MakeTokenShape());
}

absl::Status ShapeVerifier::HandleAddDependency(
    HloInstruction* add_dependency) {
  TF_RETURN_IF_ERROR(CheckIsTokenOperand(add_dependency, 1));
  return CheckShape(add_dependency, add_dependency->operand(0)->shape());
}

absl::Status ShapeVerifier::HandleGetDimensionSize(HloInstruction* get_size) {
  return CheckShape(get_size,
                    ShapeInference::InferGetDimensionSizeShape(
                        get_size->operand(0)->shape(), get_size->dimension()));
}

absl::Status ShapeVerifier::HandleSetDimensionSize(HloInstruction* set_size) {
  return CheckShape(set_size,
                    ShapeInference::InferSetDimensionSizeShape(
                        set_size->operand(0)->shape(),
                        set_size->operand(1)->shape(), set_size->dimension()));
}

absl::Status ShapeVerifier::CheckShape(
    const HloInstruction* instruction, const Shape& inferred_shape,
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
      case HloOpcode::kWhile: {
        Shape::Equal equal;
        if (only_compare_minor_to_major_in_layout) {
          equal.MinorToMajorOnlyInLayout();
        }
        return ShapesSame(instruction->shape(), inferred_shape, equal);
      }
      case HloOpcode::kDynamicUpdateSlice: {
        Shape::Equal equal;
        if (only_compare_minor_to_major_in_layout) {
          equal.MinorToMajorOnlyInLayout();
        }
        if (instruction->parent()->IsFusionComputation()) {
          // For DynamicUpdateSlice it has an "in-place" update semantics, but
          // inside of fusions memory space propagation doesn't propagate the
          // memory spaces all the way, causing possible mismatches. Relax the
          // constraint in that condition. Tiling also is not necessarily
          // meaningful within fusions, so we can relax this as well.
          equal.IgnoreMemorySpaceInLayout().IgnoreTilesInLayout();
        }
        return ShapesSame(instruction->shape(), inferred_shape, equal);
      }
      case HloOpcode::kCopy: {
        // Disallow host offloading copies which change FpPrecision.
        if (opts_.IsLayoutSensitive()) {
          if (instruction->shape().has_layout() &&
              inferred_shape.has_layout()) {
            int64_t instruction_memory_space =
                instruction->shape().layout().memory_space();
            int64_t operand_memory_space =
                inferred_shape.layout().memory_space();
            if (instruction_memory_space != operand_memory_space &&
                (instruction_memory_space == Layout::kHostMemorySpace ||
                 operand_memory_space == Layout::kHostMemorySpace)) {
              if (instruction_memory_space == Layout::kHostMemorySpace) {
                // Unfortunately it might still be a host->host copy before
                // memory space is propagated. A transpose is allowed in that
                // case.
                return instruction->shape().element_type() ==
                       inferred_shape.element_type();
              }
              // A host->device copy or a device->host copy cannot do a
              // transpose.
              return Shape::Equal().IgnoreMemorySpaceInLayout()(
                  instruction->shape(), inferred_shape);
            }
          }
        }
        [[fallthrough]];
      }

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
    return Internal(
        "Expected instruction to have shape equal to %s, actual "
        "shape is %s:\n%s",
        StringifyShape(inferred_shape), StringifyShape(instruction->shape()),
        instruction->ToString());
  }
  return absl::OkStatus();
}

absl::Status ShapeVerifier::CheckShape(
    const HloInstruction* instruction,
    const absl::StatusOr<Shape>& inferred_shape_status) {
  if (!inferred_shape_status.ok()) {
    absl::Status s = inferred_shape_status.status();
    tsl::errors::AppendToMessage(&s, ", for instruction ",
                                 instruction->ToString());
    return s;
  }
  return CheckShape(instruction, inferred_shape_status.value());
}

absl::Status ShapeVerifier::CheckUnaryShape(const HloInstruction* instruction) {
  return CheckShape(instruction,
                    ShapeInference::InferUnaryOpShape(instruction->opcode(),
                                                      instruction->operand(0)));
}

absl::Status ShapeVerifier::CheckBinaryShape(
    const HloInstruction* instruction) {
  return CheckShape(
      instruction, ShapeInference::InferBinaryOpShape(instruction->opcode(),
                                                      instruction->operand(0),
                                                      instruction->operand(1)));
}

absl::Status ShapeVerifier::CheckTernaryShape(
    const HloInstruction* instruction) {
  return CheckShape(instruction,
                    ShapeInference::InferTernaryOpShape(
                        instruction->opcode(), instruction->operand(0),
                        instruction->operand(1), instruction->operand(2)));
}

absl::Status ShapeVerifier::CheckVariadicShape(
    const HloInstruction* instruction) {
  return CheckShape(instruction,
                    ShapeInference::InferVariadicOpShape(
                        instruction->opcode(), instruction->operands()));
}

absl::Status ShapeVerifier::VerifyEntryComputationLayout(
    const HloModule& module) {
  const HloComputation* computation = module.entry_computation();
  const auto& layout = module.entry_computation_layout();
  const ShapeLayout& result_layout = layout.result_layout();

  TF_RETURN_IF_ERROR(
      ShapeUtil::ValidateShapeWithOptionalLayout(result_layout.shape()));

  // TPU layout assignment doesn't set the tiles on entry_computation_layout, so
  // let's not check that.
  if (!ShapesSame(computation->root_instruction()->shape(),
                  result_layout.shape(),
                  Shape::Equal()
                      .IgnoreTilesInLayout()
                      .IgnoreTailPaddingAlignmentInElements()
                      .IgnoreMemorySpaceInLayout())) {
    return Internal(
        "Shape of the root instruction of entry computation (%s) should be "
        "compatible to one specified in module's entry computation layout (%s)",
        StringifyShape(computation->root_instruction()->shape()),
        StringifyShape(result_layout.shape()));
  }

  if (computation->num_parameters() != layout.parameter_count()) {
    return Internal(
        "Number of parameters in entry computation layout (%d) must be same "
        "as number of parameters of entry computation (%d)",
        layout.parameter_count(), computation->num_parameters());
  }

  for (int i = 0; i < computation->num_parameters(); ++i) {
    const HloInstruction* parameter = computation->parameter_instruction(i);
    TF_RETURN_IF_ERROR(
        ShapeUtil::ValidateShapeWithOptionalLayout(layout.parameter_shape(i)));
    // TPU layout assignment doesn't set the tiles on entry_computation_layout,
    // so let's not check that.
    if (!ShapesSame(parameter->shape(), layout.parameter_shape(i),
                    Shape::Equal()
                        .IgnoreTilesInLayout()
                        .IgnoreTailPaddingAlignmentInElements()
                        .IgnoreMemorySpaceInLayout())) {
      return Internal(
          "Shape of the entry computation parameter %d is %s should be "
          "compatible to the one specified in module's entry computation "
          "layout %s",
          i, StringifyShape(parameter->shape()),
          StringifyShape(layout.parameter_shape(i)));
    }
  }

  // If result is aliased with a parameter, entry computation layout must have
  // same shape, layout and memory space for them (for example we can't alias
  // parameter and result if they have different memory spaces).
  const auto& alias_config = module.input_output_alias_config();
  TF_RETURN_IF_ERROR(alias_config.ForEachAliasWithStatus(
      [&](ShapeIndex result_index,
          HloInputOutputAliasConfig::Alias alias) -> absl::Status {
        // We skip may-alias buffers as they do not force aliasing.
        if (!alias.must_alias()) {
          return absl::OkStatus();
        }

        const Shape& result_shape =
            ShapeUtil::GetSubshape(result_layout.shape(), result_index);
        const Shape& parameter_shape = ShapeUtil::GetSubshape(
            layout.parameter_layout(alias.parameter_number).shape(),
            alias.parameter_index);

        if (result_shape != parameter_shape) {
          return Internal(
              "Shape and memory space of the result at index %s (%s) "
              "must be the same as the shape and memory spaceof aliased "
              "parameter %d at index %s (%s)",
              result_index.ToString(), StringifyShape(result_shape),
              alias.parameter_number, alias.parameter_index.ToString(),
              StringifyShape(parameter_shape));
        }

        return absl::OkStatus();
      }));

  return absl::OkStatus();
}

std::string ComputationsToString(
    absl::Span<HloComputation* const> computations) {
  return absl::StrJoin(computations, ",",
                       [](std::string* s, const HloComputation* computation) {
                         absl::StrAppend(s, computation->name());
                       });
}

absl::Status VerifyInstructionNameUnchanged(const HloModule& module,
                                            const HloVerifierOpts& opts) {
  if (!opts.verify_instruction_name_unchanged) {
    return absl::OkStatus();
  }
  for (auto* comp : module.computations()) {
    for (auto* inst : comp->instructions()) {
      if (inst->metadata().scheduling_name().empty()) {
        continue;
      }
      // We do not enforce the invariant when the instruction has been cloned
      // explicitly via .clone or .remat suffix.
      if (inst->metadata().scheduling_name() != inst->name() &&
          (!absl::StrContains(inst->name(), ".remat") &&
           !absl::StrContains(inst->name(), ".clone"))) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Expected instruction name to remain the same. Was '",
            inst->metadata().scheduling_name(), "' is '", inst->name(), "'."));
      }
    }
  }
  return absl::OkStatus();
}

// Verifies various invariants about the structure of the HLO:
//
// (1) each instruction is non-null and has a non-null parent() set to the
// HloComputation which contains it.
//
// (2) each computation is non-null and has a non-null parent() set to the
// HloModule which contains it.
//
// (3) the operands of each instruction are non-null and are in the same
// computation as the instruction.
absl::Status VerifyHloStructure(HloModule* module) {
  for (const HloComputation* computation : module->computations()) {
    if (computation == nullptr) {
      return Internal("Computation in module %s is a null pointer",
                      module->name());
    }

    if (computation->parent() == nullptr) {
      return Internal("Computation %s has a null parent pointer",
                      computation->name());
    }
    if (computation->parent() != module) {
      return Internal("Computation %s parent() does not point to parent module",
                      computation->name());
    }

    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction == nullptr) {
        return Internal("Instruction in computation %s is a null pointer",
                        computation->name());
      }
      if (instruction->parent() == nullptr) {
        return Internal("Instruction %s has a null parent pointer",
                        instruction->name());
      }
      if (instruction->parent() != computation) {
        return Internal(
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
        if (operand == nullptr) {
          return Internal(
              "Operand %d (out of %d) of instruction: %s is a null pointer", i,
              instruction->operand_count(), instruction->name());
        }
        if (operand->parent() == nullptr) {
          return Internal(
              "Operand %d (out of %d) of instruction: %s has a null pointer "
              "parent",
              i, instruction->operand_count(), instruction->name());
        }
        if (operand->parent() != instruction->parent()) {
          return Internal(
              "Operand %d (%s) of instruction %s is in a different "
              "computation: %s vs %s",
              i, operand->name(), instruction->name(),
              operand->parent() ? operand->parent()->name() : "(null)",
              instruction->parent()->name());
        }
      }
    }
  }
  return absl::OkStatus();
}

namespace {

// Checks if the given two instructions share the same channel id.
absl::Status CheckSameChannel(const HloInstruction* instr1,
                              const HloInstruction* instr2) {
  if (instr1->channel_id() != instr2->channel_id()) {
    return Internal(
        "Expected to have the same channel id, actual channel ids are: %s "
        "(%d), %s (%d)",
        instr1->ToString(), instr1->channel_id().value_or(-1),
        instr2->ToString(), instr2->channel_id().value_or(-1));
  }
  return absl::OkStatus();
}

// Checks if the given two instructions have the same is_host_transfer attribute
// value. Instructions must be send/recv instructions or their 'done' variant.
absl::Status CheckSameIsHostTransfer(const HloInstruction* instr1,
                                     const HloInstruction* instr2) {
  const HloSendRecvInstruction* send_recv1 =
      DynCast<const HloSendRecvInstruction>(instr1);
  const HloSendRecvInstruction* send_recv2 =
      DynCast<const HloSendRecvInstruction>(instr2);
  TF_RET_CHECK(send_recv1 != nullptr);
  TF_RET_CHECK(send_recv2 != nullptr);
  if (send_recv1->is_host_transfer() != send_recv2->is_host_transfer()) {
    return Internal(
        "Expected instructions to have the same is-host-transfer property: "
        "%s, "
        "%s ",
        instr1->ToString(), instr2->ToString());
  }
  return absl::OkStatus();
}

absl::Status VerifySingleUser(
    const HloInstruction* instruction,
    const absl::flat_hash_set<HloOpcode>& expected_users) {
  TF_RET_CHECK(instruction->users().size() == 1)
      << "The " << instruction->opcode()
      << " instruction requires one consumer, found "
      << instruction->users().size();

  const HloInstruction* user = instruction->users().front();
  TF_RET_CHECK(expected_users.contains(user->opcode()))
      << "The consumer of a " << instruction->opcode()
      << " instruction needs to be one of ("
      << absl::StrJoin(expected_users, ", ",
                       [](std::string* out, HloOpcode opcode) {
                         absl::StrAppend(out, HloOpcodeString(opcode));
                       })
      << "), found " << user->opcode();
  return absl::OkStatus();
}

absl::Status VerifySingleOperand(
    const HloInstruction* instruction,
    const std::vector<HloOpcode>& expected_operands) {
  TF_RET_CHECK(instruction->operands().size() == 1)
      << "The " << instruction->opcode()
      << " instruction requires one consumer, found "
      << instruction->users().size();

  const HloInstruction* operand = instruction->operand(0);
  TF_RET_CHECK(absl::c_find(expected_operands, operand->opcode()) !=
               expected_operands.end())
      << "The operand of a " << instruction->opcode()
      << " instruction needs to be "
      << absl::StrJoin(expected_operands, " or ",
                       [](std::string* out, HloOpcode opcode) {
                         absl::StrAppend(out, HloOpcodeString(opcode));
                       })
      << ", found " << operand->opcode();
  return absl::OkStatus();
}

// Checks asynchronous instruction pairs.
absl::Status VerifyAsynchronousInstructionPairs(const HloModule& module) {
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
        case HloOpcode::kSend: {
          // If the instruction is kSend or kRecv, it can have no users if and
          // only if it is wrapped in an async call.
          if (instruction->IsRoot() &&
              instruction->parent()->IsAsyncComputation()) {
            break;
          }
          TF_RETURN_IF_ERROR(VerifySingleUser(
              instruction, {HloOpcode::kSendDone, HloOpcode::kTuple}));
          break;
        }
        case HloOpcode::kSendDone: {
          TF_RETURN_IF_ERROR(VerifySingleOperand(
              instruction, {HloOpcode::kSend, HloOpcode::kGetTupleElement}));
          break;
        }
        case HloOpcode::kRecv: {
          // If the instruction is kSend or kRecv, it can have no users if and
          // only if it is wrapped in an async call.
          if (instruction->IsRoot() &&
              instruction->parent()->IsAsyncComputation()) {
            break;
          }
          TF_RETURN_IF_ERROR(VerifySingleUser(
              instruction, {HloOpcode::kRecvDone, HloOpcode::kTuple}));
          break;
        }
        case HloOpcode::kRecvDone: {
          TF_RETURN_IF_ERROR(VerifySingleOperand(
              instruction, {HloOpcode::kRecv, HloOpcode::kGetTupleElement}));
          break;
        }
        default:
          break;
      }
    }
  }
  return absl::OkStatus();
}

// Checks that the asynchronous computation only has a root and parameter
// instructions.
absl::Status VerifyAsyncComputation(const HloComputation* async_computation) {
  if (!async_computation->CanExpandIntoSingleInstruction()) {
    return FailedPrecondition(
        "Asynchronous computation %s expected to contain only the root and "
        "parameter instructions.",
        async_computation->name());
  }
  return absl::OkStatus();
}

// Checks that AllReduce instructions in the module are either all layout
// constrained or all unconstrained.
absl::Status VerifyLayoutConstrainedAllReduce(const HloModule& module) {
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
  return absl::OkStatus();
}

// Verifies that leaf nodes in an original value contain values.
absl::Status VerifyOriginalValue(const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (auto original_value = instruction->original_value()) {
        // An original value is expected to have intermediate nodes that are
        // always nullopt and leaves with actual values.
        for (const auto& leaf : original_value->leaves()) {
          if (!leaf.second.has_value()) {
            return Internal(
                "Leaf nodes in an original value is expected to contain values."
                " Instruction: %s.",
                instruction->ToString());
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

// Checks various invariants of channel instructions (send/recv and
// collectives).
absl::Status VerifyChannels(const HloModule& module,
                            const HloVerifierOpts& opts) {
  // Send/recv instruction must have a unique user. If it is the corresponding
  // send-done/recv-done operation, channel IDs must match.
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      auto channel_instr = DynCast<HloChannelInstruction>(instruction);
      if (!channel_instr || !channel_instr->channel_id()) {
        continue;
      }

      switch (instruction->opcode()) {
        case HloOpcode::kSend: {
          // If the instruction is kSend or kRecv, it can have no users if and
          // only if it is wrapped in an async call.
          if (instruction->IsRoot() &&
              instruction->parent()->IsAsyncComputation()) {
            break;
          }
          TF_RET_CHECK(instruction->users().size() == 1);
          const HloInstruction* send_done = instruction->users().front();
          if (send_done->opcode() == HloOpcode::kSendDone) {
            TF_RETURN_IF_ERROR(CheckSameChannel(instruction, send_done));
            TF_RETURN_IF_ERROR(CheckSameIsHostTransfer(instruction, send_done));
          }
          break;
        }
        case HloOpcode::kRecv: {
          // If the instruction is kSend or kRecv, it can have no users if and
          // only if it is wrapped in an async call.
          if (instruction->IsRoot() &&
              instruction->parent()->IsAsyncComputation()) {
            break;
          }
          TF_RET_CHECK(instruction->users().size() == 1);
          const HloInstruction* recv_done = instruction->users().front();
          if (recv_done->opcode() == HloOpcode::kRecvDone) {
            TF_RETURN_IF_ERROR(CheckSameChannel(instruction, recv_done));
            TF_RETURN_IF_ERROR(CheckSameIsHostTransfer(instruction, recv_done));
          }
          break;
        }
        case HloOpcode::kSendDone:
        case HloOpcode::kRecvDone:
          TF_RET_CHECK(instruction->operands().size() == 1);
          break;
        default:
          break;
      }
    }
  }

  return absl::OkStatus();
}

// CHECKs various invariants of a fusion instruction.
absl::Status CheckFusionInstruction(HloInstruction* fusion) {
  // The parent fusion instruction of the fusion computation must be 'fusion'.
  HloComputation* fused_computation = fusion->fused_instructions_computation();
  if (fusion != fused_computation->FusionInstruction()) {
    return Internal(
        "Instruction of fused computation does not match expected "
        "instruction "
        "%s.",
        fusion->ToString());
  }

  // Fused root instruction and fused parameters must all be owned by the
  // fusion computation.
  bool root_owned = false;
  const auto& fused_parameters = fusion->fused_parameters();
  const HloInstruction* fused_root = fusion->fused_expression_root();
  std::vector<bool> parameter_owned(fused_parameters.size(), false);
  for (auto* instruction : fused_computation->instructions()) {
    if (fused_root == instruction) {
      if (root_owned) {
        return Internal("Root appears more than once in %s.",
                        fusion->ToString());
      }
      root_owned = true;
    }
    for (int i = 0; i < fused_parameters.size(); ++i) {
      if (fused_parameters[i] == instruction) {
        if (parameter_owned[i]) {
          return Internal("Parameter appears more than once in %s.",
                          fusion->ToString());
        }
        parameter_owned[i] = true;
      }
    }
  }
  if (!root_owned) {
    return Internal("Root not found in computation of %s.", fusion->ToString());
  }
  // Make sure all the parameter_owned entries are set
  for (int i = 0; i < parameter_owned.size(); i++) {
    if (!parameter_owned[i]) {
      return Internal("Parameter %d not found in computation of %s.", i,
                      fusion->ToString());
    }
  }

  // Fused root must have no users.
  if (fused_root->user_count() != 0) {
    return Internal("Root of %s may not have users.", fusion->ToString());
  }

  // All uses of fused instructions must be in the fusion computation, and
  // every non-root instruction must have at least one use.
  for (auto* instruction :
       fusion->fused_instructions_computation()->instructions()) {
    if (instruction != fused_root) {
      if (instruction->user_count() == 0 &&
          !instruction->HasSideEffectNoRecurse()) {
        return Internal("Non-root instruction %s in %s must have users.",
                        instruction->ToString(), fusion->ToString());
      }
      for (auto& user : instruction->users()) {
        if (fused_computation != user->parent()) {
          return Internal(
              "Non-root instruction %s in %s may not have external users.",
              instruction->ToString(), fusion->ToString());
        }
      }
    }
  }

  // Fused parameter instructions must be numbered contiguously and match up
  // (shapes equal) with their respective operand.
  CHECK_GE(fusion->operands().size(), fused_parameters.size());
  std::vector<bool> parameter_numbers(fused_parameters.size(), false);
  for (auto fused_param : fused_parameters) {
    int64_t param_no = fused_param->parameter_number();
    if (param_no < 0) {
      return Internal("Unexpected negative parameter number %d in %s.",
                      param_no, fusion->ToString());
    }
    if (param_no >= fused_parameters.size()) {
      return Internal(
          "Unexpected parameter number %d in %s: higher then number of "
          "parameters %lu.",
          param_no, fusion->ToString(), fused_parameters.size());
    }
    if (parameter_numbers[param_no]) {
      return Internal(
          "Did not expect parameter number %d more than once in %s.", param_no,
          fusion->ToString());
    }
    parameter_numbers[param_no] = true;
  }
  // Make sure all the parameter_numbers entries were seen.
  for (int i = 0; i < parameter_numbers.size(); i++) {
    if (!parameter_numbers[i]) {
      return Internal("Did not see parameter number %d in %s.", i,
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
  return absl::OkStatus();
}

// Checks that the operand shapes are compatible to the output shape, i.e.,
// that there are no implicit broadcasts.
absl::Status CheckElementwiseInstruction(HloInstruction* instruction) {
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
  return absl::OkStatus();
}

// Visitor which verifies various fields on the HLO instruction. This class does
// not check result shape as that is checked in the ShapeVerifier.
class InstructionVerifier : public DfsHloVisitorWithDefault {
 public:
  InstructionVerifier(const HloModule* module, const HloVerifierOpts& opts)
      : opts_(opts) {
    // TODO(b/258285553): Eliminate this check when all paths that enable SPMD
    // partitioning also set the num_partitions correctly.
    const int64_t num_partitions = module->config().num_partitions();
    if (module->config().use_spmd_partitioning() &&
        opts.verify_sharding_device_numbers && num_partitions > 1) {
      num_devices_ = module->config().num_partitions();
    }
  }

  absl::Status DefaultAction(HloInstruction*) override {
    return absl::OkStatus();
  }

  absl::Status HandleFusion(HloInstruction* fusion) override {
    TF_RETURN_IF_ERROR(CheckCallableInstructionThreadName(fusion));
    return CheckFusionInstruction(fusion);
  }

  absl::Status HandleBroadcast(HloInstruction* broadcast) override {
    // If you see this failure then someone has confused the difference
    // between the HLO broadcast op, and the UserComputation broadcast
    // op. See https://groups.google.com/forum/#!topic/xla-dev/9LqijHmTt_I
    // or ComputationLowerer::Visit()
    TF_RET_CHECK(broadcast->dimensions().size() ==
                 broadcast->operand(0)->shape().dimensions().size())
        << "Broadcast HLO (" << broadcast->ToShortString()
        << ") has invalid number of dimensions: "
        << broadcast->dimensions().size()
        << " != " << broadcast->operand(0)->shape().dimensions().size();
    if (opts_.verify_broadcast_dimensions_order) {
      TF_RET_CHECK(absl::c_is_sorted(broadcast->dimensions()))
          << "Broadcast dimensions should be ordered, got: "
          << broadcast->ToString();
    }
    return absl::OkStatus();
  }

  absl::Status HandleBitcastConvert(HloInstruction* c) override {
    // Shape verifier will check all we need.
    return absl::OkStatus();
  }

  absl::Status HandleWhile(HloInstruction* xla_while) override {
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
    // Allow kWhile to contain computations on separate thread.
    TF_RETURN_IF_ERROR(CheckCallableInstructionThreadName(xla_while));

    // Verify consistency of sharding of while instructions and related
    // instructions (parameters, root) in its called computations.
    TF_RETURN_IF_ERROR(VerifyConsistentSharding(
        xla_while, {xla_while, xla_while->while_body()->root_instruction(),
                    xla_while->while_body()->parameter_instruction(0),
                    xla_while->while_condition()->parameter_instruction(0)}));

    return absl::OkStatus();
  }

  absl::Status HandleCall(HloInstruction* call) override {
    if (opts_.verify_call_nested_computation_thread_name) {
      return CheckCallableInstructionThreadName(call);
    }

    // As opposed to other callable instructions, nothing respects input/output
    // aliasing for call instructions, so make sure it's not set.
    const HloCallableInstruction* callable =
        DynCast<const HloCallableInstruction>(call);
    TF_RET_CHECK(callable != nullptr);
    TF_RET_CHECK(callable->output_to_operand_aliasing().empty())
        << "Call instruction " << call->ToString()
        << " may not have an output-to-operand aliasing set.";
    return absl::OkStatus();
  }

  absl::Status HandleConditional(HloInstruction* conditional) override {
    const std::vector<HloComputation*> branch_computations =
        conditional->branch_computations();
    std::vector<const HloInstruction*> sharding_check_instructions;
    sharding_check_instructions.reserve(branch_computations.size() + 1);
    sharding_check_instructions.push_back(conditional);

    for (const HloComputation* branch_computation : branch_computations) {
      if (branch_computation->num_parameters() != 1) {
        return FailedPrecondition(
            "Branch computation %s of %s must have 1 parameter instead of %d",
            branch_computation->name(), conditional->ToString(),
            branch_computation->num_parameters());
      }
      sharding_check_instructions.push_back(
          branch_computation->root_instruction());
    }
    // Allow kConditional to contain computations on separate thread.
    TF_RETURN_IF_ERROR(CheckCallableInstructionThreadName(conditional));

    // Verify consistency of sharding of conditional instructions and roots of
    // its branches.
    TF_RETURN_IF_ERROR(
        VerifyConsistentSharding(conditional, sharding_check_instructions));

    return absl::OkStatus();
  }

  absl::Status HandleElementwiseUnary(HloInstruction* instruction) override {
    TF_RETURN_IF_ERROR(CheckUnaryOpWithResultAccuracy(instruction));
    return CheckElementwiseInstruction(instruction);
  }

  absl::Status HandleElementwiseBinary(HloInstruction* instruction) override {
    return CheckElementwiseInstruction(instruction);
  }

  absl::Status HandleGetTupleElement(HloInstruction* gte) override {
    TF_RET_CHECK(gte->operand(0)->shape().IsTuple());
    return absl::OkStatus();
  }

  absl::Status HandleTranspose(HloInstruction* transpose) override {
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
    return absl::OkStatus();
  }

  absl::Status HandleAllReduce(HloInstruction* crs) override {
    if (crs->channel_id().has_value()) {
      TF_RET_CHECK(crs->channel_id().value() > 0)
          << "All reduce channel id must be greater than 0 for "
          << crs->ToShortString();
    }
    return absl::OkStatus();
  }

  absl::Status HandleReshape(HloInstruction* hlo) override {
    if (opts_.verify_reshape_is_bitcast && !hlo->IsFused()) {
      TF_RET_CHECK(
          ShapeUtil::ReshapeIsBitcast(hlo->operand(0)->shape(), hlo->shape()))
          << "Reshape should be a physical bitcast, got: " << hlo->ToString();
    }
    return absl::OkStatus();
  }

  absl::Status HandleCustomCall(HloInstruction* hlo) override {
    if (opts_.verify_call_nested_computation_thread_name) {
      // Allow kCustomCall to contain computations on separate thread.
      return CheckCallableInstructionThreadName(hlo);
    }
    return absl::OkStatus();
  }

  absl::Status HandleScatter(HloInstruction* scatter) override {
    int64_t rank = scatter->operand(0)->shape().dimensions().size();
    for (int64_t operand_dim :
         scatter->scatter_dimension_numbers().scatter_dims_to_operand_dims()) {
      if (operand_dim > rank) {
        return absl::OutOfRangeError(absl::StrCat(
            "The provided scatter_dims_to_operand_dim was out of range.",
            " (operand_dim: ", operand_dim, ", rank: ", rank, ")"));
      }
    }
    return absl::OkStatus();
  }

  absl::Status Preprocess(HloInstruction* instruction) override {
    auto [it, inserted] =
        instructions_by_name_.emplace(instruction->name(), instruction);
    TF_RET_CHECK(inserted) << "HLO has name that is not unique within module:\n"
                           << instruction->ToString() << " in computation: "
                           << instruction->parent()->name()
                           << "\nPrevious HLO with same name:\n"
                           << it->second->ToString() << " in computation: "
                           << it->second->parent()->name();

    if (instruction->has_sharding()) {
      absl::Status status =
          instruction->sharding().Validate(instruction->shape(), num_devices_);
      if (!status.ok()) {
        return absl::Status(
            status.code(),
            absl::StrCat("Invalid sharding for instruction: ",
                         instruction->ToString(), ": ", status.message()));
      }
    }

    if (opts_.verify_call_nested_computation_thread_name &&
        instruction->has_to_apply() &&
        instruction->to_apply()->execution_thread() !=
            instruction->parent()->execution_thread()) {
      return Internal(
          "%s top_apply computation execution thread does not match (%s vs %s)",
          instruction->name(), instruction->to_apply()->execution_thread(),
          instruction->parent()->execution_thread());
    }

    return absl::OkStatus();
  }

  absl::Status Postprocess(HloInstruction* instruction) override {
    if (opts_.verify_no_host_memory_space) {
      TF_RETURN_IF_ERROR(VerifyNoHostMemorySpace(instruction));
    }
    if (!opts_.InstructionCanChangeLayout(instruction) &&
        instruction->shape().IsArray() && instruction->shape().has_layout()) {
      const Shape& result_shape = instruction->shape();
      const Layout& result_layout = result_shape.layout();
      for (HloInstruction* operand : instruction->operands()) {
        const Shape& operand_shape = operand->shape();
        if (operand_shape.IsArray() &&
            operand_shape.dimensions().size() ==
                result_shape.dimensions().size() &&
            operand_shape.has_layout()) {
          const Layout& operand_layout = operand_shape.layout();
          Layout::Equal equal_predicate =
              Layout::Equal().IgnoreTiles().IgnoreMemorySpace();
          if (instruction->opcode() == HloOpcode::kConvert ||
              instruction->opcode() == HloOpcode::kCompare ||
              instruction->opcode() == HloOpcode::kIsFinite ||
              (instruction->opcode() == HloOpcode::kSelect &&
               operand_shape.element_type() == PRED) ||
              instruction->opcode() == HloOpcode::kScatter) {
            // Some instructions can change element_size_in_bits
            // Select instructions ignore element_size_in_bits for predicate
            equal_predicate.IgnoreElementSize();
          } else if (instruction->opcode() == HloOpcode::kDynamicSlice ||
                     instruction->opcode() == HloOpcode::kDynamicUpdateSlice ||
                     instruction->opcode() == HloOpcode::kCopy) {
            TF_RETURN_IF_ERROR(HostOffloadInstructionCanChangeMemorySpace(
                instruction, operand_layout.memory_space(),
                result_layout.memory_space()));
            equal_predicate.IgnoreMemorySpace();
          }
          TF_RET_CHECK(equal_predicate(result_layout, operand_layout))
              << "Instruction shouldn't change layouts "
              << instruction->ToString() << " From " << result_shape << " To "
              << operand_shape;
        }
      }
    }
    return absl::OkStatus();
  }

 private:
  static absl::Status VerifyConsistentSharding(
      const HloInstruction* parent,
      absl::Span<const HloInstruction* const> instructions) {
    const HloInstruction* common_sharding_inst = nullptr;
    for (const HloInstruction* check_inst : instructions) {
      if (!check_inst->has_sharding()) {
        continue;
      }
      if (!common_sharding_inst) {
        common_sharding_inst = check_inst;
        continue;
      }
      TF_RET_CHECK(check_inst->sharding() == common_sharding_inst->sharding())
          << "Inconsistent " << parent->opcode()
          << " sharding among instructions: \n"
          << common_sharding_inst->ToString() << "\n"
          << check_inst->ToString();
    }
    return absl::OkStatus();
  }

  // Verifies whether a given `instruction` is permitted to change the layout
  // memory space from `operand_memory_space` to `result_memory_space`.
  // Returns absl::OkStatus() if the instruction's layout changes are valid;
  // otherwise, returns an appropriate error status.
  static absl::Status HostOffloadInstructionCanChangeMemorySpace(
      const HloInstruction* instruction, const int64_t operand_memory_space,
      const int64_t result_memory_space) {
    TF_RET_CHECK(!(operand_memory_space == Layout::kGenericFastMemorySpace &&
                   result_memory_space != Layout::kGenericFastMemorySpace) ||
                 (operand_memory_space != Layout::kGenericFastMemorySpace &&
                  result_memory_space == Layout::kGenericFastMemorySpace))
        << "Instruction shouldn't change layout memory space between generic "
           "fast memory space and others for instruction: "
        << instruction->ToString();

    if (instruction->opcode() == HloOpcode::kDynamicSlice) {
      TF_RET_CHECK(!(operand_memory_space == Layout::kDefaultMemorySpace &&
                     result_memory_space == Layout::kHostMemorySpace))
          << "DynamicSlice instruction shouldn't change layout memory "
          << "space from device to host: " << instruction->ToString();
    } else if (instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
      TF_RET_CHECK(!(operand_memory_space == Layout::kHostMemorySpace &&
                     result_memory_space == Layout::kDefaultMemorySpace))
          << "DynamicUpdateSlice instruction shouldn't change layout "
          << "memory space from host to device: " << instruction->ToString();
    } else if (instruction->opcode() != HloOpcode::kCopy) {
      return absl::InvalidArgumentError(
          absl::StrCat("Instruction shouldn't change layout memory space: ",
                       instruction->ToString()));
    }
    return absl::OkStatus();
  }

  // Returns an error status if an instruction or any operand contains host
  // memory space.
  static absl::Status VerifyNoHostMemorySpace(
      const HloInstruction* instruction) {
    return ShapeUtil::ForEachSubshapeWithStatus(
        instruction->shape(),
        [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
          if (subshape.has_layout()) {
            const Layout& result_layout = subshape.layout();
            if (result_layout.memory_space() == Layout::kHostMemorySpace) {
              return absl::InternalError(absl::StrCat(
                  "Instruction shouldn't have the layout of host memory "
                  "space: ",
                  instruction->ToString()));
            }
          }
          return absl::OkStatus();
        });
  }

  absl::flat_hash_map<std::string, const HloInstruction*> instructions_by_name_;
  const HloVerifierOpts& opts_;
  std::optional<int64_t> num_devices_;
};

bool IsCollectivesGroupComputation(HloComputation* computation) {
  auto maybe_caller = computation->GetUniqueCaller(HloOpcode::kAsyncStart);
  if (!maybe_caller.has_value()) {
    return false;
  }
  return (*maybe_caller)
      ->get_frontend_attribute(kCollectivesGroupAttr)
      .has_value();
}

int64_t CountWriters(const HloInstruction* inst,
                     absl::Span<const int64_t> shape_index);

// Returns the number of writers for the value produced by the instruction in
// the given shape_index and used by the given user.
//
// An example of a buffer with multiple writers:
//    b1 = b(f32[32]) custom-call(b0),
//      custom_call_target="foo",
//      output_to_operand_aliasing={{}: (0, {})}
//    call1 = b(f32[32]) custom-call(b1),
//      custom_call_target="writer_1",
//      output_to_operand_aliasing={{}: (0, {})},
//    call2 = b(f32[32]) custom-call(b1),
//      custom_call_target="writer_2",
//      output_to_operand_aliasing={{}: (0, {})},
int64_t CountWritersInUser(const HloInstruction* inst,
                           absl::Span<const int64_t> shape_index,
                           const HloInstruction* user) {
  if (dynamic_cast<const HloCallableInstruction*>(user) ||
      user->opcode() == HloOpcode::kWhile ||
      user->opcode() == HloOpcode::kConditional) {
    // For HloCallableInstruction, we may overcount here if we will allow
    // a buffer operand not in results.
    //
    // For other case, Without interprocedural analysis, we assume if a buffer
    // is passed into a while loop, it is written there.
    return 1;
  }
  if (user->opcode() == HloOpcode::kGetTupleElement &&
      user->tuple_index() == shape_index[0]) {
    return CountWriters(user, shape_index.subspan(1));
  }
  if (user->opcode() == HloOpcode::kTuple) {
    if (inst->parent()->root_instruction() == user) {
      // We assume if a buffer is passed into a while-body root, it will be
      // written to.
      if (!inst->parent()->caller_instructions(HloOpcode::kWhile).empty()) {
        return 1;
      }
    } else {
      std::vector<int64_t> new_shape_index;
      new_shape_index.reserve(shape_index.size() + 1);
      new_shape_index.push_back(user->operand_index(inst));
      new_shape_index.insert(new_shape_index.end(), shape_index.begin(),
                             shape_index.end());
      return CountWriters(user, new_shape_index);
    }
  }

  return 0;
}

// Returns the number of writers for the value produced by the instruction in
// the given shape_index. This is to support the verification that a buffer can
// have at most one writer and we may return early when we find more than one
// writers, but we choose not to return early.
int64_t CountWriters(const HloInstruction* inst,
                     absl::Span<const int64_t> shape_index) {
  int64_t num_writers = 0;
  for (const HloInstruction* user : inst->users()) {
    num_writers += CountWritersInUser(inst, shape_index, user);
  }

  return num_writers;
}

// Verifies a buffer result produced by the given instruction can have at most
// one writer.
absl::Status CheckBufferHasUniqueWriter(const HloInstruction* inst,
                                        const ShapeIndex& result_index) {
  if (CountWriters(inst, result_index) > 1) {
    return InvalidArgument(
        "an HLO buffer value has more than one writers (or unpins): '%s' '%s'",
        inst->ToString(), result_index.ToString());
  }

  return absl::OkStatus();
}

// Verifies all buffer results produced by the given instruction can have at
// most one writer.
absl::Status CheckBufferHasUniqueWriters(const HloInstruction* inst) {
  return ShapeUtil::ForEachSubshapeWithStatus(
      inst->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsBuffer()) {
          TF_RETURN_IF_ERROR(CheckBufferHasUniqueWriter(inst, index));
        }
        return absl::OkStatus();
      });
}

absl::Status VerifyPin(const HloCustomCallInstruction* inst) {
  if (inst->operand_count() != 1 || !inst->operand(0)->shape().IsArray() ||
      inst->operand(0)->shape().IsBuffer()) {
    return InvalidArgument(
        "custom-call to Pin must have one array non-buffer operand");
  }

  if (!inst->shape().IsBuffer()) {
    return InvalidArgument("custom-call to Pin must have one buffer result");
  }

  if (!xla::Shape::Equal()(inst->operand(0)->shape(),
                           inst->shape().buffer_shape())) {
    return InvalidArgument(
        "custom-call to Pin must have the same shape as the operand");
  }

  if (inst->output_to_operand_aliasing().size() != 1) {
    return InvalidArgument(
        "custom-call to Pin must have one output-to-operand aliasing");
  }

  return CheckBufferHasUniqueWriter(inst, {});
}

absl::Status VerifyCreateBuffer(const HloInstruction* inst) {
  if (inst->operand_count() != 0) {
    return InvalidArgument("custom-call to CreateBuffer can't have an operand");
  }

  if (!inst->shape().IsBuffer()) {
    return InvalidArgument(
        "custom-call to CreateBuffer must have one buffer result");
  }

  return CheckBufferHasUniqueWriter(inst, {});
}

absl::Status VerifyUnpin(const HloCustomCallInstruction* inst) {
  if (inst->operand_count() != 1 || !inst->operand(0)->shape().IsBuffer()) {
    return InvalidArgument("custom-call to Unpin must have one buffer operand");
  }

  if (!inst->shape().IsArray() || inst->shape().IsBuffer()) {
    return InvalidArgument(
        "custom-call to Unpin must have one array non-buffer result");
  }

  if (!xla::Shape::Equal()(inst->operand(0)->shape().buffer_shape(),
                           inst->shape())) {
    return InvalidArgument(
        "custom-call to Unpin must have the same shape as the operand");
  }

  if (inst->output_to_operand_aliasing().size() != 1) {
    return InvalidArgument(
        "custom-call to Unpin must have one output-to-operand aliasing");
  }

  return absl::OkStatus();
}

absl::Status VerifyNoBuffers(const Shape& shape, const HloInstruction* inst) {
  return ShapeUtil::ForEachSubshapeWithStatus(
      shape,
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsBuffer()) {
          return InvalidArgument(
              "Seen buffers while buffers aren't allowed "
              "in this context: %s",
              inst->ToString());
        }
        return absl::OkStatus();
      });
}

// Verifies that an operand with a buffer type should be mentioned in one pair
// of output-to-operand-aliasing.
absl::Status VerifyBuffersInOperands(const HloCustomCallInstruction* inst) {
  // Collect the operand parts that are mentioned in the output-to-operand
  // aliasing, and the number of times they are mentioned.
  absl::flat_hash_map<std::pair<int64_t, ShapeIndex>, int32_t>
      aliasing_part_to_count;
  for (const auto& pair : inst->output_to_operand_aliasing()) {
    if (aliasing_part_to_count.contains(pair.second)) {
      aliasing_part_to_count[pair.second]++;
    } else {
      aliasing_part_to_count[pair.second] = 1;
    }
  }

  int64_t operand_index = 0;
  for (auto* operand : inst->operands()) {
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        operand->shape(),
        [&](const Shape& subshape,
            const ShapeIndex& shape_index) -> absl::Status {
          if (!subshape.IsBuffer()) {
            return absl::OkStatus();
          }
          std::pair<int64_t, ShapeIndex> operand_part =
              std::make_pair(operand_index, shape_index);
          if (!aliasing_part_to_count.contains(operand_part)) {
            return InvalidArgument(
                "buffer is used in operands but not in results: operand %d "
                "ShapeIndex %s",
                operand_index, shape_index.ToString());
          }
          // The operand aliases with multiple results.
          if (aliasing_part_to_count[operand_part] > 1) {
            return InvalidArgument(
                "buffer is used in results multiple times: operand %d "
                "ShapeIndex "
                "%s",
                operand_index, shape_index.ToString());
          }
          return absl::OkStatus();
        }));
    operand_index++;
  }

  return absl::OkStatus();
}

// Verifies that a result with a buffer type should be mentioned in one pair
// of output-to-operand-aliasing, and returns the ShapeIndex for the buffers
// in the results.
absl::Status VerifyBuffersInResults(
    const HloCustomCallInstruction* inst,
    absl::flat_hash_set<ShapeIndex>& buffer_results) {
  // Collect the results that are mentioned in the output-to-operand aliasing,
  // and the number of times they are mentioned.
  absl::flat_hash_map<ShapeIndex, int32_t> aliasing_part_to_count;
  for (const auto& pair : inst->output_to_operand_aliasing()) {
    if (aliasing_part_to_count.contains(pair.first)) {
      aliasing_part_to_count[pair.first]++;
    } else {
      aliasing_part_to_count[pair.first] = 1;
    }
  }

  return ShapeUtil::ForEachSubshapeWithStatus(
      inst->shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
        if (subshape.IsBuffer()) {
          if (!aliasing_part_to_count.contains(index)) {
            return InvalidArgument(
                "buffer is used in results but not in operands: %s",
                index.ToString());
          }
          // The result aliases with multiple operands.
          if (aliasing_part_to_count[index] > 1) {
            return InvalidArgument(
                "buffer is used in operands multiple times: %s",
                index.ToString());
          }
          buffer_results.insert(index);
        }
        return absl::OkStatus();
      });
}

// Verifies pin/unpin related custom-calls as well as general custom-calls that
// may use buffers.
//
// For general custom-calls, before we reach here, we already verify that an
// alias pair of operand and result should be both buffers or both non-buffers.
// We further verify the following:
// - An operand or result with a buffer type should be mentioned in one pair of
//   output-to-operand-aliasing.
// - An HLO buffer result can only be updated at most once.
//
absl::Status VerifyCustomCall(const HloCustomCallInstruction* inst) {
  if (inst->IsCustomCall(kPinCustomCallTarget)) {
    return VerifyPin(Cast<HloCustomCallInstruction>(inst));
  }
  if (inst->IsCustomCall(kCreateBufferCustomCallTarget)) {
    return VerifyCreateBuffer(inst);
  }
  if (inst->IsCustomCall(kUnpinCustomCallTarget)) {
    return VerifyUnpin(Cast<HloCustomCallInstruction>(inst));
  }

  TF_RETURN_IF_ERROR(VerifyBuffersInOperands(inst));
  // Record the ShapeIndex for the buffers in the results.
  absl::flat_hash_set<ShapeIndex> buffer_results;
  TF_RETURN_IF_ERROR(VerifyBuffersInResults(inst, buffer_results));

  // Ensure that an SSA buffer result can have at most one writer.
  for (const auto& result_index : buffer_results) {
    TF_RETURN_IF_ERROR(CheckBufferHasUniqueWriter(inst, result_index));
  }

  return absl::OkStatus();
}

absl::Status VerifyNoBuffersInContext(const HloInstruction* inst) {
  TF_RETURN_IF_ERROR(VerifyNoBuffers(inst->shape(), inst));
  for (auto* operand : inst->operands()) {
    TF_RETURN_IF_ERROR(VerifyNoBuffers(operand->shape(), inst));
  }
  return absl::OkStatus();
}

absl::Status VerifyBuffers(const HloModule& module) {
  for (auto* comp : module.computations()) {
    for (auto* inst : comp->instructions()) {
      if (inst->opcode() == HloOpcode::kCustomCall) {
        TF_RETURN_IF_ERROR(
            VerifyCustomCall(Cast<HloCustomCallInstruction>(inst)));
      } else if (inst->opcode() == HloOpcode::kWhile) {
        TF_RETURN_IF_ERROR(CheckBufferHasUniqueWriters(inst));
      } else if (inst->opcode() == HloOpcode::kParameter) {
        if (comp->IsEntryComputation()) {
          TF_RETURN_IF_ERROR(VerifyNoBuffersInContext(inst));
        }
        TF_RETURN_IF_ERROR(CheckBufferHasUniqueWriters(inst));
      } else if (inst->opcode() != HloOpcode::kGetTupleElement &&
                 inst->opcode() != HloOpcode::kTuple) {
        TF_RETURN_IF_ERROR(VerifyNoBuffersInContext(inst));
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> HloVerifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto disabled = module->config().debug_options().xla_disable_hlo_passes();
  if (std::find(disabled.begin(), disabled.end(), name()) != disabled.end()) {
    return false;
  }
  auto status_or_changed = [&]() -> absl::StatusOr<bool> {
    TF_RET_CHECK(!module->name().empty());

    if (module->entry_computation()->IsFusionComputation()) {
      return InvalidArgument(
          "Module entry computation cannot be a fusion computation");
    }

    TF_RETURN_IF_ERROR(VerifyHloStructure(module));
    TF_RETURN_IF_ERROR(VerifyAsynchronousInstructionPairs(*module));
    TF_RETURN_IF_ERROR(
        VerifyChannels(*module, target_metadata_->GetVerifierOpts()));
    TF_RETURN_IF_ERROR(VerifyInstructionNameUnchanged(
        *module, target_metadata_->GetVerifierOpts()));

    std::unique_ptr<ShapeVerifier> shape_verifier =
        target_metadata_->GetVerifier();
    InstructionVerifier instruction_verifier(
        module, target_metadata_->GetVerifierOpts());
    for (auto* computation : module->computations(execution_threads)) {
      TF_RETURN_IF_ERROR(computation->Accept(shape_verifier.get()));
      TF_RETURN_IF_ERROR(computation->Accept(&instruction_verifier));
      // Verify that async computations contain a single instruction or a
      // collection of send/recv instructions. This is needed to represent NCCL
      // groups on GPU.
      if (computation->IsAsyncComputation() &&
          !computation->OnlyContainsSendRecv() &&
          !IsCollectivesGroupComputation(computation)) {
        TF_RETURN_IF_ERROR(VerifyAsyncComputation(computation));
      }
    }

    TF_RETURN_IF_ERROR(VerifyBuffers(*module));

    TF_RETURN_IF_ERROR(shape_verifier->VerifyEntryComputationLayout(*module));

    // If the module has a schedule, it must be valid.
    if (module->has_schedule()) {
      TF_RETURN_IF_ERROR(module->schedule().Verify());
    }

    if (HloInstruction::IsThreadIncluded(
            module->entry_computation()->execution_thread(),
            execution_threads)) {
      TF_RETURN_IF_ERROR(module->input_output_alias_config().Verify(
          *module, [this](const Shape& shape) -> int64_t {
            if (target_metadata_->GetVerifierOpts().IsLayoutSensitive()) {
              return target_metadata_->GetVerifierOpts().ShapeSize(shape);
            } else {
              return 0;
            }
          }));
    }

    TF_RETURN_IF_ERROR(module->buffer_donor_config().Verify(*module));
    TF_RETURN_IF_ERROR(VerifyLayoutConstrainedAllReduce(*module));
    TF_RETURN_IF_ERROR(VerifyOriginalValue(*module));
    return false;
  }();
  if (status_or_changed.ok()) {
    return status_or_changed.value();
  }
  return absl::Status(status_or_changed.status().code(),
                      absl::StrCat("during context [", context_, "]: ",
                                   status_or_changed.status().message()));
}

MetadataTracker::MetadataTracker(absl::string_view prefix) : prefix_(prefix) {}

MetadataTracker::~MetadataTracker() {
  if (instruction_count_ == 0) {
    return;
  }
  const std::map<std::string, double> values = {
      {"instruction_count", 1.0 * instruction_count_},
      {"op_type_coverage", 1.0 * has_op_type_count_ / instruction_count_},
      {"op_name_coverage", 1.0 * has_op_name_count_ / instruction_count_},
      {"source_file_coverage",
       1.0 * has_source_file_count_ / instruction_count_},
      {"dummy_source_file_coverage",
       1.0 * has_dummy_source_file_count_ / instruction_count_},
      {"source_line_coverage",
       1.0 * has_source_line_count_ / instruction_count_},
      {"creation_pass_coverage",
       1.0 * has_creation_pass_id_count_ / instruction_count_},
      {"logical_creation_pass_coverage",
       1.0 * has_logical_creation_pass_id_count_ / instruction_count_},
      {"size_of_generated_code_in_bytes_coverage",
       1.0 * has_size_of_generated_code_in_bytes_count_ / instruction_count_},
      {"size_of_memory_working_set_in_bytes_coverage",
       1.0 * has_size_of_memory_working_set_in_bytes_count_ /
           instruction_count_},
      {"profile_info_coverage",
       1.0 * has_profile_info_count_ / instruction_count_}};
  LOG(INFO) << prefix_ << " "
            << absl::StrJoin(values, ",", absl::PairFormatter("="));
}

void MetadataTracker::HandleMetadata(const OpMetadata& metadata) {
  ++instruction_count_;
  if (!metadata.op_type().empty()) {
    ++has_op_type_count_;
  }
  if (!metadata.op_name().empty()) {
    ++has_op_name_count_;
  }
  if (!metadata.source_file().empty()) {
    ++has_source_file_count_;
    if (absl::StrContains(metadata.source_file(), "dummy")) {
      ++has_dummy_source_file_count_;
    }
  }
  if (metadata.source_line() != 0) {
    ++has_source_line_count_;
  }
  if (metadata.size_of_generated_code_in_bytes() != 0) {
    ++has_size_of_generated_code_in_bytes_count_;
  }
  if (metadata.size_of_memory_working_set_in_bytes() != 0) {
    ++has_size_of_memory_working_set_in_bytes_count_;
  }
  if (metadata.has_profile_info()) {
    ++has_profile_info_count_;
  }
}

absl::Status MetadataTracker::DefaultAction(HloInstruction* instruction) {
  HandleMetadata(instruction->metadata());
  return absl::OkStatus();
}

}  // namespace xla
