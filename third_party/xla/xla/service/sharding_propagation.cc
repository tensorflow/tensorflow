/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/sharding_propagation.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/hlo_sharding_metadata.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/protobuf_util.h"
#include "xla/service/dot_as_convolution_util.h"
#include "xla/service/host_memory_offload_annotations.h"
#include "xla/service/spmd/shard_barrier_partitioner.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/sharding_op_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Returns true iff the specified hlo or sharding has a spatially partitioned
// sharding (tiled or replicated) that can be propagated by sharding
// propagation.
bool IsSpatiallyPartitioned(const HloSharding& sharding) {
  if (sharding.IsTuple()) {
    return absl::c_any_of(sharding.tuple_elements(), IsSpatiallyPartitioned);
  } else {
    return !sharding.IsTileMaximal() || sharding.IsReplicated();
  }
}
bool IsSpatiallyPartitioned(const HloInstruction* hlo) {
  return hlo->has_sharding() && IsSpatiallyPartitioned(hlo->sharding());
}

// Returns
// - 1, iff `lhs` is strictly better than `rhs`.
// - 2, iff `rhs` is strictly better than `lhs`.
// - 0 or 3, otherwise.
//
// Notes:
// - We think manual shardings are strictly better than tile maximal shardings.
// - For tuples we consider lhs to have a better sharding if none of the
//   elements are worse and at least one element is better then in rhs
//   sharding.
int MaskTupleShardingStrictlyBetter(const HloSharding& lhs,
                                    const HloSharding& rhs) {
  DCHECK(lhs.IsTuple());
  DCHECK(rhs.IsTuple());
  const auto& lhs_shardings = lhs.tuple_elements();
  const auto& rhs_shardings = rhs.tuple_elements();
  CHECK_EQ(lhs_shardings.size(), rhs_shardings.size());
  int mask = 0;
  for (int64_t i = 0; i < lhs_shardings.size(); ++i) {
    const auto& lhs_shard = lhs_shardings[i];
    const auto& rhs_shard = rhs_shardings[i];
    CHECK_EQ(lhs_shard.IsTuple(), rhs_shard.IsTuple());
    if (lhs_shard.IsTuple()) {
      mask |= MaskTupleShardingStrictlyBetter(lhs_shard, rhs_shard);
    } else {
      if (lhs_shard.IsManualLeaf() && rhs_shard.IsTileMaximalLeaf()) {
        mask |= 1;
      }
      if (rhs_shard.IsManualLeaf() && lhs_shard.IsTileMaximalLeaf()) {
        mask |= 2;
      }
    }
    if (mask == 3) break;
  }
  return mask;
}

bool IsShardingStrictlyBetter(const HloSharding& lhs, const HloSharding& rhs) {
  CHECK_EQ(lhs.IsTuple(), rhs.IsTuple()) << lhs << " <> " << rhs;
  if (lhs.IsTuple()) {
    return MaskTupleShardingStrictlyBetter(lhs, rhs) == 1;
  }
  return lhs.IsManualLeaf() && rhs.IsTileMaximalLeaf();
}

// Implementation for returning a improved sharding from another sharding.
std::optional<HloSharding> ReturnImprovedShardingImpl(
    HloSharding from, const HloSharding* to_improved,
    const Shape& to_improved_shape, bool may_combine_partial_sharding,
    bool allow_aggressive_resharding = false) {
  // Always allow improve the sharding if it's straightly better.
  if (to_improved != nullptr && IsShardingStrictlyBetter(from, *to_improved)) {
    return std::move(from);
  }
  // We don't want to propagate tile maximal shardings.
  if (!IsSpatiallyPartitioned(from)) {
    return std::nullopt;
  }
  // Any sharding is better than no sharding.
  if (to_improved == nullptr) {
    return std::move(from);
  }
  // We don't want to propagate manual shardings.
  if (from.IsManual()) {
    return std::nullopt;
  }
  int64_t sharding_tiles = from.NumTiles();
  if (hlo_sharding_util::MergeSharding(*to_improved, &from,
                                       may_combine_partial_sharding)) {
    // Override existing tiled sharding only when the new sharding is compatible
    // with the existing one. This avoids unexpected resharding when `sharding`
    // just has more tiles than existing sharding but they are not mergeable.
    if (!allow_aggressive_resharding && to_improved_shape.IsArray() &&
        !to_improved->IsTileMaximal() && from.NumTiles() == sharding_tiles) {
      if (!hlo_sharding_util::IsSubTilingOrEqualSharding(to_improved_shape,
                                                         from, *to_improved)) {
        VLOG(10) << "Not merging because of different device distribution";
        VLOG(10) << "Instr sharding: " << to_improved->ToString();
        VLOG(10) << "New sharding " << from.ToString();
        return std::nullopt;
      }
    }
    return std::move(from);
  }
  return std::nullopt;
}

// Returning the improved sharding of an instruction from some other sharding.
std::optional<HloSharding> ReturnImprovedSharding(
    HloSharding sharding, HloInstruction* instruction,
    bool may_combine_partial_sharding,
    bool allow_aggressive_resharding = false) {
  return ReturnImprovedShardingImpl(
      std::move(sharding),
      instruction->has_sharding() ? &instruction->sharding() : nullptr,
      instruction->shape(), may_combine_partial_sharding,
      allow_aggressive_resharding);
}

// Same as above, but return the improved subsharding of a tuple-shaped
// instruction.
std::optional<HloSharding> ReturnImprovedSubSharding(
    HloSharding sharding, HloInstruction* instruction, const ShapeIndex& index,
    bool may_combine_partial_sharding,
    bool allow_aggressive_resharding = false) {
  if (instruction->has_sharding()) {
    const HloSharding to_improved =
        instruction->sharding().GetSubSharding(instruction->shape(), index);
    return ReturnImprovedShardingImpl(
        std::move(sharding), &to_improved,
        ShapeUtil::GetSubshape(instruction->shape(), index),
        may_combine_partial_sharding, allow_aggressive_resharding);

  } else {
    return ReturnImprovedShardingImpl(
        std::move(sharding), nullptr,
        ShapeUtil::GetSubshape(instruction->shape(), index),
        may_combine_partial_sharding, allow_aggressive_resharding);
  }
}

// Updates the sharding of the specified instruction with the specified sharding
// if it is better than the current one and returns true if a new sharding have
// been applied. If may_combine_partial_sharding is true, this may combine the
// new and existing sharding if they are both partial tiling partial
// replication.
bool MaybeImproveInstructionSharding(HloSharding sharding,
                                     HloInstruction* instruction,
                                     bool may_combine_partial_sharding,
                                     bool allow_aggressive_resharding = false) {
  if (auto new_sharding = ReturnImprovedSharding(
          std::move(sharding), instruction, may_combine_partial_sharding,
          allow_aggressive_resharding)) {
    instruction->set_sharding(std::move(*new_sharding));
    return true;
  }
  return false;
}

// Same as above, but improve the subsharding of an maybe tuple-shaped
// instruction.
bool MaybeImproveInstructionSubSharding(
    HloSharding sharding, HloInstruction* instruction, const ShapeIndex& index,
    bool may_combine_partial_sharding,
    bool allow_aggressive_resharding = false) {
  if (instruction->shape().IsTuple()) {
    if (auto new_sub_sharding = ReturnImprovedSubSharding(
            std::move(sharding), instruction, index,
            may_combine_partial_sharding, allow_aggressive_resharding)) {
      HloSharding new_sharding =
          instruction->has_sharding()
              ? instruction->sharding()
              : HloSharding::Single(instruction->shape(),
                                    HloSharding::Replicate());
      ShapeTree<HloSharding> sharding_shape_tree =
          new_sharding.GetAsShapeTree(instruction->shape());
      *sharding_shape_tree.mutable_element(index) = new_sub_sharding.value();
      instruction->set_sharding(HloSharding::Tuple(sharding_shape_tree));
      return true;
    } else {
      return false;
    }
  }
  CHECK(index.size() == 1 && index[0] == 0);
  return MaybeImproveInstructionSharding(std::move(sharding), instruction,
                                         may_combine_partial_sharding,
                                         allow_aggressive_resharding);
}

// We consider a convolution kernel to be small iff it is smaller along all
// spatial dimensions than the output of the convolution. The rational is that
// we can either shard the kernel or the output and we want to shard the larger
// one for better efficiency.
bool IsConvolutionKernelSmall(const HloInstruction* instruction) {
  CHECK_EQ(instruction->opcode(), HloOpcode::kConvolution);
  const HloInstruction* rhs = instruction->operand(1);
  const auto& dnums = instruction->convolution_dimension_numbers();
  int64_t kernel_dim_prod = 1;
  int64_t output_dim_prod = 1;
  for (int64_t i = 0; i < dnums.input_spatial_dimensions().size(); ++i) {
    int64_t kernel_dim =
        rhs->shape().dimensions(dnums.kernel_spatial_dimensions(i));
    kernel_dim_prod *= kernel_dim;
    int64_t output_dim =
        instruction->shape().dimensions(dnums.output_spatial_dimensions(i));
    output_dim_prod *= output_dim;
    if (kernel_dim >= output_dim &&
        (i < 2 || kernel_dim > 3 || kernel_dim_prod >= output_dim_prod)) {
      return false;
    }
  }
  return true;
}

bool IsPassthroughCustomOps(const HloInstruction* hlo) {
  if (hlo->IsCustomCall(
          absl::Span<const absl::string_view>{"Sharding", "X64Combine"})) {
    return true;
  }
  if (hlo->operand_count() != 1 || !hlo->shape().IsArray() ||
      !hlo->operand(0)->shape().IsArray() ||
      hlo->operand(0)->shape().rank() != hlo->shape().rank()) {
    return false;
  }

  return hlo->IsCustomCall(
      {"ResizeNearest", "ResizeBilinear", "ResizeNearestGrad",
       "ResizeBilinearGrad", "Cholesky",
       host_memory_offload_annotations::kMoveToDeviceCustomCallTarget,
       host_memory_offload_annotations::kMoveToHostCustomCallTarget});
}

// Return the operand which is the most suitable for determining the sharding
// for the specified instruction or nullptr if there isn't any suitable operand.
const HloInstruction* PickRepresentativeOperand(
    const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kMap:
    case HloOpcode::kPad:
    case HloOpcode::kPower:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
      // For these opcodes the output sharding has to be determined by the
      // sharding of the first operand but we can only determine sharding based
      // on it if it already has a sharding.
      if (instruction->operand(0)->has_sharding()) {
        return instruction->operand(0);
      }
      return nullptr;
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kAtan2:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kDivide:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kLogistic:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kRemainder:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kTopK:
    case HloOpcode::kSort:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kStochasticConvert:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kWhile:
    case HloOpcode::kXor: {
      // For these opcodes the output sharding can be determined by any operand
      // so we find the operand with the most specific sharding.
      const HloInstruction* best_operand = nullptr;
      for (const HloInstruction* operand : instruction->operands()) {
        if (operand->has_sharding() &&
            (best_operand == nullptr ||
             hlo_sharding_util::IsShardingMoreSpecific(
                 operand->sharding(), best_operand->sharding()))) {
          best_operand = operand;
        }
      }
      return best_operand;
    }
    case HloOpcode::kCustomCall: {
      if (IsPassthroughCustomOps(instruction)) {
        return instruction->operand(0);
      }
      return nullptr;
    }
    // There is no suitable operand for the rest of the opcodes.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCall:
    case HloOpcode::kCholesky:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kConditional:
    case HloOpcode::kConstant:
    case HloOpcode::kConvolution:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kDomain:
    case HloOpcode::kDot:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kFft:
    case HloOpcode::kFusion:
    case HloOpcode::kGather:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kInfeed:
    case HloOpcode::kIota:
    case HloOpcode::kOutfeed:
    case HloOpcode::kParameter:
    case HloOpcode::kPartitionId:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReplicaId:
    case HloOpcode::kReshape:
    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kTranspose:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kTuple:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
      return nullptr;
  }
}

bool SupportSpatialPartitioning(
    const HloInstruction* instruction,
    const ShardingPropagation::ComputationMap& computation_map, bool is_spmd,
    bool allow_spmd_sharding_propagation_to_output,
    bool allow_spmd_sharding_propagation_to_parameters,
    const CustomCallShardingHelper* sharding_helper) {
  const bool is_entry_root = instruction->parent()
                                 ->parent()
                                 ->entry_computation()
                                 ->root_instruction() == instruction;
  if (instruction->parent()->root_instruction() == instruction &&
      computation_map.find(instruction->parent()) == computation_map.end() &&
      !(is_entry_root && allow_spmd_sharding_propagation_to_output)) {
    // We don't support sharding the root instruction of a computation yet,
    // unless the computation is a while body.
    return false;
  }

  if (instruction->IsElementwise() &&
      (instruction->opcode() != HloOpcode::kRng || is_spmd)) {
    return true;
  }
  switch (instruction->opcode()) {
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConditional:
    case HloOpcode::kConstant:
    case HloOpcode::kConvolution:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kDot:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kInfeed:
    case HloOpcode::kIota:
    case HloOpcode::kPad:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReshape:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSlice:
    case HloOpcode::kSort:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
    case HloOpcode::kReduce:
    case HloOpcode::kRngBitGenerator:
      return true;
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
      // Only if channel_id is not specified.
      return instruction->channel_id() == std::nullopt;
    case HloOpcode::kParameter:
      return allow_spmd_sharding_propagation_to_parameters ||
             computation_map.find(instruction->parent()) !=
                 computation_map.end();
    case HloOpcode::kReverse:
      return is_spmd;
    case HloOpcode::kCustomCall:
      if (!is_spmd) {
        return false;
      }
      if (auto* partitioner =
              GetCustomCallPartitioner(instruction->custom_call_target())) {
        return partitioner->IsCustomCallShardable(instruction);
      }
      return (IsPassthroughCustomOps(instruction) ||
              sharding_helper->IsCustomCallShardable(instruction));
    default:
      return false;
  }
}

// Helper to lookahead sharding of user of an instruction to be used as guidance
// for ambiguous cases.
std::optional<HloSharding> LookaheadUserSharding(HloInstruction* instr,
                                                 bool is_spmd,
                                                 const CallGraph& call_graph) {
  if (instr->user_count() != 1) {
    return std::nullopt;
  }
  HloInstruction* current_user = instr->users()[0];
  std::optional<HloSharding> sharding;
  std::vector<HloInstruction*> users_chain = {instr, current_user};
  // Collect single user instructions along the way.
  while (!current_user->has_sharding()) {
    // Only consider single user chains.
    if (current_user->users().size() != 1) {
      users_chain.clear();
      break;
    }
    current_user = current_user->users()[0];
    users_chain.push_back(current_user);
  }
  // Early exit for unsupported cases.
  if (users_chain.empty()) {
    return std::nullopt;
  }
  for (int i = users_chain.size() - 1; i >= 1; --i) {
    HloInstruction* user = users_chain[i];
    HloInstruction* current = users_chain[i - 1];
    CHECK(user->has_sharding());
    sharding = ShardingPropagation::GetShardingFromUser(
        *current, *user, INT64_MAX, is_spmd, call_graph,
        /*sharding_helper=*/nullptr);
    // We need to set the sharding to the instruction, because
    // GetShardingFromUser() interface uses sharding from the instruction
    // itself. It will be cleared out later.
    if (sharding.has_value() && i != 1) {
      current->set_sharding(*sharding);
      continue;
    }
    break;
  }
  // Clear the sharding of the middle instructions we set the sharding of
  // because they were unsharded.
  for (int i = 1; i < users_chain.size() - 1; ++i) {
    users_chain[i]->clear_sharding();
  }
  return sharding;
}

// Infer output sharding on index parallel dimensions for gather/scatter from
// gather operand/indices or scatter operands/indices/updates.
HloSharding InferParallelShardingFromOperand(
    const HloInstruction* operand, const Shape& shape,
    absl::Span<const int64_t> output_aligned_operand_parallel_dims,
    absl::Span<const int64_t> output_parallel_dims) {
  const HloSharding& operand_sharding = operand->sharding();
  if (operand_sharding.IsTileMaximal()) {
    return operand_sharding;
  }
  std::vector<int64_t> output_tile_dims(shape.rank(), 1);
  std::vector<int64_t> operand_non_parallel_dims;
  operand_non_parallel_dims.reserve(operand->shape().rank());
  // Detect non parallel dimensions in the operand.
  for (int i = 0; i < operand->shape().rank(); ++i) {
    if (!absl::c_linear_search(output_aligned_operand_parallel_dims, i)) {
      operand_non_parallel_dims.push_back(i);
    }
  }
  // Collect tile dimensions in the operand. The order of the parallel
  // dimensions in output_aligned_operand_parallel_dims is the same as that of
  // the output
  for (int i = 0; i < output_aligned_operand_parallel_dims.size(); ++i) {
    const int64_t operand_idx = output_aligned_operand_parallel_dims[i];
    const int64_t output_idx = output_parallel_dims[i];
    output_tile_dims[output_idx] =
        operand_sharding.tile_assignment().dim(operand_idx);
  }
  HloSharding replicate_non_parallel_dims =
      hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
          operand_sharding, operand_non_parallel_dims);
  if (replicate_non_parallel_dims.IsTileMaximal()) {
    return replicate_non_parallel_dims;
  }
  for (int64_t i = replicate_non_parallel_dims.TiledDataRank();
       i < replicate_non_parallel_dims.tile_assignment().num_dimensions();
       ++i) {
    output_tile_dims.push_back(
        replicate_non_parallel_dims.tile_assignment().dim(i));
  }
  auto output_tile_assignment =
      replicate_non_parallel_dims.tile_assignment().Reshape(output_tile_dims);
  return replicate_non_parallel_dims.ReplicateOnLastTileDim()
             ? HloSharding::PartialTile(output_tile_assignment,
                                        replicate_non_parallel_dims.metadata())
             : HloSharding::Subgroup(
                   output_tile_assignment,
                   replicate_non_parallel_dims.subgroup_types(),
                   replicate_non_parallel_dims.metadata());
}

// Infer output sharding on index parallel dimensions for gather from operand
// and indices.
bool InferGatherParallelShardingFromOperands(
    HloInstruction* instruction,
    const hlo_sharding_util::GatherScatterParallelDims& parallel_dims,
    bool may_combine_partial_sharding) {
  CHECK(DynCast<HloGatherInstruction>(instruction));
  bool changed = false;
  auto aligned_operand_parallel_dims =
      hlo_sharding_util::IndexAlignedOperandParallelDims(parallel_dims);
  auto output_parallel_dims = hlo_sharding_util::GetGatherParallelOutputDims(
      *instruction, parallel_dims);
  // Infer output sharding from scatter operand sharding.
  if (IsSpatiallyPartitioned(instruction->operand(0))) {
    changed |= MaybeImproveInstructionSharding(
        InferParallelShardingFromOperand(
            instruction->operand(0), instruction->shape(),
            absl::MakeConstSpan(aligned_operand_parallel_dims),
            absl::MakeConstSpan(output_parallel_dims)),
        instruction, may_combine_partial_sharding);
  }
  // Infer output sharding from scatter indices sharding.
  if (IsSpatiallyPartitioned(instruction->operand(1))) {
    changed |= MaybeImproveInstructionSharding(
        InferParallelShardingFromOperand(
            instruction->operand(1), instruction->shape(),
            absl::MakeConstSpan(parallel_dims.indices_parallel_dims),
            absl::MakeConstSpan(output_parallel_dims)),
        instruction, may_combine_partial_sharding);
  }
  return changed;
}

// Infer output sharding on index parallel dimensions for scatter from operands,
// indices and updates.
bool InferScatterParallelShardingFromOperands(
    HloInstruction* instruction,
    const hlo_sharding_util::GatherScatterParallelDims& parallel_dims,
    bool may_combine_partial_sharding) {
  HloScatterInstruction* scatter = DynCast<HloScatterInstruction>(instruction);
  CHECK(scatter);
  const int64_t operand_count = scatter->scatter_operand_count();
  auto scatter_operands = scatter->scatter_operands();
  auto scatter_indices = scatter->scatter_indices();
  auto scatter_updates = scatter->scatter_updates();
  bool changed = false;
  auto aligned_operand_parallel_dims =
      hlo_sharding_util::IndexAlignedOperandParallelDims(parallel_dims);
  auto update_parallel_dims = hlo_sharding_util::GetScatterParallelUpdateDims(
      *instruction, parallel_dims);
  auto output_parallel_dims = aligned_operand_parallel_dims;
  // Infer output sharding from scatter operand sharding.
  Shape shape = operand_count == 1
                    ? instruction->shape()
                    : ShapeUtil::GetSubshape(instruction->shape(), {0});
  for (int64_t i = 0; i != operand_count; ++i) {
    if (IsSpatiallyPartitioned(scatter_operands[i])) {
      changed |= MaybeImproveInstructionSubSharding(
          InferParallelShardingFromOperand(
              scatter_operands[i], shape,
              absl::MakeConstSpan(aligned_operand_parallel_dims),
              absl::MakeConstSpan(output_parallel_dims)),
          instruction, {i}, may_combine_partial_sharding);
    }
  }
  // Infer output sharding from scatter indices sharding.
  if (IsSpatiallyPartitioned(scatter_indices)) {
    auto parallel_sharding_from_indices = InferParallelShardingFromOperand(
        scatter_indices, shape,
        absl::MakeConstSpan(parallel_dims.indices_parallel_dims),
        absl::MakeConstSpan(output_parallel_dims));
    for (int64_t i = 0; i != operand_count; ++i) {
      changed |= MaybeImproveInstructionSubSharding(
          parallel_sharding_from_indices, instruction, {i},
          may_combine_partial_sharding);
    }
  }
  // Infer output sharding from scatter update sharding.
  for (int64_t i = 0; i != operand_count; ++i) {
    if (IsSpatiallyPartitioned(scatter_updates[i])) {
      changed |= MaybeImproveInstructionSubSharding(
          InferParallelShardingFromOperand(
              scatter_updates[i], shape,
              absl::MakeConstSpan(update_parallel_dims),
              absl::MakeConstSpan(output_parallel_dims)),
          instruction, {i}, may_combine_partial_sharding);
    }
  }
  return changed;
}

bool CanPropagateThroughAtAggressiveLevel(const HloInstruction& inst,
                                          int64_t aggressiveness) {
  // At minimum aggressiveness, only allow pass-through ops.
  if (aggressiveness < 1 &&
      !(inst.IsElementwise() || inst.IsCustomCall("Sharding")) &&
      inst.opcode() != HloOpcode::kTranspose &&
      inst.opcode() != HloOpcode::kReshape &&
      inst.opcode() != HloOpcode::kTuple &&
      inst.opcode() != HloOpcode::kGetTupleElement &&
      inst.opcode() != HloOpcode::kWhile &&
      inst.opcode() != HloOpcode::kDynamicSlice &&
      inst.opcode() != HloOpcode::kDynamicUpdateSlice &&
      inst.opcode() != HloOpcode::kOptimizationBarrier &&
      inst.opcode() != HloOpcode::kConcatenate &&
      inst.opcode() != HloOpcode::kCall && inst.opcode() != HloOpcode::kCopy) {
    return false;
  }
  // Broadcast propagation should have at least aggressiveness 2.
  if (aggressiveness < 2 && inst.opcode() == HloOpcode::kBroadcast) {
    return false;
  }
  return true;
}

HloSharding InferDotOperandSharding(
    const HloInstruction* instruction,
    const dot_as_convolution_util::DotConvolutionDimsInfo& dnums,
    int64_t operand_index, bool may_combine_partial_sharding) {
  auto operand = instruction->operand(operand_index);
  auto other = instruction->operand(1 - operand_index);
  std::vector<int64_t> output_dims_to_replicate;
  std::vector<int64_t> other_operand_dims_to_replicate;
  for (const auto& dim : operand_index == 0 ? dnums.rhs_non_contracting_dims
                                            : dnums.lhs_non_contracting_dims) {
    output_dims_to_replicate.push_back(dim.output);
    other_operand_dims_to_replicate.push_back(operand_index == 0 ? dim.rhs
                                                                 : dim.lhs);
  }
  // If this dot is interpreted from a conv, then contracting dims may have
  // corresponding spatial dimensions in the output, and this operand's
  // non-contracting dims may have corresponding spatial dims in the other
  // operand.
  for (const auto& dim : dnums.contracting_dims) {
    if (dim.output >= 0) {
      output_dims_to_replicate.push_back(dim.output);
    }
  }
  for (const auto& dim : operand_index == 0 ? dnums.lhs_non_contracting_dims
                                            : dnums.rhs_non_contracting_dims) {
    int64_t other_dim = operand_index == 0 ? dim.rhs : dim.lhs;
    if (other_dim >= 0) {
      other_operand_dims_to_replicate.push_back(other_dim);
    }
  }
  auto output_other_dims_replicated =
      hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
          instruction->sharding(), output_dims_to_replicate);
  std::vector<int64_t> output_to_operand_dims(instruction->shape().rank(), -1);
  std::vector<int64_t> operand_to_output_dims(operand->shape().rank(), -1);
  for (const auto& dim : dnums.batch_dims) {
    output_to_operand_dims[dim.output] = operand_index == 0 ? dim.lhs : dim.rhs;
    operand_to_output_dims[operand_index == 0 ? dim.lhs : dim.rhs] = dim.output;
  }
  for (const auto& dim : operand_index == 0 ? dnums.lhs_non_contracting_dims
                                            : dnums.rhs_non_contracting_dims) {
    output_to_operand_dims[dim.output] = operand_index == 0 ? dim.lhs : dim.rhs;
    operand_to_output_dims[operand_index == 0 ? dim.lhs : dim.rhs] = dim.output;
  }
  auto sharding = *hlo_sharding_util::TransposeShardingWithCollapsedDims(
      output_other_dims_replicated, output_to_operand_dims,
      operand_to_output_dims);
  if (IsSpatiallyPartitioned(other)) {
    auto other_operand_dims_replicated =
        hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
            other->sharding(), other_operand_dims_to_replicate);
    std::vector<int64_t> other_to_operand_dims(other->shape().rank(), -1);
    std::vector<int64_t> operand_to_other_dims(operand->shape().rank(), -1);
    for (const auto& dim : dnums.batch_dims) {
      other_to_operand_dims[operand_index == 0 ? dim.rhs : dim.lhs] =
          operand_index == 0 ? dim.lhs : dim.rhs;
      operand_to_other_dims[operand_index == 0 ? dim.lhs : dim.rhs] =
          operand_index == 0 ? dim.rhs : dim.lhs;
    }
    for (const auto& dim : dnums.contracting_dims) {
      other_to_operand_dims[operand_index == 0 ? dim.rhs : dim.lhs] =
          operand_index == 0 ? dim.lhs : dim.rhs;
      operand_to_other_dims[operand_index == 0 ? dim.lhs : dim.rhs] =
          operand_index == 0 ? dim.rhs : dim.lhs;
    }
    HloSharding sharding_from_other =
        *hlo_sharding_util::TransposeShardingWithCollapsedDims(
            other_operand_dims_replicated, other_to_operand_dims,
            operand_to_other_dims);
    if (hlo_sharding_util::MergeSharding(sharding, &sharding_from_other,
                                         may_combine_partial_sharding)) {
      sharding = std::move(sharding_from_other);
    }
  }
  return sharding;
}

// Checks if two HloShardings have the same metadata attached.
bool SameShardingMetadata(const HloSharding& a, const HloSharding& b) {
  DCHECK_EQ(a, b);

  auto same_metadata = [](absl::Span<const OpMetadata> a,
                          absl::Span<const OpMetadata> b) {
    if (a.size() != b.size()) return false;
    for (int i = 0, e = a.size(); i < e; ++i) {
      if (!protobuf_util::ProtobufEquals(a[i], b[i])) {
        return false;
      }
    }
    return true;
  };

  if (a.IsTuple()) {
    for (int i = 0, e = a.tuple_elements().size(); i < e; ++i) {
      if (!same_metadata(a.tuple_elements()[i].metadata(),
                         b.tuple_elements()[i].metadata())) {
        return false;
      }
    }
    return true;
  } else {
    return same_metadata(a.metadata(), b.metadata());
  }
}

// Assigns metadata to optional sharding on instructions if instructions have
// metadata. If sharding already has some metadata, no new metadata will be
// added.
bool AssignShardingMetadata(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      const auto& metadata = instruction->metadata();
      if (!instruction->has_sharding() || metadata.ByteSizeLong() == 0) {
        continue;
      }

      HloSharding sharding_with_metadata =
          instruction->sharding().WithMetadata({metadata},
                                               /*overwrite=*/false);
      if (!SameShardingMetadata(instruction->sharding(),
                                sharding_with_metadata)) {
        instruction->set_sharding(std::move(sharding_with_metadata));
        changed = true;
      }
    }
  }
  return changed;
}

// Removes all sharding metadata from shardings on instructions.
bool RemoveShardingMetadata(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (!instruction->has_sharding()) {
        continue;
      }
      HloSharding sharding_no_metadata =
          instruction->sharding().WithoutMetadata();
      if (!SameShardingMetadata(instruction->sharding(),
                                sharding_no_metadata)) {
        instruction->set_sharding(std::move(sharding_no_metadata));
        changed = true;
      }
    }
  }
  return changed;
}

// If a while contains a channel instruction on device D, check that any other
// instructions with a device assignment are on D. Further, annotate the root
// instruction of the while body to ensure that HLO partitioning will keep the
// entire while instruction on D.
Status CheckAndUpdateDeviceAssignmentsInWhileBody(
    HloInstruction* while_instruction) {
  auto bad_status = [](HloInstruction* instruction, int64_t device,
                       HloInstruction* channel_instruction,
                       int64_t correct_device) {
    return FailedPrecondition(
        "Instruction: %s is on device: %d, which conflicts with device: %d "
        "of channel instruction: %s",
        instruction->name(), device, correct_device,
        channel_instruction->name());
  };

  CHECK_EQ(while_instruction->opcode(), HloOpcode::kWhile);
  HloComputation* while_body = while_instruction->while_body();
  // Maps a device number to an instruction in the while_body with that
  // device assignment.
  std::map<int64_t, HloInstruction*> devices_to_instructions;
  std::optional<int64_t> unique_device = std::nullopt;
  HloInstruction* channel_instruction = nullptr;

  for (HloInstruction* instruction : while_body->instructions()) {
    if (instruction->sharding_unique_device()) {
      auto opcode = instruction->opcode();
      int64_t device = *instruction->sharding_unique_device();
      if (unique_device.has_value()) {
        if (*unique_device != device) {
          return bad_status(instruction, device, channel_instruction,
                            *unique_device);
        }
      } else if (((opcode == HloOpcode::kSend || opcode == HloOpcode::kRecv) &&
                  !Cast<HloSendRecvInstruction>(instruction)
                       ->is_host_transfer())
                 // Cross-replica AllReduces don't have a channel_id, and we
                 // don't enforce any invariant about their device assignment.
                 || ((opcode == HloOpcode::kAllReduce ||
                      opcode == HloOpcode::kReduceScatter) &&
                     instruction->channel_id())) {
        channel_instruction = instruction;
        unique_device = device;
        if (!devices_to_instructions.empty()) {
          for (auto it = devices_to_instructions.begin();
               it != devices_to_instructions.end(); ++it) {
            if (*unique_device != it->first) {
              return bad_status(it->second, it->first, channel_instruction,
                                *unique_device);
            }
          }
        }
      } else {
        devices_to_instructions[device] = instruction;
      }
    }
  }

  if (unique_device.has_value()) {
    auto while_device = while_instruction->sharding_unique_device();
    if (while_device.has_value() && *unique_device != *while_device) {
      return bad_status(while_instruction, *while_device, channel_instruction,
                        *unique_device);
    }
    auto body_root = while_body->root_instruction();
    auto root_device = body_root->sharding_unique_device();
    if (!root_device.has_value()) {
      body_root->set_device_sharding(*unique_device);
    } else if (*unique_device != *root_device) {
      return bad_status(body_root, *root_device, channel_instruction,
                        *unique_device);
    }
  }
  return OkStatus();
}

// Refines a pair of auto/manual shardings based on auto sharding `to_merge`
// along `unspecified_dims`. Returns if anything changed.
bool RefineManualAutoShardingFromAuto(
    const HloSharding& to_merge, absl::Span<const int64_t> unspecified_dims,
    HloSharding* auto_sharding, HloSharding* manual_sharding) {
  if (!manual_sharding->IsManualSubgroup() ||
      auto_sharding->IsManualSubgroup() ||
      !manual_sharding->HasPartialReplication() ||
      manual_sharding->subgroup_types().size() != 2) {
    // We do not support nested subgroup manual. man_conversion_op must have
    // replication in order to be merged.
    return false;
  }
  HloSharding partial_rep =
      hlo_sharding_util::PartiallyReplicateTiledShardingOnAllDimsExcept(
          to_merge, unspecified_dims);
  if (partial_rep.IsTileMaximal()) {
    return false;
  }

  // Merge with the non-manual partial annotation.
  if (!hlo_sharding_util::MergeShardingIfCompatible(
          partial_rep, auto_sharding->NumTiles() + 1, auto_sharding)) {
    return false;
  }

  // Merge with the manual partial annotation.
  const int64_t data_rank = partial_rep.TiledDataRank();
  // We are also merging the non-manual sharding into the manual sharding. To
  // leverage existing merging implementation, we treat the manual dim as a
  // data dim, and add it right before the replication dim.
  std::vector<int64_t> partial_manual_shape(
      partial_rep.tile_assignment().dimensions().begin(),
      partial_rep.tile_assignment().dimensions().end());
  partial_manual_shape.insert(partial_manual_shape.begin() + data_rank, 1);
  auto partial_tiling_for_manual =
      partial_rep.tile_assignment().Reshape(partial_manual_shape);
  HloSharding partial_rep_for_manual = HloSharding::PartialTile(
      partial_tiling_for_manual, partial_rep.metadata());
  auto man_tiling = manual_sharding->tile_assignment();
  if (manual_sharding->subgroup_types().back() != OpSharding::REPLICATED) {
    // Move the manual dim before replication dim.
    std::vector<int> transposed_dims(man_tiling.num_dimensions());
    absl::c_iota(transposed_dims, 0);
    std::swap(transposed_dims.back(), transposed_dims[data_rank]);
    man_tiling = man_tiling.Transpose(transposed_dims);
  }
  HloSharding tmp_sharding_for_merging = HloSharding::PartialTile(
      std::move(man_tiling), manual_sharding->metadata());
  if (!hlo_sharding_util::MergeShardingIfCompatible(
          partial_rep_for_manual, tmp_sharding_for_merging.NumTiles() + 1,
          &tmp_sharding_for_merging)) {
    return false;
  }

  std::vector<OpSharding::Type> subgroup_types;
  subgroup_types.push_back(OpSharding::MANUAL);
  if (tmp_sharding_for_merging.HasPartialReplication()) {
    subgroup_types.push_back(OpSharding::REPLICATED);
  }
  *manual_sharding = HloSharding::Subgroup(
      tmp_sharding_for_merging.tile_assignment(), subgroup_types,
      tmp_sharding_for_merging.metadata());
  return true;
}

// Refines a pair of auto/manual shardings based on manual sharding `to_merge`
// along `unspecified_dims`. Returns if anything changed.
bool RefineManualAutoShardingFromManual(
    const HloSharding& to_merge, absl::Span<const int64_t> unspecified_dims,
    HloSharding* auto_sharding, HloSharding* manual_sharding) {
  if (!to_merge.IsManualSubgroup() || !manual_sharding->IsManualSubgroup() ||
      !manual_sharding->HasPartialReplication() ||
      auto_sharding->IsManualSubgroup() ||
      manual_sharding->subgroup_types().size() != 2) {
    return false;
  }
  HloSharding partial_rep =
      hlo_sharding_util::PartiallyReplicateTiledShardingOnAllDimsExcept(
          to_merge, unspecified_dims);
  if (partial_rep.IsTileMaximal()) {
    return false;
  }
  if (!hlo_sharding_util::MergeShardingIfCompatible(
          partial_rep, manual_sharding->NumTiles() + 1, manual_sharding)) {
    return false;
  }
  HloSharding partial_rep_for_auto = HloSharding::Subgroup(
      partial_rep.tile_assignment(),
      std::vector<OpSharding::Type>(partial_rep.subgroup_types().size(),
                                    OpSharding::REPLICATED),
      partial_rep.metadata());
  if (!hlo_sharding_util::MergeShardingIfCompatible(
          partial_rep_for_auto, auto_sharding->NumTiles() + 1, auto_sharding)) {
    return false;
  }
  return true;
}

bool InferUnspecifiedDimsFromOperand(HloInstruction* annotate_op,
                                     absl::Span<const int64_t> unspecified_dims,
                                     HloInstruction** man_conversion_op_after) {
  // ProcessShardingInstruction will either keep the "Sharding" custom call as
  // is or replace it with a copy.
  CHECK(annotate_op->IsCustomCall("Sharding") ||
        annotate_op->opcode() == HloOpcode::kCopy);
  if (!IsSpatiallyPartitioned(annotate_op->operand(0))) {
    return false;
  }
  const HloSharding& operand_sharding = annotate_op->operand(0)->sharding();
  if (!operand_sharding.IsTiled()) {
    return false;
  }
  HloInstruction* man_conversion_op = nullptr;
  if (annotate_op->user_count() == 1) {
    HloInstruction* user = annotate_op->users()[0];
    if (user->IsCustomCall("SPMDFullToShardShape") ||
        user->IsCustomCall("SPMDShardToFullShape")) {
      std::vector<int64_t> user_unspec_dims;
      if (!sharding_op_util::ParseAttributes(
               Cast<HloCustomCallInstruction>(user)->opaque(),
               &user_unspec_dims)
               .ok()) {
        return false;
      }
      absl::c_sort(user_unspec_dims);
      if (unspecified_dims != user_unspec_dims) {
        // The manual/auto conversion op must have the same set of unspecified
        // dims.
        return false;
      }
      man_conversion_op = user;
    }
  }
  *man_conversion_op_after = man_conversion_op;
  if (man_conversion_op == nullptr) {
    HloSharding partial_replicated =
        hlo_sharding_util::PartiallyReplicateTiledShardingOnAllDimsExcept(
            operand_sharding, unspecified_dims);
    HloSharding sharding = annotate_op->sharding();
    if (!hlo_sharding_util::MergeShardingIfCompatible(
            partial_replicated, sharding.NumTiles() + 1, &sharding)) {
      return false;
    }
    annotate_op->set_sharding(sharding);
    return true;
  }
  if (man_conversion_op->IsCustomCall("SPMDFullToShardShape")) {
    HloSharding auto_sharding = annotate_op->sharding();
    HloSharding manual_sharding = man_conversion_op->sharding();
    if (!RefineManualAutoShardingFromAuto(operand_sharding, unspecified_dims,
                                          &auto_sharding, &manual_sharding)) {
      return false;
    }
    annotate_op->set_sharding(auto_sharding);
    man_conversion_op->set_sharding(manual_sharding);
    return true;
  }

  CHECK(man_conversion_op->IsCustomCall("SPMDShardToFullShape"));
  HloSharding manual_sharding = annotate_op->sharding();
  HloSharding auto_sharding = man_conversion_op->sharding();
  if (!RefineManualAutoShardingFromManual(operand_sharding, unspecified_dims,
                                          &auto_sharding, &manual_sharding)) {
    return false;
  }
  annotate_op->set_sharding(manual_sharding);
  man_conversion_op->set_sharding(auto_sharding);
  return true;
}

bool InferUnspecifiedDimsFromOneUser(HloInstruction* annotate_op,
                                     const HloInstruction* user,
                                     int64_t aggressiveness, bool is_spmd,
                                     absl::Span<const int64_t> unspecified_dims,
                                     HloInstruction* man_conversion_op,
                                     const CallGraph& call_graph) {
  CHECK(annotate_op->IsCustomCall("Sharding") ||
        annotate_op->opcode() == HloOpcode::kCopy);
  if (!user->has_sharding() || !user->sharding().IsTiled()) {
    return false;
  }
  std::optional<HloSharding> user_sharding =
      ShardingPropagation::GetShardingFromUser(
          man_conversion_op == nullptr ? *annotate_op : *man_conversion_op,
          *user, aggressiveness, is_spmd, call_graph,
          /*sharding_helper=*/nullptr);
  if (!user_sharding.has_value() || user_sharding->IsTileMaximal()) {
    return false;
  }
  if (man_conversion_op == nullptr) {
    HloSharding partial_replicated =
        hlo_sharding_util::PartiallyReplicateTiledShardingOnAllDimsExcept(
            *user_sharding, unspecified_dims);
    HloSharding sharding = annotate_op->sharding();
    if (!hlo_sharding_util::MergeShardingIfCompatible(
            partial_replicated, sharding.NumTiles() + 1, &sharding)) {
      return false;
    }
    annotate_op->set_sharding(sharding);
    return true;
  }
  if (man_conversion_op->IsCustomCall("SPMDFullToShardShape")) {
    HloSharding auto_sharding = annotate_op->sharding();
    HloSharding manual_sharding = man_conversion_op->sharding();
    if (!RefineManualAutoShardingFromManual(*user_sharding, unspecified_dims,
                                            &auto_sharding, &manual_sharding)) {
      return false;
    }
    annotate_op->set_sharding(auto_sharding);
    man_conversion_op->set_sharding(manual_sharding);
    return true;
  }
  CHECK(man_conversion_op->IsCustomCall("SPMDShardToFullShape"));
  HloSharding manual_sharding = annotate_op->sharding();
  HloSharding auto_sharding = man_conversion_op->sharding();
  if (!RefineManualAutoShardingFromAuto(*user_sharding, unspecified_dims,
                                        &auto_sharding, &manual_sharding)) {
    return false;
  }
  annotate_op->set_sharding(manual_sharding);
  man_conversion_op->set_sharding(auto_sharding);
  return true;
}

bool InferUnspecifiedDimsFromUsers(HloInstruction* annotate_op,
                                   absl::Span<const int64_t> unspecified_dims,
                                   int64_t aggressiveness, bool is_spmd,
                                   HloInstruction** man_conversion_op_after,
                                   const CallGraph& call_graph) {
  HloInstruction* man_conversion_op = nullptr;
  if (annotate_op->user_count() == 1) {
    HloInstruction* user = annotate_op->users()[0];
    if (user->IsCustomCall("SPMDFullToShardShape") ||
        user->IsCustomCall("SPMDShardToFullShape")) {
      std::vector<int64_t> user_unspec_dims;
      absl::c_sort(user_unspec_dims);
      if (!sharding_op_util::ParseAttributes(
               Cast<HloCustomCallInstruction>(user)->opaque(),
               &user_unspec_dims)
               .ok() ||
          unspecified_dims != user_unspec_dims) {
        // The manual/auto conversion op must have the same set of unspecified
        // dims.
        return false;
      }
      man_conversion_op = user;
    }
  }
  *man_conversion_op_after = man_conversion_op;

  HloInstruction* op_for_users =
      man_conversion_op == nullptr ? annotate_op : man_conversion_op;
  bool changed = false;
  for (HloInstruction* user : op_for_users->users()) {
    changed |= InferUnspecifiedDimsFromOneUser(
        annotate_op, user, aggressiveness, is_spmd, unspecified_dims,
        man_conversion_op, call_graph);
  }
  return changed;
}

bool InferUnspecifiedDimsFromShardGroup(
    HloInstruction* annotate_op, absl::Span<const int64_t> unspecified_dims,
    const absl::flat_hash_set<HloInstruction*>& shard_group) {
  // ProcessShardingInstruction will either keep the "Sharding" custom call as
  // is or replace it with a copy.
  CHECK(annotate_op->IsCustomCall("Sharding") ||
        annotate_op->opcode() == HloOpcode::kCopy);

  // Do not propagate sharding to ShardBarrierTo custom-call.
  if (annotate_op->IsCustomCall(spmd::kShardBarrierTo)) {
    return false;
  }

  bool changed = false;
  for (const HloInstruction* member : shard_group) {
    if (member == annotate_op) {
      continue;
    }
    // Do not propagate sharding from ShardBarrierFrom custom-call.
    if (member->IsCustomCall(spmd::kShardBarrierFrom)) {
      continue;
    }
    if (!IsSpatiallyPartitioned(member)) {
      continue;
    }
    const HloSharding& member_sharding = member->sharding();
    if (!member_sharding.IsTiled()) {
      continue;
    }
    HloSharding partial_replicated =
        hlo_sharding_util::PartiallyReplicateTiledShardingOnAllDimsExcept(
            member_sharding, unspecified_dims);
    HloSharding sharding = annotate_op->sharding();
    if (!hlo_sharding_util::MergeShardingIfCompatible(
            partial_replicated, sharding.NumTiles() + 1, &sharding)) {
      continue;
    }
    annotate_op->set_sharding(sharding);
    changed |= true;
  }
  return changed;
}

// Returns whether an op is a target for CSE prevention.
bool IsCSEPreventionTarget(const HloInstruction* instruction) {
  // Scalar broadcasts are the most common CSE target that causes cross-layer
  // propagation on unrelated subgraphs.
  return instruction->opcode() == HloOpcode::kBroadcast &&
         instruction->operand(0)->shape().rank() == 0;
}

// Marks a sharding as for CSE prevention/
HloSharding SetCSEPreventionSharding(const HloSharding& sharding) {
  OpMetadata metadata;
  metadata.set_op_name("_sharding_propagation_cse_prevention");
  return sharding.WithMetadata({metadata}, /*overwrite=*/true);
}

// Returns if the sharding is for CSE prevention.
bool IsCSEPreventionSharding(const HloSharding& sharding) {
  if (sharding.metadata().size() != 1) {
    return false;
  }
  return sharding.metadata()[0].op_name() ==
         "_sharding_propagation_cse_prevention";
}

}  // namespace

bool InferDotShardingFromOperands(
    HloInstruction* instruction, const CallGraph& call_graph,
    const dot_as_convolution_util::DotConvolutionDimsInfo& dnums,
    bool may_combine_partial_sharding, bool is_spmd) {
  auto from_operand = [&](int64_t operand_index) {
    auto operand = instruction->operand(operand_index);
    const HloSharding& operand_sharding = operand->sharding();
    if (operand_sharding.IsTileMaximal()) {
      return operand_sharding;
    }
    std::vector<int64_t> contracting_dims;
    contracting_dims.reserve(dnums.contracting_dims.size());
    for (const auto& dim : dnums.contracting_dims) {
      contracting_dims.push_back(operand_index == 0 ? dim.lhs : dim.rhs);
    }
    // It's possible that some size-1 spatial dims of convolutions are parsed as
    // non-contracting dims. We might have tiled dimensions on them.
    for (const auto& dim : operand_index == 0
                               ? dnums.rhs_non_contracting_dims
                               : dnums.lhs_non_contracting_dims) {
      int64_t d = operand_index == 0 ? dim.lhs : dim.rhs;
      if (d >= 0) {
        contracting_dims.push_back(d);
      }
    }
    auto replicate_contracting_dims =
        hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
            operand_sharding, contracting_dims);
    std::vector<int64_t> out_dims_to_op_perm(instruction->shape().rank(), -1);
    std::vector<int64_t> op_dims_to_output_perm(operand->shape().rank(), -1);
    for (const auto& dim : dnums.batch_dims) {
      out_dims_to_op_perm[dim.output] = operand_index == 0 ? dim.lhs : dim.rhs;
      op_dims_to_output_perm[operand_index == 0 ? dim.lhs : dim.rhs] =
          dim.output;
    }
    for (const auto& dim : operand_index == 0
                               ? dnums.lhs_non_contracting_dims
                               : dnums.rhs_non_contracting_dims) {
      out_dims_to_op_perm[dim.output] = operand_index == 0 ? dim.lhs : dim.rhs;
      op_dims_to_output_perm[operand_index == 0 ? dim.lhs : dim.rhs] =
          dim.output;
    }
    return *hlo_sharding_util::TransposeShardingWithCollapsedDims(
        replicate_contracting_dims, op_dims_to_output_perm,
        out_dims_to_op_perm);
  };
  std::optional<HloSharding> improved_operand_0;
  std::optional<HloSharding> improved_operand_1;
  if (IsSpatiallyPartitioned(instruction->operand(0))) {
    improved_operand_0 = ReturnImprovedSharding(
        from_operand(0), instruction, may_combine_partial_sharding,
        /*allow_aggressive_resharding=*/false);
  }
  if (IsSpatiallyPartitioned(instruction->operand(1))) {
    improved_operand_1 = ReturnImprovedSharding(
        from_operand(1), instruction, may_combine_partial_sharding,
        /*allow_aggressive_resharding=*/false);
  }
  // If not improved sharding found then do not set any sharding.
  if (!improved_operand_0.has_value() && !improved_operand_1.has_value()) {
    return false;
  }
  // Sharding found from operand 0 but not operand 1. Set sharding from operand
  // 0
  if (improved_operand_0.has_value() && !improved_operand_1.has_value()) {
    instruction->set_sharding(*improved_operand_0);
    return true;
  }
  // Sharding found from operand 1 but not operand 0. Set sharding from operand
  // 1
  if (!improved_operand_0.has_value() && improved_operand_1.has_value()) {
    instruction->set_sharding(*improved_operand_1);
    return true;
  }
  CHECK(improved_operand_0.has_value() && improved_operand_1.has_value());
  std::optional<HloSharding> lookahead_sharding =
      LookaheadUserSharding(instruction, is_spmd, call_graph);
  std::array<HloSharding, 2> sharding_priority = {*improved_operand_0,
                                                  *improved_operand_1};
  bool priority_defined_with_lookahead = false;
  // Found sharding from lookahead.
  if (lookahead_sharding.has_value()) {
    const bool operand_0_is_lookahead_subtiling =
        hlo_sharding_util::IsSubTilingOrEqualSharding(
            instruction->shape(), *lookahead_sharding, *improved_operand_0);
    const bool operand_1_is_lookahead_subtiling =
        hlo_sharding_util::IsSubTilingOrEqualSharding(
            instruction->shape(), *lookahead_sharding, *improved_operand_1);
    // If the sharding from operand 0 is a subtiling of the user, but not the
    // one from operand 1 prioritize that sharding.
    if (operand_0_is_lookahead_subtiling && !operand_1_is_lookahead_subtiling) {
      priority_defined_with_lookahead = true;
    }
    // If the sharding from operand 1 is a subtiling of the user, but not the
    // one from operand 0 prioritize that sharding.
    if (!operand_0_is_lookahead_subtiling && operand_1_is_lookahead_subtiling) {
      instruction->set_sharding(*improved_operand_1);
      std::swap(sharding_priority[0], sharding_priority[1]);
      priority_defined_with_lookahead = true;
    }
  }
  // If lookahead didn't define a priority then use size.
  if (!priority_defined_with_lookahead &&
      ShapeUtil::ByteSizeOf(instruction->operand(0)->shape()) <
          ShapeUtil::ByteSizeOf(instruction->operand(1)->shape())) {
    std::swap(sharding_priority[0], sharding_priority[1]);
  }
  // Set primary sharding to the instruction and then try to improve it with
  // the secondary sharding.
  instruction->set_sharding(sharding_priority[0]);
  MaybeImproveInstructionSharding(sharding_priority[1], instruction,
                                  may_combine_partial_sharding);
  return true;
}

// Convolution handling for InferShardingFromOperands().
bool InferConvolutionShardingFromOperands(HloInstruction* instruction,
                                          const CallGraph& call_graph,
                                          int64_t aggressiveness,
                                          bool may_combine_partial_sharding,
                                          bool is_spmd) {
  auto get_partitions_for_dims =
      [&](const HloInstruction* inst,
          absl::Span<
              const dot_as_convolution_util::DotConvolutionDimsInfo::DimNums>
              dims,
          int lhs_or_rhs) {
        int64_t partitions = 1;
        if (!inst->has_sharding()) {
          return partitions;
        }
        const auto& sharding = inst->sharding();
        if (sharding.IsTileMaximal()) {
          return partitions;
        }
        for (const auto& dim : dims) {
          if (lhs_or_rhs == 0) {
            partitions *= sharding.tile_assignment().dim(dim.lhs);
          } else {
            CHECK_EQ(lhs_or_rhs, 1);
            partitions *= sharding.tile_assignment().dim(dim.rhs);
          }
        }
        return partitions;
      };
  auto dot_dims =
      dot_as_convolution_util::ParseConvolutionDimsInfo(instruction);
  const int64_t lhs_conv_spatial_partitions = get_partitions_for_dims(
      instruction->operand(0), dot_dims.conv_spatial_dims, 0);
  const int64_t rhs_conv_spatial_partitions = get_partitions_for_dims(
      instruction->operand(1), dot_dims.conv_spatial_dims, 1);
  if (dot_dims.conv_spatial_dims.empty() ||
      (lhs_conv_spatial_partitions == 1 && rhs_conv_spatial_partitions == 1 &&
       instruction->batch_group_count() == 1 &&
       instruction->feature_group_count() == 1)) {
    return InferDotShardingFromOperands(instruction, call_graph, dot_dims,
                                        may_combine_partial_sharding, is_spmd);
  }
  const auto& dnums = instruction->convolution_dimension_numbers();
  const HloInstruction* lhs = instruction->operand(0);
  auto get_tiled_sharding_based_on_lhs = [&] {
    CHECK(!lhs->sharding().IsTileMaximal());
    std::vector<int64_t> output_to_lhs_indices(instruction->shape().rank());
    output_to_lhs_indices[dnums.output_batch_dimension()] =
        dnums.input_batch_dimension();
    output_to_lhs_indices[dnums.output_feature_dimension()] =
        dnums.input_feature_dimension();
    for (int64_t i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
      output_to_lhs_indices[dnums.output_spatial_dimensions(i)] =
          dnums.input_spatial_dimensions(i);
    }
    return hlo_sharding_util::TransposeSharding(lhs->sharding(),
                                                output_to_lhs_indices);
  };
  if (!IsSpatiallyPartitioned(lhs)) {
    return false;
  }
  if (lhs->sharding().IsTileMaximal()) {
    return MaybeImproveInstructionSharding(lhs->sharding(), instruction,
                                           may_combine_partial_sharding);
  }

  if (IsConvolutionKernelSmall(instruction)) {
    // If the kernel is small compared to the input then we can generate an
    // output what is sharded the same way as the input.
    const auto& tile_assignment = lhs->sharding().tile_assignment();
    if (tile_assignment.dim(dnums.input_feature_dimension()) > 1) {
      return false;
    }
    return MaybeImproveInstructionSharding(get_tiled_sharding_based_on_lhs(),
                                           instruction,
                                           may_combine_partial_sharding);
  }
  // If the kernel is large (e.g., backward convolution) then we only support
  // replicated output. We intend to keep the sharding along the batch dimension
  // between lhs and output.
  return MaybeImproveInstructionSharding(
      hlo_sharding_util::PartiallyReplicateTiledShardingOnAllDimsExcept(
          lhs->sharding(), {dnums.input_batch_dimension()}),
      instruction, may_combine_partial_sharding);
}

std::optional<HloSharding> InferBroadcastOperandSharding(
    const HloInstruction& instruction, bool is_spmd) {
  if (instruction.sharding().IsReplicated() ||
      instruction.sharding().IsManual()) {
    return instruction.sharding();
  }
  std::vector<int64_t> dims_to_replicate;
  bool needs_replication = false;
  for (int64_t i = 0; i < instruction.shape().rank(); ++i) {
    if (absl::c_count(instruction.dimensions(), i) == 0) {
      dims_to_replicate.push_back(i);
      if (instruction.sharding().tile_assignment().dim(i) > 1) {
        needs_replication = true;
      }
    }
  }
  // If not SPMD, only support when none of the partitioned dimensions in
  // the broadcast output belong to new dimensions.
  if (!is_spmd && needs_replication) {
    return std::nullopt;
  }
  return hlo_sharding_util::RemoveShapeDimensions(
      hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
          instruction.sharding(), dims_to_replicate),
      dims_to_replicate);
}

bool InferReduceShardingFromOperand(HloInstruction* instruction,
                                    bool may_combine_partial_sharding,
                                    bool is_spmd) {
  auto get_maybe_tuple_sharding = [&](HloSharding sharding) {
    if (instruction->shape().IsArray()) {
      return sharding;
    }
    std::vector<HloSharding> tuple(instruction->shape().tuple_shapes_size(),
                                   std::move(sharding));
    return HloSharding::Tuple(instruction->shape(), tuple);
  };
  auto* reduce = Cast<HloReduceInstruction>(instruction);
  bool changed = false;
  for (HloInstruction* operand : reduce->inputs()) {
    if (!IsSpatiallyPartitioned(operand)) {
      continue;
    }
    if (operand->sharding().IsReplicated() ||
        (!is_spmd &&
         absl::c_any_of(instruction->dimensions(), [operand](int64_t dim) {
           return operand->sharding().tile_assignment().dim(dim) > 1;
         }))) {
      // We are reducing along one of the sharded dimensions. We only
      // support this in SPMD.
      changed |= MaybeImproveInstructionSharding(
          get_maybe_tuple_sharding(
              hlo_sharding_util::ReplicateAllDataDims(operand->sharding())),
          reduce, may_combine_partial_sharding,
          /*allow_aggressive_resharding=*/
          ComputeNonRootUsers(instruction) == 1);
      continue;
    }
    auto after_partial_replication =
        operand->sharding().IsReplicated()
            ? operand->sharding()
            : hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
                  operand->sharding(), reduce->dimensions());
    if (after_partial_replication.IsReplicated()) {
      changed |= MaybeImproveInstructionSharding(
          get_maybe_tuple_sharding(after_partial_replication), reduce,
          may_combine_partial_sharding,
          /*allow_aggressive_resharding=*/
          ComputeNonRootUsers(instruction) == 1);
      continue;
    }
    // Use the same sharding for all tuple elements, because they are part
    // of the same reduce instruction.
    HloSharding new_sharding =
        get_maybe_tuple_sharding(hlo_sharding_util::RemoveShapeDimensions(
            after_partial_replication, reduce->dimensions()));
    changed |= MaybeImproveInstructionSharding(
        std::move(new_sharding), reduce, may_combine_partial_sharding,
        /*allow_aggressive_resharding=*/
        ComputeNonRootUsers(reduce) == 1);
  }
  return changed;
}

// Remove Sharding custom-call instruction by folding the sharding attribute
// to its operand. If the operand already has a different sharding, insert a
// copy node for reshard.
// `unspecified_dims` will be populated with the converted copies if the custom
// call is partially specified.
absl::StatusOr<bool> ProcessShardingInstruction(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    bool replace_sharding_with_copy,
    absl::flat_hash_map<const HloInstruction*, std::vector<int64_t>>*
        unspecified_dims,
    std::vector<HloSharding>* saved_root_shardings,
    absl::flat_hash_map<int64_t, HloSharding>* saved_parameter_shardings,
    absl::flat_hash_map<HloInstruction*, int64_t>*
        instruction_to_shard_group_id,
    absl::flat_hash_map<int64_t, absl::flat_hash_set<HloInstruction*>>*
        shard_group_id_to_shard_as_group,
    absl::flat_hash_map<int64_t, absl::flat_hash_set<HloInstruction*>>*
        shard_group_id_to_shard_like_group,
    const std::vector<bool>*
        allow_spmd_sharding_propagation_to_parameters_vector) {
  bool changed = false;

  const bool use_shard_group = instruction_to_shard_group_id &&
                               shard_group_id_to_shard_as_group &&
                               shard_group_id_to_shard_like_group;
  // Process shard group instruction and returns if current instruction needs
  // to be removed.
  auto process_shard_group_instruction =
      [&](HloInstruction* instruction,
          bool replaced_with_copy) -> absl::StatusOr<bool> {
    // Run shard group processing IFF it's not CSE prevention.
    if (replace_sharding_with_copy) {
      if (use_shard_group && instruction->has_sharding() &&
          instruction->sharding().IsShardGroup()) {
        if (instruction->IsCustomCall("Sharding")) {
          CHECK(instruction->operand(0)->opcode() != HloOpcode::kParameter ||
                (allow_spmd_sharding_propagation_to_parameters_vector &&
                 allow_spmd_sharding_propagation_to_parameters_vector->size() ==
                     module->entry_computation()->num_parameters() &&
                 allow_spmd_sharding_propagation_to_parameters_vector->at(
                     instruction->operand(0)->parameter_number())));
        }
        if (instruction->IsCustomCall("Sharding") && !replaced_with_copy) {
          // Pass shard group to operand sharding custom-call if it's not
          // replaced with a copy, meaning that the shardings are to annotate
          // shard_group.
          HloSharding operand_sharding =
              instruction->operand(0)->has_sharding()
                  ? instruction->operand(0)->sharding()
                  : HloSharding::Unknown();
          operand_sharding.SetShardGroup(
              instruction->sharding().GetShardGroup());
          instruction->mutable_operand(0)->set_sharding(
              std::move(operand_sharding));
          return true;
        } else {
          // Otherwise store the shard group relations.
          const int64_t shard_group_id =
              instruction->sharding().GetShardGroup().shard_group_id;
          (*instruction_to_shard_group_id)[instruction] = shard_group_id;
          if (instruction->sharding().IsShardAs()) {
            auto& shard_as_group =
                (*shard_group_id_to_shard_as_group)[shard_group_id];
            if (!shard_as_group.empty()) {
              CHECK(ShapeUtil::SameDimensions(
                  instruction->shape(), (*shard_as_group.begin())->shape()))
                  << "Instruction: " << instruction->ToString()
                  << " has different shape from the shapes of the other "
                     "instructions within the same shard_as group: "
                  << (*shard_as_group.begin())->shape().ToString();
            }
            shard_as_group.insert(instruction);
          } else {
            auto& shard_like_group =
                (*shard_group_id_to_shard_like_group)[shard_group_id];
            if (!shard_like_group.empty()) {
              CHECK(ShapeUtil::SameDimensions(
                  instruction->shape(), (*shard_like_group.begin())->shape()))
                  << "Instruction: " << instruction->ToString()
                  << " has different shape from the shapes of the other "
                     "instructions within the same shard_like group: "
                  << (*shard_like_group.begin())->shape().ToString();
            }
            shard_like_group.insert(instruction);
          }
          HloSharding sharding = instruction->sharding();
          sharding.ClearShardGroup();
          instruction->set_sharding(std::move(sharding));
        }
      }
    }
    return false;
  };

  for (HloComputation* computation : module->computations(execution_threads)) {
    auto instructions = computation->MakeInstructionPostOrder();
    for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
      HloInstruction* instruction = *it;
      if (instruction->IsCustomCall("Sharding")) {
        HloSharding original_sharding = instruction->sharding();
        TF_RET_CHECK(instruction->has_sharding())
            << "Sharding instruction must have a sharding attribute";
        VLOG(3) << "ProcessShardingInstruction: " << instruction->ToString();

        std::vector<int64_t> unspec_dims;
        TF_RETURN_IF_ERROR(sharding_op_util::ParseAttributes(
            Cast<HloCustomCallInstruction>(instruction)->opaque(),
            &unspec_dims));

        bool replaced_with_copy =
            replace_sharding_with_copy &&
            (!original_sharding.IsUnknown() ||
             instruction->operand(0)->opcode() == HloOpcode::kParameter);
        // Replace the sharding instruction with a copy node so that it does not
        // need special handling.
        if (replaced_with_copy) {
          auto copy = computation->AddInstruction(HloInstruction::CreateUnary(
              instruction->shape(), HloOpcode::kCopy,
              instruction->mutable_operand(0)));
          TF_ASSIGN_OR_RETURN(
              std::ignore, computation->ReplaceInstruction(
                               instruction, copy, /*preserve_sharding=*/false,
                               /*relay_control_dependency=*/false,
                               /*remove_unused_operands=*/false));
          copy->set_sharding(std::move(original_sharding));
          instruction = copy;
          changed = true;
        }

        TF_ASSIGN_OR_RETURN(
            bool shard_group_remove_instruction,
            process_shard_group_instruction(instruction, replaced_with_copy));
        if (!unspec_dims.empty()) {
          absl::c_sort(unspec_dims);
          unspecified_dims->emplace(instruction, std::move(unspec_dims));
        } else if (!instruction->operand(0)->has_sharding()) {
          instruction->mutable_operand(0)->set_sharding(
              instruction->sharding());
        }
        if (shard_group_remove_instruction) {
          TF_ASSIGN_OR_RETURN(std::ignore,
                              computation->ReplaceInstruction(
                                  instruction, instruction->mutable_operand(0),
                                  /*preserve_sharding=*/false,
                                  /*relay_control_dependency=*/false,
                                  /*remove_unused_operands=*/false));
        }
      } else {
        TF_ASSIGN_OR_RETURN(std::ignore,
                            process_shard_group_instruction(
                                instruction, /*replaced_with_copy=*/false));
      }
    }
  }

  // Save the original shardings of parameters/outputs.
  HloInstruction* root_instr = module->entry_computation()->root_instruction();
  if (saved_root_shardings != nullptr && root_instr->shape().IsTuple() &&
      root_instr->has_sharding()) {
    saved_root_shardings->reserve(
        root_instr->sharding().tuple_elements().size());
    for (const HloSharding& sharding :
         root_instr->sharding().tuple_elements()) {
      saved_root_shardings->push_back(sharding);
    }
  }
  if (saved_parameter_shardings != nullptr) {
    auto params = module->entry_computation()->parameter_instructions();
    for (int64_t i = 0; i < params.size(); ++i) {
      if (params[i]->has_sharding()) {
        saved_parameter_shardings->insert({i, params[i]->sharding()});
      }
    }
  }
  return changed;
}

// Compute the number of users that are only internal to the computation.
int64_t ComputeNonRootUsers(const HloInstruction* instr) {
  int64_t non_root_users = instr->users().size();
  for (int i = 0; i < instr->users().size(); ++i) {
    if (instr->users()[i] == instr->parent()->root_instruction()) {
      --non_root_users;
    }
  }
  return non_root_users;
}

/*static*/ Status ShardingPropagation::NormalizeDomain(
    const DomainMetadata::Domain& domain, const DomainMetadata* metadata) {
  if (metadata != nullptr) {
    TF_ASSIGN_OR_RETURN(const auto& sharding_metadata,
                        ShardingMetadata::ToShardingMetadata(metadata));
    const auto& sharding = sharding_metadata->sharding();
    if (sharding != nullptr) {
      bool is_spatially_partitioned = !sharding->HasUniqueDevice();
      if (sharding->IsTuple()) {
        is_spatially_partitioned = absl::c_any_of(
            sharding->tuple_elements(),
            [](const HloSharding& s) { return !s.HasUniqueDevice(); });
      }
      if (is_spatially_partitioned) {
        for (HloInstruction* d : domain.exit_domains) {
          HloInstruction* operand = d->mutable_operand(0);
          // Set sharding only if it is different. We don't overwrite the
          // metadata if it has the same sharding besides metadata.
          if (!operand->has_sharding() || operand->sharding() != *sharding) {
            HloSharding operand_sharding = *sharding;
            if (operand->shape().IsTuple() && !sharding->IsTuple()) {
              // Expand sharding into tuple sharding per
              // CloneShardingForDomain() in
              // third_party/tensorflow/compiler/xla/hlo/ir/hlo_sharding_metadata.cc
              operand_sharding =
                  HloSharding::SingleTuple(operand->shape(), *sharding);
            }
            operand->set_sharding(std::move(operand_sharding));
          }
        }
        return OkStatus();
      }
    }
  }
  return ShardingMetadata::NormalizeShardingDomain(domain, metadata);
}

// Return the sharding that should be propagated from user to instruction.
std::optional<HloSharding> ShardingPropagation::GetShardingFromUser(
    const HloInstruction& instruction, const HloInstruction& user,
    int64_t aggressiveness, bool is_spmd, const CallGraph& call_graph,
    const CustomCallShardingHelper* sharding_helper) {
  if (!CanPropagateThroughAtAggressiveLevel(user, aggressiveness)) {
    return std::nullopt;
  }
  if (!IsSpatiallyPartitioned(&user)) {
    return std::nullopt;
  }
  const bool may_combine_partial_sharding = is_spmd && aggressiveness > 0;

  switch (user.opcode()) {
    case HloOpcode::kBroadcast: {
      return InferBroadcastOperandSharding(user, is_spmd);
    }
    case HloOpcode::kConcatenate: {
      if (aggressiveness == 0) {
        return std::nullopt;
      }
      if (user.sharding().IsReplicated()) {
        return user.sharding();
      }

      const int64_t cdim = user.concatenate_dimension();
      auto& tile_assignment = user.sharding().tile_assignment();
      if (tile_assignment.dim(cdim) == 1) {
        // If we are concatenating along a non-sharded dimension then the
        // operands should have the same sharding as the result.
        return user.sharding();
      }

      if (is_spmd) {
        // SPMD doesn't support tiling with part of the devices. Return the same
        // sharding.
        return user.sharding();
      }

      // If we are concatenating along a sharded dimension then we want the
      // operands to be distributed among the devices their data is used.
      int64_t start_offset = 0;
      for (HloInstruction* op : user.operands()) {
        if (op == &instruction) {
          break;
        }
        start_offset += op->shape().dimensions(cdim);
      }
      const int64_t tile_shape = CeilOfRatio(
          user.shape().dimensions(cdim), tile_assignment.dimensions()[cdim]);
      std::vector<int64_t> start_indices(tile_assignment.num_dimensions());
      std::vector<int64_t> end_indices(tile_assignment.dimensions().begin(),
                                       tile_assignment.dimensions().end());
      start_indices[cdim] = start_offset / tile_shape;
      end_indices[cdim] = CeilOfRatio(
          start_offset + instruction.shape().dimensions(cdim), tile_shape);
      auto new_tile_assignment =
          tile_assignment.array().Slice(start_indices, end_indices);
      if (new_tile_assignment.num_elements() == 1) {
        return HloSharding::AssignDevice(*new_tile_assignment.begin(),
                                         user.sharding().metadata());
      }
      return HloSharding::Tile(std::move(new_tile_assignment),
                               user.sharding().metadata());
    }
    case HloOpcode::kConvolution: {
      auto dot_dims = dot_as_convolution_util::ParseConvolutionDimsInfo(&user);
      if (dot_dims.conv_spatial_dims.empty()) {
        int64_t op_idx = user.operand_index(&instruction);
        return InferDotOperandSharding(&user, dot_dims, op_idx,
                                       may_combine_partial_sharding);
      }
      return std::nullopt;
    }
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice: {
      if (aggressiveness == 0) {
        return std::nullopt;
      }
      if (user.sharding().IsReplicated()) {
        return user.sharding();
      }
      if (user.opcode() == HloOpcode::kDynamicUpdateSlice &&
          &instruction == user.operand(0)) {
        return user.sharding();
      }
      const HloInstruction* operand = user.opcode() == HloOpcode::kDynamicSlice
                                          ? user.operand(0)
                                          : user.operand(1);
      if (&instruction != operand) {
        return std::nullopt;
      }

      if (is_spmd) {
        return user.sharding();
      }
      const auto& tile_assignment = user.sharding().tile_assignment();
      for (int64_t i = 0; i < user.shape().rank(); ++i) {
        if (tile_assignment.dim(i) > 1 &&
            user.shape().dimensions(i) != operand->shape().dimensions(i)) {
          return std::nullopt;
        }
      }
      return user.sharding();
    }
    case HloOpcode::kReduceWindow: {
      auto* reduce_window = Cast<HloReduceWindowInstruction>(&user);
      if (!absl::c_linear_search(reduce_window->inputs(), &instruction)) {
        return std::nullopt;
      }
      if (reduce_window->shape().IsTuple()) {
        auto sub_sharding = reduce_window->sharding().GetSubSharding(
            reduce_window->shape(),
            {reduce_window->operand_index(&instruction)});
        return sub_sharding;
      }
      return reduce_window->sharding();
    }
    case HloOpcode::kReshape: {
      return hlo_sharding_util::PropagateShardingThroughReshape(
          user.shape(), instruction.shape(), user.sharding());
    }
    case HloOpcode::kPad: {
      if (&instruction != user.operand(0)) {
        return std::nullopt;
      }
      return user.sharding();
    }
    case HloOpcode::kSlice: {
      return user.sharding();
    }
    case HloOpcode::kTranspose: {
      // Calculate the dimension numbers for reversing the current transpose
      // and then use TransposeSharding to convert the output sharding to an
      // input sharding.
      std::vector<int64_t> reverse_dimensions(user.dimensions().size());
      for (int64_t i = 0; i < user.dimensions().size(); ++i) {
        reverse_dimensions[user.dimensions(i)] = i;
      }
      return hlo_sharding_util::TransposeSharding(user.sharding(),
                                                  reverse_dimensions);
    }
    case HloOpcode::kTuple: {
      auto sub_sharding = user.sharding().GetSubSharding(
          user.shape(), {user.operand_index(&instruction)});
      // In case the instruction is used as the operands multiple times within
      // this tuple, we will return the most specific sharding and propagate up.
      for (int64_t i = 0; i < user.shape().tuple_shapes_size(); ++i) {
        if (user.operand(i) == &instruction) {
          // Only evaluate GetSubSharding if this operand is of interest,
          // as it is relatively expensive.
          HloSharding alternative_sub_sharding =
              user.sharding().GetSubSharding(user.shape(), {i});
          if (hlo_sharding_util::IsShardingMoreSpecific(
                  alternative_sub_sharding, sub_sharding)) {
            sub_sharding = alternative_sub_sharding;
          }
        }
      }
      return sub_sharding;
    }
    case HloOpcode::kGetTupleElement: {
      int64_t sharding_index = 0;
      for (int i = 0; i < instruction.shape().tuple_shapes_size(); ++i) {
        if (i == user.tuple_index()) {
          break;
        }
        if (instruction.shape().tuple_shapes(i).IsArray()) {
          sharding_index += 1;
        } else {
          sharding_index +=
              ShapeUtil::GetLeafCount(instruction.shape().tuple_shapes(i));
        }
      }
      if (user.shape().IsArray()) {
        // Use ReplicateAllDataDims instead of HloSharding::Replicate() to
        // preserve manual subgroups.
        HloSharding new_sharding =
            instruction.has_sharding()
                ? instruction.sharding()
                : HloSharding::SingleTuple(
                      instruction.shape(),
                      hlo_sharding_util::ReplicateAllDataDims(user.sharding()));
        new_sharding.tuple_elements()[sharding_index] = user.sharding();
        return new_sharding;
      } else {
        if (user.sharding().tuple_elements().empty()) {
          return std::nullopt;
        }
        HloSharding new_sharding =
            instruction.has_sharding()
                ? instruction.sharding()
                : HloSharding::SingleTuple(
                      instruction.shape(),
                      hlo_sharding_util::ReplicateAllDataDims(
                          user.sharding().tuple_elements()[0]));
        for (int64_t i = 0; i < user.sharding().tuple_elements().size(); ++i) {
          new_sharding.tuple_elements()[sharding_index + i] =
              user.sharding().tuple_elements()[i];
        }
        return new_sharding;
      }
    }
    case HloOpcode::kDot: {
      int64_t op_idx = user.operand_index(&instruction);
      auto dnums = dot_as_convolution_util::ParseDotGeneralFromDot(&user);
      return InferDotOperandSharding(&user, dnums, op_idx,
                                     may_combine_partial_sharding);
    }
    case HloOpcode::kReduce: {
      if (instruction.shape().rank() == 0) {
        return std::nullopt;
      }
      auto user_sharding =
          user.shape().IsTuple()
              ? user.sharding().GetSubSharding(
                    user.shape(), {user.operand_index(&instruction)})
              : user.sharding();
      if (!user_sharding.IsTileMaximal()) {
        std::vector<int64_t> target_tile_assignment_dimensions(
            instruction.shape().rank() +
            (user_sharding.ReplicateOnLastTileDim() ? 1 : 0) +
            user_sharding.subgroup_types().size());
        const auto& dimensions = user.dimensions();
        int64_t next_output_dim = 0;
        for (int64_t i = 0; i < target_tile_assignment_dimensions.size(); ++i) {
          if (absl::c_find(dimensions, i) == dimensions.end()) {
            target_tile_assignment_dimensions[i] =
                user_sharding.tile_assignment().dim(next_output_dim++);
          } else {
            target_tile_assignment_dimensions[i] = 1;
          }
        }
        auto tile_assignment = user_sharding.tile_assignment().Reshape(
            target_tile_assignment_dimensions);
        user_sharding =
            user_sharding.ReplicateOnLastTileDim()
                ? HloSharding::PartialTile(tile_assignment,
                                           user_sharding.metadata())
                : HloSharding::Subgroup(tile_assignment,
                                        user_sharding.subgroup_types(),
                                        user_sharding.metadata());
      }

      // Try to merge with sharding from other operands if they can improve
      // current sharding.
      const auto* reduce = Cast<const HloReduceInstruction>(&user);
      for (const HloInstruction* operand : reduce->inputs()) {
        if (operand != &instruction && operand->has_sharding()) {
          hlo_sharding_util::MergeShardingIfCompatible(
              operand->sharding(), user_sharding.NumTiles() + 1,
              &user_sharding);
        }
      }
      return user_sharding;
    }
    case HloOpcode::kSort: {
      HloSharding user_sharding = user.sharding();
      if (user_sharding.IsTuple()) {
        return user_sharding.GetSubSharding(user.shape(),
                                            {user.operand_index(&instruction)});
      }
      return user_sharding;
    }
    case HloOpcode::kReverse: {
      return hlo_sharding_util::ReverseSharding(user.sharding(),
                                                user.dimensions());
    }
    case HloOpcode::kOutfeed: {
      if (&instruction != user.operand(0)) {
        return std::nullopt;
      }
      std::vector<Shape> operand_shapes(user.operand_count());
      for (int i = 0; i < user.operand_count(); ++i) {
        operand_shapes[i] = user.operand(i)->shape();
      }
      return user.sharding().GetSubSharding(
          ShapeUtil::MakeTupleShape(operand_shapes), {0});
    }
    case HloOpcode::kGather: {
      if (&instruction == user.operand(1)) {
        return hlo_sharding_util::
            GatherIndexShardingFromOutputIndexPassthroughDimensions(
                user.sharding(), &user);
      }
      if (is_spmd) {
        return hlo_sharding_util::GatherOperandShardingFromOutput(
            user.sharding(), user, call_graph);
      }
      return std::nullopt;
    }
    case HloOpcode::kScatter: {
      auto& scatter_user = *Cast<HloScatterInstruction>(&user);
      const int64_t operand_count = scatter_user.scatter_operand_count();
      auto scatter_operands = scatter_user.scatter_operands();
      auto scatter_indices = scatter_user.scatter_indices();
      auto scatter_updates = scatter_user.scatter_updates();
      // Infer sharding for scatter operand.
      const int64_t operand_index =
          absl::c_find(scatter_operands, &instruction) -
          scatter_operands.cbegin();
      if (operand_index < operand_count) {
        return user.sharding().IsTuple() ? user.sharding().GetSubSharding(
                                               user.shape(), {operand_index})
                                         : user.sharding();
      }
      // Infer sharding for scatter indices.
      if (&instruction == scatter_indices) {
        std::vector<const HloInstruction*> partitioned_updates;
        for (const HloInstruction* update : scatter_updates) {
          if (IsSpatiallyPartitioned(update)) {
            partitioned_updates.push_back(update);
          }
        }
        if (partitioned_updates.empty()) {
          return std::nullopt;
        }
        std::vector<HloSharding> shardings;
        absl::c_transform(
            partitioned_updates, std::back_inserter(shardings),
            [&scatter_user](const HloInstruction* update) {
              return hlo_sharding_util::
                  ScatterIndexShardingFromUpdateIndexPassthroughDimensions(
                      update->sharding(), &scatter_user);
            });
        return hlo_sharding_util::FindCommonSharding(shardings);
      }
      // Infer sharding for scatter update.
      const int64_t update_index = absl::c_find(scatter_updates, &instruction) -
                                   scatter_updates.cbegin();
      CHECK_LE(update_index, operand_count);
      auto from_indices =
          IsSpatiallyPartitioned(scatter_indices)
              ? hlo_sharding_util::
                    ScatterUpdateShardingFromIndexIndexPassthroughDimensions(
                        scatter_indices->sharding(), &scatter_user)
              : HloSharding::Replicate();
      if (is_spmd) {
        auto from_output = hlo_sharding_util::ScatterUpdateShardingFromOutput(
            user.sharding().IsTuple()
                ? user.sharding().GetSubSharding(user.shape(), {update_index})
                : user.sharding(),
            scatter_user, call_graph);
        if (from_output.has_value()) {
          // Use sharding from output as primary sharding since it prioritize
          // parallel sharding first as this is how it is in spmd_partitioner.
          hlo_sharding_util::MergeShardingIfCompatible(
              from_indices, from_output->NumTiles() + 1, &*from_output);
          if (!from_output->IsTileMaximal()) {
            return from_output;
          }
        }
      }
      if (!from_indices.IsTileMaximal()) {
        return from_indices;
      }
      return std::nullopt;
    }
    case HloOpcode::kCustomCall: {
      bool compatible_shapes = ShapeUtil::CompatibleIgnoringElementType(
          instruction.shape(), user.shape());
      if (!compatible_shapes) {
        // Incompatible shapes, we will not propagate sharding.
        return std::nullopt;
      }
      if (!sharding_helper) {
        // No available sharding helper and shapes are compatible, we will
        // propagate sharding.
        return user.sharding();
      }
      if (sharding_helper->CanPropagateShardingToOperands(&user)) {
        return user.sharding();
      }
      return std::nullopt;
    }
    default: {
      // If the user output shape is compatible with the current instruction
      // shape excluding element type and the current instruction is supported
      // by spatial partitioning, then the user sharding can be used for
      // propagation to the current instruction.
      if (ShapeUtil::CompatibleIgnoringElementType(instruction.shape(),
                                                   user.shape())) {
        return user.sharding();
      }
      return std::nullopt;
    }
  }
}

// Only pass through sharding annotation at the first iteration when:
//  1. Operand is sharded;  2. Only non-concat dim is sharded;
//  3. Concat is for params in the repeated layers which follows the
//     pattern of param/gte -> reshape -> concat.
bool AggressiveConcatOperandShardingCanPassThrough(
    const HloInstruction* concat_operand) {
  return (
      IsSpatiallyPartitioned(concat_operand) &&
      (concat_operand->has_sharding() &&
       concat_operand->sharding().NumTiles() > 1) &&
      concat_operand->opcode() == HloOpcode::kReshape &&
      (concat_operand->operand(0)->opcode() == HloOpcode::kParameter ||
       concat_operand->operand(0)->opcode() == HloOpcode::kGetTupleElement));
}

// DynamicSlice or DynamicUpdateSlice handling for InferShardingFromOperands().
bool InferDynamicSliceOrDynamicUpdateSliceShardingFromOperands(
    HloInstruction* instruction, int64_t aggressiveness,
    bool may_combine_partial_sharding) {
  const HloInstruction* operand =
      instruction->opcode() == HloOpcode::kDynamicSlice
          ? instruction->operand(0)
          : instruction->operand(1);
  auto slice_dim_is_sharded = [&]() {
    if (!IsSpatiallyPartitioned(operand) || operand->sharding().IsManual() ||
        operand->sharding().NumTiles() == 1) {
      return false;
    }
    for (int64_t i = 0; i < instruction->shape().rank(); ++i) {
      const auto& tile_assignment = operand->sharding().tile_assignment();
      if (tile_assignment.dim(i) > 1 && instruction->shape().dimensions(i) !=
                                            operand->shape().dimensions(i)) {
        return true;
      }
    }
    return false;
  };

  // Do not pass through sharding annotation at the first iteration
  // if slice dim is sharded.
  if (aggressiveness == 0 && slice_dim_is_sharded()) {
    return false;
  }

  auto propagate_slicing = [&]() {
    if (!IsSpatiallyPartitioned(operand)) {
      return false;
    }
    if (slice_dim_is_sharded()) {
      return false;
    }
    return MaybeImproveInstructionSharding(
        operand->sharding(), instruction, may_combine_partial_sharding,
        /*allow_aggressive_resharding=*/
        ComputeNonRootUsers(instruction) == 1);
  };
  auto propagate_base = [&]() {
    if (instruction->opcode() != HloOpcode::kDynamicUpdateSlice) {
      return false;
    }
    if (!IsSpatiallyPartitioned(instruction->operand(0))) {
      return false;
    }
    return MaybeImproveInstructionSharding(instruction->operand(0)->sharding(),
                                           instruction,
                                           may_combine_partial_sharding);
  };
  bool changed = propagate_slicing();
  changed |= propagate_base();
  return changed;
}

bool ShardingPropagation::InferShardingFromShardGroup(
    HloInstruction* instruction, const ComputationMap& computation_map,
    int64_t aggressiveness,
    const absl::flat_hash_set<HloInstruction*>& shard_group) {
  if (!CanPropagateThroughAtAggressiveLevel(*instruction, aggressiveness)) {
    return false;
  }
  // Do not change manual sharding.
  if (instruction->has_sharding() && instruction->sharding().IsManual()) {
    return false;
  }
  // Do not propagate sharding to ShardBarrierTo custom-call.
  if (instruction->IsCustomCall(spmd::kShardBarrierTo)) {
    return false;
  }
  // Propagate manual sharding.
  if (!instruction->has_sharding() || instruction->sharding().IsTileMaximal()) {
    for (const HloInstruction* member : shard_group) {
      if (!member->has_sharding() || !member->sharding().IsManual() ||
          member == instruction) {
        continue;
      }
      instruction->set_sharding(member->sharding());
      return true;
    }
  }

  const bool may_combine_partial_sharding = is_spmd_ && aggressiveness > 0;
  bool changed = false;
  for (const HloInstruction* member : shard_group) {
    // Do not propagate sharding from ShardBarrierFrom custom-call.
    if (member == instruction ||
        member->IsCustomCall(spmd::kShardBarrierFrom)) {
      continue;
    }
    changed |= MaybeImproveInstructionSharding(member->sharding(), instruction,
                                               may_combine_partial_sharding);
  }
  return changed;
}

// Tries to update the sharding of the specified instruction based on its
// operands and returns true if the sharding of the instruction have been
// changed and false otherwise.
bool ShardingPropagation::InferShardingFromOperands(
    HloInstruction* instruction, const ComputationMap& computation_map,
    int64_t aggressiveness, const CallGraph& call_graph,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!CanPropagateThroughAtAggressiveLevel(*instruction, aggressiveness)) {
    return false;
  }
  // Do not change manual sharding.
  if (instruction->has_sharding() && instruction->sharding().IsManual()) {
    return false;
  }
  // Propagate manual sharding. Avoid tuple shaped HLOs that group independent
  // together. Reduce, ReduceWindow, and Sort can be tuples but the elements
  // are correlated, so we propagate manual sharding through them.

  // For custom-calls with manual operand, the default propagation logic will
  // just assign manual to the whole custom-call.
  const bool custom_call_condition =
      instruction->opcode() == HloOpcode::kCustomCall &&
      instruction->shape().IsTuple();
  // For asynchronous instructions with manual operand, we assign manual to the
  // whole instructions if the async_execution_thread is not in the
  // execution_threads.
  const bool async_instr_condition =
      instruction->IsAsynchronous() &&
      !HloInstruction::IsThreadIncluded(instruction->async_execution_thread(),
                                        execution_threads);

  if ((!instruction->has_sharding() ||
       instruction->sharding().IsTileMaximal()) &&
      (instruction->shape().IsArray() ||
       instruction->opcode() == HloOpcode::kReduce ||
       instruction->opcode() == HloOpcode::kSort ||
       instruction->opcode() == HloOpcode::kReduceWindow ||
       custom_call_condition || async_instr_condition)) {
    for (const HloInstruction* op : instruction->operands()) {
      if (!op->has_sharding() || !op->sharding().IsManual()) continue;
      // Do not pass through manual sharding to SPMDShardToFullShape.
      if (instruction->IsCustomCall("SPMDShardToFullShape")) {
        return false;
      }
      // Do not pass through manual sharding to concat or dynamic slice when
      // aggressiveneess is 0.
      if (aggressiveness == 0 &&
          (instruction->opcode() == HloOpcode::kConcatenate ||
           instruction->opcode() == HloOpcode::kDynamicSlice)) {
        return false;
      }
      instruction->set_sharding(
          HloSharding::Manual(op->sharding().metadata())
              .NormalizeTupleSharding(instruction->shape()));
      return true;
    }
  }
  const bool may_combine_partial_sharding = is_spmd_ && aggressiveness > 0;
  if (!SupportSpatialPartitioning(
          instruction, computation_map, is_spmd_,
          allow_spmd_sharding_propagation_to_output_,
          /*allow_spmd_sharding_propagation_to_parameters=*/false,
          sharding_helper_.get())) {
    // If an array shaped HLO doesn't support spatial partitioning but at least
    // one of its operand is replicated then we make the HLO replicated as well.
    if (instruction->shape().IsTuple() || instruction->operand_count() == 0 ||
        instruction == instruction->parent()->root_instruction() ||
        instruction->HasSideEffect()) {
      return false;
    }
    for (const HloInstruction* op : instruction->operands()) {
      if (op->has_sharding() && op->sharding().IsTileMaximal() &&
          !op->sharding().HasUniqueDevice()) {
        return MaybeImproveInstructionSharding(op->sharding(), instruction,
                                               may_combine_partial_sharding);
      }
    }
    return false;
  }
  auto get_maybe_tuple_sharding = [&](HloSharding sharding) {
    if (instruction->shape().IsArray()) {
      return sharding;
    }
    std::vector<HloSharding> tuple(instruction->shape().tuple_shapes_size(),
                                   std::move(sharding));
    return HloSharding::Tuple(instruction->shape(), tuple);
  };
  switch (instruction->opcode()) {
    case HloOpcode::kGetTupleElement: {
      const HloInstruction* operand = instruction->operand(0);
      if (!IsSpatiallyPartitioned(operand)) {
        return false;
      }
      HloSharding new_sharding = operand->sharding().GetSubSharding(
          operand->shape(), {instruction->tuple_index()});
      if (new_sharding.IsManual()) {
        instruction->set_sharding(std::move(new_sharding));
        return true;
      }
      return MaybeImproveInstructionSharding(
          std::move(new_sharding), instruction, may_combine_partial_sharding,
          /*allow_aggressive_resharding=*/
          ComputeNonRootUsers(instruction) == 1);
    }
    case HloOpcode::kTuple: {
      if (absl::c_none_of(instruction->operands(),
                          [](const HloInstruction* hlo) {
                            return IsSpatiallyPartitioned(hlo);
                          })) {
        // None of the operands have a spatially partitioned sharding.
        return false;
      }
      const Shape& shape = instruction->shape();
      // Go through each operand and if the operand has a sharding that is
      // better than the current sharding for that tuple element then update
      // it. If the current sharding does not exist, assume its replicated.
      std::vector<HloSharding> sub_shardings;
      if (instruction->has_sharding()) {
        sub_shardings = instruction->sharding().tuple_elements();
      } else {
        // If instruction does not have a sharding, assume its replicated to
        // allow refinement.
        sub_shardings.assign(HloSharding::RequiredLeaves(shape),
                             HloSharding::Replicate());
      }
      // This is required to allow manual sharding on operands to be propagated
      // to the tuple. hlo_sharding_util::IsShardingMoreSpecific() returns false
      // if any of the shardings involved is manual, so using it directly will
      // prevent manual sharding on an operand to be propagated to the tuple
      // when it has no existing sharding.
      auto is_more_specific = [instruction](const HloSharding& operand_sharding,
                                            const HloSharding& existing) {
        // If the instruction originally had no sharding, always prefer operand
        // sharding.
        return !instruction->has_sharding() ||
               hlo_sharding_util::IsShardingMoreSpecific(operand_sharding,
                                                         existing);
      };

      int64_t sub_sharding_index = 0;
      for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
        const HloInstruction* operand = instruction->operand(i);
        if (operand->has_sharding()) {
          if (operand->shape().IsTuple()) {
            for (int64_t j = 0, e = ShapeUtil::GetLeafCount(operand->shape());
                 j < e; ++j) {
              if (is_more_specific(operand->sharding().tuple_elements()[j],
                                   sub_shardings[sub_sharding_index + j])) {
                sub_shardings[sub_sharding_index + j] =
                    operand->sharding().tuple_elements()[j];
              }
            }
          } else {
            std::optional<HloSharding> op_sharding =
                hlo_sharding_util::GetOutputSharding(operand);
            CHECK(op_sharding.has_value())
                << "Expected sharding for " << operand->ToString();
            if (is_more_specific(op_sharding.value(),
                                 sub_shardings[sub_sharding_index])) {
              sub_shardings[sub_sharding_index] = op_sharding.value();
            }
          }
        }
        sub_sharding_index += ShapeUtil::GetLeafCount(operand->shape());
      }

      HloSharding new_sharding = HloSharding::Tuple(shape, sub_shardings);
      if (!instruction->has_sharding() ||
          new_sharding != instruction->sharding()) {
        instruction->set_sharding(std::move(new_sharding));
        return true;
      }
      return false;
    }
    case HloOpcode::kReduce: {
      // Reduce could have a tuple shape, where the first half of operands are
      // the arrays to reduce, and the second half of operands are the init
      // values.
      return InferReduceShardingFromOperand(
          instruction, may_combine_partial_sharding, is_spmd_);
    }
    case HloOpcode::kBroadcast: {
      // Make forward propagation through broadcast low priority to avoid
      // resharding after broadcast.
      if (aggressiveness < 3) {
        return false;
      }
      const HloInstruction* op = instruction->operand(0);
      if (!IsSpatiallyPartitioned(op) || op->sharding().IsReplicated()) {
        return false;
      }
      // The output will be tiled along the broadcasted dimension the same way
      // as the input for the broadcast while the other dimensions are kept
      // non-tiled.
      std::vector<int64_t> target_tile_assignment_dimensions;
      const auto& dimensions = instruction->dimensions();
      for (int64_t i = 0; i < instruction->shape().rank(); ++i) {
        auto it = absl::c_find(dimensions, i);
        if (it == dimensions.end()) {
          target_tile_assignment_dimensions.push_back(1);
        } else {
          const int64_t source_dim = std::distance(dimensions.begin(), it);
          target_tile_assignment_dimensions.push_back(
              op->sharding().tile_assignment().dim(source_dim));
        }
      }
      for (int64_t i = op->sharding().TiledDataRank();
           i < op->sharding().tile_assignment().num_dimensions(); ++i) {
        target_tile_assignment_dimensions.push_back(
            op->sharding().tile_assignment().dim(i));
      }
      auto new_tile_assignment = op->sharding().tile_assignment().Reshape(
          target_tile_assignment_dimensions);
      HloSharding new_sharding =
          op->sharding().ReplicateOnLastTileDim()
              ? HloSharding::PartialTile(new_tile_assignment,
                                         op->sharding().metadata())
              : HloSharding::Subgroup(new_tile_assignment,
                                      op->sharding().subgroup_types(),
                                      op->sharding().metadata());
      return MaybeImproveInstructionSharding(
          std::move(new_sharding), instruction, may_combine_partial_sharding,
          /*allow_aggressive_resharding=*/ComputeNonRootUsers(instruction) ==
              1);
    }
    case HloOpcode::kConcatenate: {
      const HloInstruction* operand = PickRepresentativeOperand(instruction);
      if (!operand || !IsSpatiallyPartitioned(operand)) {
        return false;
      }

      if (aggressiveness == 0) {
        for (const HloInstruction* concat_operand : instruction->operands()) {
          if (!AggressiveConcatOperandShardingCanPassThrough(concat_operand)) {
            return false;
          }
          const auto& tile_assignment =
              concat_operand->sharding().tile_assignment();
          for (int64_t i = 0; i < instruction->shape().rank(); ++i) {
            if (absl::c_linear_search(instruction->dimensions(), i) &&
                tile_assignment.dim(i) > 1) {
              return false;
            }
          }
        }
      }
      return MaybeImproveInstructionSharding(
          operand->sharding(), instruction, may_combine_partial_sharding,
          /*allow_aggressive_resharding=*/ComputeNonRootUsers(instruction) ==
              1);
    }
    case HloOpcode::kConvolution:
      return InferConvolutionShardingFromOperands(
          instruction, call_graph, aggressiveness, may_combine_partial_sharding,
          is_spmd_);
    case HloOpcode::kTranspose: {
      const HloInstruction* input = instruction->operand(0);
      if (!IsSpatiallyPartitioned(input)) {
        return false;
      }
      HloSharding sharding = hlo_sharding_util::TransposeSharding(
          input->sharding(), instruction->dimensions());
      return MaybeImproveInstructionSharding(
          std::move(sharding), instruction, may_combine_partial_sharding,
          /*allow_aggressive_resharding=*/ComputeNonRootUsers(instruction) ==
              1);
    }
    case HloOpcode::kReduceWindow: {
      auto* reduce_window = Cast<HloReduceWindowInstruction>(instruction);
      auto has_dilation = [](const WindowDimension& dimensions) {
        return dimensions.base_dilation() > 1 ||
               dimensions.window_dilation() > 1;
      };
      if (absl::c_any_of(instruction->window().dimensions(), has_dilation)) {
        VLOG(2) << "Not applying sharding to reduce window because dilatation "
                   "isn't supported yet: "
                << reduce_window->ToString();
        return false;
      }
      bool changed = false;
      for (HloInstruction* operand : reduce_window->inputs()) {
        if (!IsSpatiallyPartitioned(operand)) {
          continue;
        }
        changed |= MaybeImproveInstructionSharding(
            get_maybe_tuple_sharding(operand->sharding()), reduce_window,
            may_combine_partial_sharding,
            /*allow_aggressive_resharding=*/
            ComputeNonRootUsers(instruction) == 1);
      }
      return changed;
    }
    case HloOpcode::kSelectAndScatter: {
      // Shard according to first operand, as output keeps the same shape.
      const HloInstruction* lhs = instruction->operand(0);
      if (!IsSpatiallyPartitioned(lhs)) {
        return false;
      }

      auto has_base_dilation = [](const WindowDimension& dimensions) {
        return dimensions.base_dilation() > 1;
      };
      if (absl::c_any_of(instruction->window().dimensions(),
                         has_base_dilation)) {
        VLOG(2) << "Not applying sharding to select-and-scatter because "
                   "base dilation isn't supported yet: "
                << instruction->ToString();
        return false;
      }
      return MaybeImproveInstructionSharding(
          lhs->sharding(), instruction, may_combine_partial_sharding,
          /*allow_aggressive_resharding=*/ComputeNonRootUsers(instruction) ==
              1);
    }
    case HloOpcode::kReshape: {
      if (!IsSpatiallyPartitioned(instruction->operand(0))) {
        return false;
      }
      HloSharding new_sharding =
          hlo_sharding_util::PropagateShardingThroughReshape(
              instruction->operand(0)->shape(), instruction->shape(),
              instruction->operand(0)->sharding());
      return MaybeImproveInstructionSharding(
          std::move(new_sharding), instruction, may_combine_partial_sharding,
          /*allow_aggressive_resharding=*/
          ComputeNonRootUsers(instruction) == 1);
      return false;
    }
    case HloOpcode::kReverse: {
      const HloInstruction* operand = instruction->operand(0);
      if (!IsSpatiallyPartitioned(operand)) {
        return false;
      }
      return MaybeImproveInstructionSharding(
          hlo_sharding_util::ReverseSharding(operand->sharding(),
                                             instruction->dimensions()),
          instruction, may_combine_partial_sharding,
          /*allow_aggressive_resharding=*/ComputeNonRootUsers(instruction) ==
              1);
    }
    case HloOpcode::kDot: {
      const auto& dnums =
          dot_as_convolution_util::ParseDotGeneralFromDot(instruction);
      return InferDotShardingFromOperands(instruction, call_graph, dnums,
                                          may_combine_partial_sharding,
                                          is_spmd_);
    }
    case HloOpcode::kParameter: {
      auto parent_it = computation_map.find(instruction->parent());
      if (parent_it == computation_map.end()) {
        return false;
      }
      const HloInstruction* parent = parent_it->second;
      switch (parent->opcode()) {
        case HloOpcode::kConditional: {
          for (int64_t i = 1; i < parent->operand_count(); ++i) {
            if (parent->called_computations()[i - 1] == instruction->parent()) {
              if (parent->operand(i)->has_sharding()) {
                return MaybeImproveInstructionSharding(
                    parent->operand(i)->sharding(), instruction,
                    may_combine_partial_sharding);
              }
              return false;
            }
          }
          return false;
        }
        case HloOpcode::kCall: {
          int64_t i = instruction->parameter_number();
          if (parent->operand(i)->has_sharding()) {
            return MaybeImproveInstructionSharding(
                parent->operand(i)->sharding(), instruction,
                may_combine_partial_sharding);
          }
          return false;
        }
        default:
          return false;
      }
    }
    case HloOpcode::kSort: {
      const HloInstruction* operand = PickRepresentativeOperand(instruction);
      if (!operand || !IsSpatiallyPartitioned(operand)) {
        return false;
      }
      HloSortInstruction* sort = DynCast<HloSortInstruction>(instruction);
      CHECK(sort);
      const int64_t sort_dim = sort->sort_dimension();
      if (!operand->sharding().IsTileMaximal() &&
          operand->sharding().tile_assignment().dim(sort_dim) != 1) {
        // In case of a sort operand sharded along the sort dimension, the
        // sharding is propagated only if there exists a free (unsharded)
        // dimension that we can later move the sharding into.
        if (!hlo_sharding_util::IsSortOperandShardingMovable(operand, sort_dim))
          return false;
      }

      if (instruction->shape().IsTuple()) {
        return MaybeImproveInstructionSharding(
            HloSharding::SingleTuple(instruction->shape(), operand->sharding()),
            instruction, may_combine_partial_sharding,
            /*allow_aggressive_resharding=*/
            ComputeNonRootUsers(instruction) == 1);
      } else {
        return MaybeImproveInstructionSharding(
            operand->sharding(), instruction, may_combine_partial_sharding,
            /*allow_aggressive_resharding=*/
            ComputeNonRootUsers(instruction) == 1);
      }
    }
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice: {
      return InferDynamicSliceOrDynamicUpdateSliceShardingFromOperands(
          instruction, aggressiveness, may_combine_partial_sharding);
    }
    case HloOpcode::kGather: {
      bool changed = false;
      if (IsSpatiallyPartitioned(instruction->operand(1))) {
        HloSharding new_sharding = hlo_sharding_util::
            GatherOutputShardingFromIndexIndexPassthroughDimensions(
                instruction->operand(1)->sharding(), instruction);
        changed |= MaybeImproveInstructionSharding(
            std::move(new_sharding), instruction, may_combine_partial_sharding);
      }
      if (is_spmd_) {
        auto gather_parallel_dims =
            hlo_sharding_util::GetGatherParallelBatchDims(*instruction,
                                                          call_graph);
        if (gather_parallel_dims) {
          changed |= InferGatherParallelShardingFromOperands(
              instruction, *gather_parallel_dims, may_combine_partial_sharding);
        }
        if (IsSpatiallyPartitioned(instruction->operand(0))) {
          absl::Span<const int64_t> operand_parallel_dims;
          if (gather_parallel_dims) {
            operand_parallel_dims = absl::MakeConstSpan(
                gather_parallel_dims->operand_parallel_dims);
          }
          HloSharding filtered_operand_sharding =
              hlo_sharding_util::PartiallyReplicateTiledShardingOnDims(
                  instruction->operand(0)->sharding(), operand_parallel_dims);
          auto maybe_from_data = hlo_sharding_util::
              GatherOutputShardingFromOperandOperandPassthroughDimensions(
                  filtered_operand_sharding, *instruction);
          if (maybe_from_data) {
            changed |= MaybeImproveInstructionSharding(
                std::move(*maybe_from_data), instruction,
                may_combine_partial_sharding);
          }
        }
      }
      return changed;
    }
    case HloOpcode::kScatter: {
      auto& scatter = *Cast<HloScatterInstruction>(instruction);
      const int64_t operand_count = scatter.scatter_operand_count();
      auto scatter_operands = scatter.scatter_operands();
      auto scatter_indices = scatter.scatter_indices();
      auto scatter_updates = scatter.scatter_updates();
      bool changed = false;
      if (is_spmd_) {
        for (int64_t i = 0; i != operand_count; ++i) {
          if (IsSpatiallyPartitioned(scatter_operands[i])) {
            changed |= MaybeImproveInstructionSubSharding(
                scatter_operands[i]->sharding(), instruction, {i},
                may_combine_partial_sharding);
          }
        }
        if (!IsSpatiallyPartitioned(scatter_indices) &&
            absl::c_none_of(scatter_updates, [](const HloInstruction* update) {
              return IsSpatiallyPartitioned(update);
            })) {
          return changed;
        }
        auto scatter_parallel_dims =
            hlo_sharding_util::GetScatterParallelBatchDims(*instruction,
                                                           call_graph);
        if (scatter_parallel_dims) {
          changed |= InferScatterParallelShardingFromOperands(
              instruction, *scatter_parallel_dims,
              may_combine_partial_sharding);
        }
        for (int64_t i = 0; i != operand_count; ++i) {
          if (IsSpatiallyPartitioned(scatter_updates[i])) {
            auto maybe_from_update =
                hlo_sharding_util::ScatterOutputShardingFromUpdate(
                    scatter_updates[i]->sharding(), scatter);
            if (maybe_from_update) {
              changed |= MaybeImproveInstructionSubSharding(
                  std::move(*maybe_from_update), instruction, {i},
                  may_combine_partial_sharding);
            }
          }
        }
      } else {
        for (int64_t i = 0; i != operand_count; ++i) {
          changed |= MaybeImproveInstructionSubSharding(
              HloSharding::Replicate(), instruction, {i},
              may_combine_partial_sharding);
        }
      }
      return changed;
    }
    case HloOpcode::kWhile: {
      if (!instruction->operand(0)->has_sharding()) {
        return false;
      }
      auto sharding = instruction->operand(0)->sharding();
      if (instruction->has_sharding()) {
        hlo_sharding_util::MergeSharding(instruction->sharding(), &sharding,
                                         may_combine_partial_sharding);
      }
      return MaybeImproveInstructionSharding(std::move(sharding), instruction,
                                             may_combine_partial_sharding);
    }
    case HloOpcode::kCustomCall: {
      HloSharding inferred_operand_sharding = HloSharding::Replicate();
      if (auto* partitioner =
              GetCustomCallPartitioner(instruction->custom_call_target());
          partitioner && partitioner->IsCustomCallShardable(instruction)) {
        if (auto sharding =
                partitioner->InferShardingFromOperands(instruction)) {
          inferred_operand_sharding = *sharding;
        } else {
          return false;
        }
      } else if (sharding_helper_->IsCustomCallShardable(instruction)) {
        if (auto sharding =
                sharding_helper_->InferShardingFromOperands(instruction)) {
          inferred_operand_sharding = *sharding;
        } else {
          return false;
        }
      } else {
        const HloInstruction* operand = PickRepresentativeOperand(instruction);
        if (!operand || !IsSpatiallyPartitioned(operand)) {
          return false;
        }
        inferred_operand_sharding = operand->sharding();
      }
      return MaybeImproveInstructionSharding(
          inferred_operand_sharding, instruction, may_combine_partial_sharding,
          /*allow_aggressive_resharding=*/ComputeNonRootUsers(instruction) ==
              1);
    }
    default: {
      if (instruction->IsElementwise() && may_combine_partial_sharding) {
        bool changed = false;
        for (auto operand : instruction->operands()) {
          if (IsSpatiallyPartitioned(operand)) {
            if (instruction->opcode() == HloOpcode::kRng) {
              // Rng is considered elementwise but has operands with different
              // shapes.
              changed |= MaybeImproveInstructionSharding(
                  hlo_sharding_util::ReplicateAllDataDims(
                      operand->sharding(), instruction->shape().rank()),
                  instruction, may_combine_partial_sharding,
                  ComputeNonRootUsers(instruction) == 1);
              continue;
            }
            changed |= MaybeImproveInstructionSharding(
                operand->sharding(), instruction, may_combine_partial_sharding,
                /*allow_aggressive_resharding=*/
                instruction->operands().size() == 1 &&
                    ComputeNonRootUsers(instruction) == 1);
          }
        }
        return changed;
      }
      const HloInstruction* operand = PickRepresentativeOperand(instruction);
      if (!operand || !IsSpatiallyPartitioned(operand)) {
        return false;
      }
      return MaybeImproveInstructionSharding(
          operand->sharding(), instruction, may_combine_partial_sharding,
          /*allow_aggressive_resharding=*/ComputeNonRootUsers(instruction) ==
              1);
    }
  }
  return false;
}  // NOLINT(readability/fn_size)

// Tries to update the sharding of the specified instruction based on its
// users and returns true if the sharding of the instruction have been changed
// and false otherwise.
bool ShardingPropagation::InferShardingFromUsers(
    HloInstruction* instruction,
    const ShardingPropagation::ComputationMap& computation_map,
    int64_t aggressiveness, bool is_spmd,
    const CustomCallShardingHelper* sharding_helper,
    const CallGraph& call_graph) {
  if (aggressiveness < 2 && instruction->opcode() == HloOpcode::kBroadcast) {
    return false;
  }
  // Do not change manual sharding.
  if (instruction->has_sharding() && instruction->sharding().IsManual()) {
    return false;
  }
  // Propagate manual sharding.
  if (!instruction->has_sharding() || instruction->sharding().IsTileMaximal()) {
    for (const HloInstruction* user : instruction->users()) {
      if (!user->has_sharding() || user->IsCustomCall("SPMDFullToShardShape"))
        continue;
      if (instruction->shape().IsArray() && user->sharding().IsManual()) {
        instruction->set_sharding(
            HloSharding::Manual(user->sharding().metadata()));
        return true;
      } else {
        std::optional<HloSharding> user_sharding =
            ShardingPropagation::GetShardingFromUser(
                *instruction, *user, aggressiveness, is_spmd, call_graph,
                sharding_helper);
        if (user_sharding && user_sharding->IsManual()) {
          instruction->set_sharding(std::move(*user_sharding));
          return true;
        }
      }
    }
  }

  if (!SupportSpatialPartitioning(
          instruction, computation_map, is_spmd,
          /*allow_spmd_sharding_propagation_to_output=*/false,
          allow_spmd_sharding_propagation_to_parameters_, sharding_helper)) {
    return false;
  }
  bool improved_sharding = false;
  const bool may_combine_partial_sharding = is_spmd && aggressiveness > 0;
  for (const HloInstruction* user : instruction->users()) {
    std::optional<HloSharding> user_sharding =
        ShardingPropagation::GetShardingFromUser(*instruction, *user,
                                                 aggressiveness, is_spmd,
                                                 call_graph, sharding_helper);
    if (user_sharding && instruction->opcode() == HloOpcode::kCustomCall) {
      if (auto* partitioner =
              GetCustomCallPartitioner(instruction->custom_call_target())) {
        if (partitioner->IsCustomCallShardable(instruction)) {
          user_sharding = partitioner->PropagateUserSharding(instruction, user,
                                                             *user_sharding);
        }
      } else if (sharding_helper->IsCustomCallShardable(instruction)) {
        user_sharding = sharding_helper->PropagateUserSharding(
            instruction, user, *user_sharding);
      }
    }
    if (user_sharding) {
      improved_sharding |= MaybeImproveInstructionSharding(
          std::move(*user_sharding), instruction, may_combine_partial_sharding);
    }
  }
  return improved_sharding;
}

Status SetParameterShapes(
    HloModule* module, const std::vector<Shape>& parameter_shapes,
    const std::vector<bool>&
        allow_spmd_sharding_propagation_to_parameters_vector) {
  for (int64_t i = 0; i < module->entry_computation()->num_parameters(); ++i) {
    if (!allow_spmd_sharding_propagation_to_parameters_vector[i]) {
      continue;
    }
    TF_RETURN_IF_ERROR(module->mutable_config()
                           .mutable_entry_computation_layout()
                           ->mutable_parameter_layout(i)
                           ->CopyLayoutFromShape(parameter_shapes[i]));
  }
  return OkStatus();
}

Status SetResultShape(HloModule* module, const Shape& result_shape) {
  if (!module->entry_computation_layout().LayoutIsSet() ||
      !module->entry_computation_layout().result_layout().LayoutIsSet()) {
    return OkStatus();
  }
  TF_RETURN_IF_ERROR(module->mutable_config()
                         .mutable_entry_computation_layout()
                         ->mutable_result_layout()
                         ->CopyLayoutFromShape(result_shape));
  return OkStatus();
}

Status ShardingPropagation::CanonicalizeLayouts(HloModule* module) {
  if (!allow_spmd_sharding_propagation_to_output_ &&
      !allow_spmd_sharding_propagation_to_parameters_) {
    return OkStatus();
  }
  if (!module->layout_canonicalization_callback()) {
    LOG(INFO) << "There is no registered layout_canonicalization_callback.";
    return OkStatus();
  }
  TF_ASSIGN_OR_RETURN(auto shapes_with_layout,
                      module->layout_canonicalization_callback()(*module));
  if (allow_spmd_sharding_propagation_to_parameters_) {
    TF_RETURN_IF_ERROR(SetParameterShapes(
        module, shapes_with_layout.first,
        allow_spmd_sharding_propagation_to_parameters_vector_));
  }
  if (allow_spmd_sharding_propagation_to_output_) {
    TF_RETURN_IF_ERROR(SetResultShape(module, shapes_with_layout.second));
  }
  return OkStatus();
}

absl::StatusOr<bool> ShardingPropagation::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::optional<absl::flat_hash_map<const HloInstruction*, HloSharding>>
      original_sharding;
  bool any_changed = false;
  // Preprocessing for CSE prevention propagation: record the original shardings
  // so that we can revert to them at the end, and only keep those on CSE
  // prevention instructions.
  if (cse_prevention_only_) {
    original_sharding.emplace();
    for (auto computation : module->computations(execution_threads)) {
      for (auto instruction : computation->instructions()) {
        if (instruction->has_sharding()) {
          original_sharding->emplace(instruction, instruction->sharding());
        }
      }
    }
  } else {
    // The current pass is not for CSE prevention, but we remove the shardings
    // added by previous passes for CSE prevention.
    for (auto computation : module->computations(execution_threads)) {
      for (auto instruction : computation->instructions()) {
        if (instruction->has_sharding() &&
            IsCSEPreventionSharding(instruction->sharding())) {
          instruction->clear_sharding();
          any_changed = true;
        }
      }
    }
  }
  any_changed |= propagate_metadata_
                     ? AssignShardingMetadata(module, execution_threads)
                     : RemoveShardingMetadata(module, execution_threads);
  absl::flat_hash_map<const HloInstruction*, std::vector<int64_t>>
      unspecified_dims;
  std::vector<HloSharding> saved_root_shardings;
  absl::flat_hash_map<int64_t, HloSharding> saved_parameter_shardings;
  absl::flat_hash_map<HloInstruction*, int64_t> instruction_to_shard_group_id;
  absl::flat_hash_map<int64_t, absl::flat_hash_set<HloInstruction*>>
      shard_group_id_to_shard_as_group;
  absl::flat_hash_map<int64_t, absl::flat_hash_set<HloInstruction*>>
      shard_group_id_to_shard_like_group;
  TF_ASSIGN_OR_RETURN(
      bool changed,
      ProcessShardingInstruction(
          module, execution_threads, !cse_prevention_only_, &unspecified_dims,
          allow_spmd_sharding_propagation_to_output_ ? &saved_root_shardings
                                                     : nullptr,
          allow_spmd_sharding_propagation_to_parameters_
              ? &saved_parameter_shardings
              : nullptr,
          &instruction_to_shard_group_id, &shard_group_id_to_shard_as_group,
          &shard_group_id_to_shard_like_group,
          &allow_spmd_sharding_propagation_to_parameters_vector_));
  any_changed |= changed;

  for (const auto& [shard_group_id, shard_as_group] :
       shard_group_id_to_shard_as_group) {
    VLOG(5) << "Shard-As group " << shard_group_id << " contains:";
    for (auto instruction : shard_as_group) {
      VLOG(5) << "  " << instruction->ToString();
    }
  }

  for (const auto& [shard_group_id, shard_like_group] :
       shard_group_id_to_shard_like_group) {
    VLOG(5) << "Shard-Like group " << shard_group_id << " contains:";
    for (auto instruction : shard_like_group) {
      VLOG(5) << "  " << instruction->ToString();
    }
  }

  // Check sizes of the given allow_spmd_sharding_propagation vectors
  if (allow_spmd_sharding_propagation_to_output_) {
    CHECK(!module->entry_computation()->root_instruction()->has_sharding() ||
          allow_spmd_sharding_propagation_to_output_vector_.size() == 1 ||
          module->entry_computation()
                  ->root_instruction()
                  ->sharding()
                  .tuple_elements()
                  .size() ==
              allow_spmd_sharding_propagation_to_output_vector_.size())
        << "allow-spmd-sharding-propagation-to-output-vector's size can be "
           "either 1 or the number of elements in the root tuple of entry "
           "computation.";
  }
  if (allow_spmd_sharding_propagation_to_parameters_) {
    auto is_same_sized_tuple = [](HloModule* module, int64_t size) {
      if (module->entry_computation()->num_parameters() != 1) {
        return false;
      }
      HloInstruction* param =
          module->entry_computation()->parameter_instruction(0);
      return param->shape().IsTuple() &&
             size == param->shape().tuple_shapes_size();
    };
    auto size = allow_spmd_sharding_propagation_to_parameters_vector_.size();
    CHECK(size == 1 || size == module->entry_computation()->num_parameters() ||
          is_same_sized_tuple(module, size))
        << "allow-spmd-sharding-propagation-to-parameters-vector's size can be "
           "either 1 or the number of parameters in the entry computation.";
  }
  // Association of partitionable embedded computations with their parent
  // instruction.
  ComputationMap computation_map;

  // Instructions that are related through a computation and need to share the
  // same sharding.
  auto get_related_instructions = [this](HloInstruction* inst) {
    if (inst->opcode() == HloOpcode::kWhile) {
      return std::vector<HloInstruction*>{
          inst, inst->while_body()->root_instruction(),
          inst->while_body()->parameter_instruction(0),
          inst->while_condition()->parameter_instruction(0)};
    } else if (inst->opcode() == HloOpcode::kConditional) {
      const auto& called_computations = inst->called_computations();
      std::vector<HloInstruction*> comps;
      comps.reserve(called_computations.size() + 1);
      comps.push_back(inst);
      for (HloComputation* c : called_computations) {
        comps.push_back(c->root_instruction());
      }
      return comps;
    } else if (inst->opcode() == HloOpcode::kCustomCall) {
      if (sharding_helper_ && sharding_helper_->IsCustomCallShardable(inst)) {
        return sharding_helper_->GetRelatedInstructions(inst);
      } else {
        return std::vector<HloInstruction*>{};
      }
    } else if (inst->opcode() == HloOpcode::kCall) {
      HloComputation* callee = inst->called_computations().front();
      return std::vector<HloInstruction*>{inst, callee->root_instruction()};
    } else {
      CHECK(false);
    }
  };

  // If instruction is a while, or the root or a parameter of a while body,
  // then propagate its sharding to the while instruction, to its body root,
  // and to its condition parameter.
  std::function<void(HloInstruction*, absl::flat_hash_set<HloInstruction*>*)>
      maybe_computation_propagation =
          [&](HloInstruction* instruction,
              absl::flat_hash_set<HloInstruction*>* changed) {
            auto propagate_to_instruction = [&](HloInstruction* search_inst) {
              auto related_instructions = get_related_instructions(search_inst);
              if (absl::c_count(related_instructions, instruction)) {
                for (HloInstruction* inst : related_instructions) {
                  if (!inst->has_sharding() ||
                      inst->sharding() != instruction->sharding()) {
                    VLOG(2) << "Add computation sharding: " << inst->name()
                            << " " << instruction->sharding().ToString();
                    inst->copy_sharding(instruction);
                    changed->insert(inst);
                    maybe_computation_propagation(inst, changed);
                  }
                }
              }
            };

            if (instruction->opcode() == HloOpcode::kConditional ||
                instruction->opcode() == HloOpcode::kWhile ||
                instruction->opcode() == HloOpcode::kCustomCall ||
                instruction->opcode() == HloOpcode::kCall) {
              propagate_to_instruction(instruction);
            }

            if (instruction->opcode() == HloOpcode::kParameter ||
                instruction->parent()->root_instruction() == instruction) {
              auto it = computation_map.find(instruction->parent());
              if (it != computation_map.end()) {
                propagate_to_instruction(it->second);
              }
            }
          };

  for (auto computation : module->computations(execution_threads)) {
    for (auto instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        TF_RETURN_IF_ERROR(
            CheckAndUpdateDeviceAssignmentsInWhileBody(instruction));
      }
    }
  }

  // Populate computation_map in order to associate while bodies to their
  // while instructions.
  for (auto computation : module->computations(execution_threads)) {
    for (auto instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kWhile ||
          instruction->opcode() == HloOpcode::kConditional ||
          instruction->opcode() == HloOpcode::kCall) {
        // Check if any of the related instructions has sharding, in which case
        // propagate it to the other instructions, so they all share the same
        // sharding, in case the user didn't shard all of them. We don't check
        // that user shardings are consistent, because such check is already
        // done by HLO verifier.
        const HloInstruction* sharded_inst = nullptr;
        auto related_instructions = get_related_instructions(instruction);
        for (auto inst : related_instructions) {
          if (inst->has_sharding()) {
            sharded_inst = inst;
            break;
          }
        }
        if (sharded_inst != nullptr) {
          // Set the same sharding to all the other related instructions.
          for (auto inst : related_instructions) {
            inst->copy_sharding(sharded_inst);
          }
        }
        if (instruction->opcode() == HloOpcode::kWhile) {
          computation_map[instruction->while_body()] = instruction;
        } else {
          for (HloComputation* c : instruction->called_computations()) {
            computation_map[c] = instruction;
          }
        }
      }
    }
  }
  // Collect all pre-sharded instructions as we aren't allowed to modify their
  // sharding.
  absl::flat_hash_set<const HloInstruction*> provided_shardings;
  for (const HloComputation* computation :
       module->computations(execution_threads)) {
    for (const HloInstruction* inst : computation->instructions()) {
      if (inst->has_sharding() &&
          inst != module->entry_computation()->root_instruction() &&
          inst->opcode() != HloOpcode::kParameter &&
          !inst->sharding().IsUnknown()) {
        provided_shardings.insert(inst);
      }
    }
  }

  if (!allow_spmd_sharding_propagation_to_output_ &&
      (!module->entry_computation()->root_instruction()->has_sharding() ||
       !module->entry_computation()
            ->root_instruction()
            ->sharding()
            .IsUnknown())) {
    // Consider the root instruction of the entry module as one with provided
    // sharding as its sharding have to match with the one expected by the host.
    provided_shardings.insert(module->entry_computation()->root_instruction());
  }

  if (!allow_spmd_sharding_propagation_to_parameters_) {
    for (auto param : module->entry_computation()->parameter_instructions()) {
      if (param->has_sharding() && !param->sharding().IsUnknown()) {
        provided_shardings.insert(param);
      }
    }
  }

  // Replace all unknown shardings with replicated sharding for propagation.
  for (HloComputation* computation : module->computations(execution_threads)) {
    auto instructions = computation->MakeInstructionPostOrder();
    for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
      HloInstruction* instruction = *it;
      if (instruction->has_sharding() && instruction->sharding().IsUnknown()) {
        instruction->set_sharding(
            HloSharding::Replicate(instruction->sharding().metadata()));
      }
    }
  }
  // Iterate to a fixpoint that is guaranteed to be reached because we only
  // strictly improve the sharding of the graph and it can't be improved
  // indefinitely.
  int64_t iterations = 0;

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  auto run_to_fix_point = [&](int64_t aggressiveness,
                              bool propagate_shard_group) {
    absl::flat_hash_set<const HloInstruction*>
        already_inferred_from_shard_group;
    absl::flat_hash_set<const HloInstruction*> already_inferred_from_operands;
    absl::flat_hash_set<const HloInstruction*> already_inferred_from_users;
    bool changed_last_iter = true;
    const bool may_merge_partial = is_spmd_ && aggressiveness > 0;
    while (changed_last_iter) {
      changed_last_iter = false;
      int64_t inferred_from_shard_group_counter = 0;
      int64_t inferred_from_operand_counter = 0;
      int64_t inferred_from_user_counter = 0;
      int64_t instruction_counter = 0;
      int64_t already_sharded_counter = 0;
      for (const HloComputation* computation :
           module->computations(execution_threads)) {
        VLOG(2) << "Consider computation: " << computation->name();
        std::vector<HloInstruction*> instructions =
            computation->MakeInstructionPostOrder();

        instruction_counter += instructions.size();
        already_sharded_counter += absl::c_count_if(
            instructions,
            [](const HloInstruction* inst) { return inst->has_sharding(); });
        auto clear_cache = [&](HloInstruction* hlo,
                               HloInstruction* hlo_for_users = nullptr) {
          for (auto operand : hlo->operands()) {
            already_inferred_from_users.erase(operand);
          }
          if (hlo_for_users == nullptr) {
            hlo_for_users = hlo;
          }
          for (auto user : hlo_for_users->users()) {
            already_inferred_from_operands.erase(user);
          }
          if (instruction_to_shard_group_id.contains(hlo)) {
            const int64_t shard_group_id =
                instruction_to_shard_group_id.at(hlo);
            const absl::flat_hash_set<HloInstruction*>& shard_group =
                shard_group_id_to_shard_as_group.contains(shard_group_id)
                    ? shard_group_id_to_shard_as_group.at(shard_group_id)
                    : shard_group_id_to_shard_like_group.at(shard_group_id);
            for (HloInstruction* member : shard_group) {
              if (member != hlo) {
                already_inferred_from_shard_group.erase(member);
              }
            }
          }
        };
        // 1. Iterate the shard groups to take shardings from instructions of
        // the same group.
        if (propagate_shard_group) {
          for (HloInstruction* instruction : instructions) {
            if (already_inferred_from_shard_group.contains(instruction)) {
              continue;
            }
            if (!instruction_to_shard_group_id.contains(instruction)) {
              continue;
            }
            const int64_t shard_group_id =
                instruction_to_shard_group_id.at(instruction);
            const absl::flat_hash_set<HloInstruction*>& shard_group =
                shard_group_id_to_shard_as_group.contains(shard_group_id)
                    ? shard_group_id_to_shard_as_group.at(shard_group_id)
                    : shard_group_id_to_shard_like_group.at(shard_group_id);
            if (provided_shardings.contains(instruction)) {
              if (!may_merge_partial) {
                continue;
              }
              auto it = unspecified_dims.find(instruction);
              if (it != unspecified_dims.end() &&
                  InferUnspecifiedDimsFromShardGroup(instruction, it->second,
                                                     shard_group)) {
                ++inferred_from_shard_group_counter;
                VLOG(2) << "Refined partial sharding (shard group): "
                        << instruction->ToString();
                clear_cache(instruction);
                already_inferred_from_shard_group.insert(instruction);
                changed_last_iter = true;
              }
              continue;
            }
            already_inferred_from_shard_group.insert(instruction);
            if (InferShardingFromShardGroup(instruction, computation_map,
                                            aggressiveness, shard_group)) {
              ++inferred_from_shard_group_counter;
              any_changed = true;
              VLOG(2) << "Add sharding (shard group): "
                      << instruction->ToString();
              absl::flat_hash_set<HloInstruction*> changed_in_comp_prop;
              maybe_computation_propagation(instruction, &changed_in_comp_prop);
              clear_cache(instruction);
              for (auto hlo : changed_in_comp_prop) {
                clear_cache(hlo);
              }
              changed_last_iter = true;
            }
          }
        }
        // 2. Iterate the HLO graph in post order taking shardings from
        // operands.
        for (HloInstruction* instruction : instructions) {
          if (already_inferred_from_operands.contains(instruction)) {
            continue;
          }
          if (provided_shardings.contains(instruction)) {
            if (!may_merge_partial) {
              continue;
            }
            auto it = unspecified_dims.find(instruction);
            HloInstruction* man_conversion_op_after;
            if (it != unspecified_dims.end() &&
                InferUnspecifiedDimsFromOperand(instruction, it->second,
                                                &man_conversion_op_after)) {
              ++inferred_from_operand_counter;
              VLOG(2) << "Refined partial sharding (forward-pass): "
                      << instruction->ToString();
              clear_cache(instruction, man_conversion_op_after);
              already_inferred_from_operands.insert(instruction);
              changed_last_iter = true;
            }
            continue;
          }
          already_inferred_from_operands.insert(instruction);
          if (InferShardingFromOperands(instruction, computation_map,
                                        aggressiveness, *call_graph,
                                        execution_threads)) {
            ++inferred_from_operand_counter;
            any_changed = true;
            VLOG(2) << "Add sharding (forward-pass): "
                    << instruction->ToString();
            absl::flat_hash_set<HloInstruction*> changed_in_comp_prop;
            maybe_computation_propagation(instruction, &changed_in_comp_prop);
            clear_cache(instruction);
            for (auto hlo : changed_in_comp_prop) {
              clear_cache(hlo);
            }
            changed_last_iter = true;
          }
        }
        // 3. Iterate the HLO graph in reverse post order taking shardings from
        // users.
        for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
          if ((*it)->IsCustomCall("SPMDFullToShardShape") ||
              (*it)->IsCustomCall("SPMDShardToFullShape")) {
            // The manual conversion op is processed together with the sharding
            // op before it. If the conversion op is removed from cache, the
            // sharding op should also be removed.
            if (!already_inferred_from_users.contains(*it)) {
              already_inferred_from_users.erase((*it)->operand(0));
            }
          }
          if (already_inferred_from_users.contains(*it)) {
            continue;
          }
          if (provided_shardings.contains(*it)) {
            if (!may_merge_partial) {
              continue;
            }
            auto uit = unspecified_dims.find(*it);
            HloInstruction* man_conversion_op_after;
            if (uit != unspecified_dims.end() &&
                InferUnspecifiedDimsFromUsers(
                    *it, uit->second, aggressiveness, is_spmd_,
                    &man_conversion_op_after, *call_graph)) {
              ++inferred_from_user_counter;
              VLOG(2) << "Refined partial sharding (backward-pass): "
                      << (*it)->ToString();
              clear_cache(*it, man_conversion_op_after);
              already_inferred_from_users.insert(*it);
              if (man_conversion_op_after != nullptr) {
                already_inferred_from_users.insert(man_conversion_op_after);
              }
              changed_last_iter = true;
            }
            continue;
          }
          already_inferred_from_users.insert(*it);
          if (InferShardingFromUsers(*it, computation_map, aggressiveness,
                                     is_spmd_, sharding_helper_.get(),
                                     *call_graph)) {
            ++inferred_from_user_counter;
            any_changed = true;
            VLOG(2) << "Add sharding (backward-pass): " << (*it)->ToString();
            absl::flat_hash_set<HloInstruction*> changed_in_comp_prop;
            maybe_computation_propagation(*it, &changed_in_comp_prop);
            clear_cache(*it);
            for (auto hlo : changed_in_comp_prop) {
              clear_cache(hlo);
            }
            changed_last_iter = true;
          }
        }
      }
      VLOG(1) << "Sharding propagation iteration " << iterations << ";"
              << "\n  total instructions: " << instruction_counter
              << "\n  instructions already sharded: " << already_sharded_counter
              << "\n  shardings inferred from shard group: "
              << inferred_from_shard_group_counter
              << "\n  shardings inferred from operands: "
              << inferred_from_operand_counter
              << "\n  shardings inferred from users: "
              << inferred_from_user_counter
              << "\n  aggressiveness: " << aggressiveness;
      ++iterations;
    }
    return OkStatus();
  };
  for (int64_t aggressiveness = 0; aggressiveness < 4; ++aggressiveness) {
    TF_RETURN_IF_ERROR(
        run_to_fix_point(aggressiveness, /*propagate_shard_group=*/true));
  }

  // Align the shardings from the same shard_as group so that they will adopt
  // the same sharding.
  for (const auto& [shard_as_group_id, shard_as_group] :
       shard_group_id_to_shard_as_group) {
    // If all the inferred shardings of the instructions from the same shard
    // group are compatible with each other, then we will merge all of them to
    // get the most specific sharding. If some of them are not compatible, then
    // it will just choose the a random sharding among them(say the first one),
    // with the guarantee that the defaultly chosen sharding will not be from a
    // ShardBarrierFrom op if there is one within the ShardAs group.
    HloSharding default_sharding = HloSharding::Replicate();
    std::vector<HloSharding> shardings;
    for (HloInstruction* instruction : shard_as_group) {
      if (instruction->has_sharding()) {
        shardings.push_back(instruction->sharding());
        if (!instruction->IsCustomCall(spmd::kShardBarrierFrom) &&
            default_sharding.IsReplicated()) {
          default_sharding = instruction->sharding();
        }
      }
    }

    HloSharding common_sharding =
        hlo_sharding_util::FindCommonSharding(shardings, default_sharding);
    VLOG(2) << "Aligning shard group: " << shard_as_group_id
            << " to sharding:" << common_sharding.ToString();
    for (HloInstruction* member : shard_as_group) {
      if (member->IsCustomCall(spmd::kShardBarrierTo)) {
        continue;
      }
      if (provided_shardings.contains(member)) {
        auto it = unspecified_dims.find(member);
        if (it != unspecified_dims.end()) {
          HloSharding partial_replicated =
              hlo_sharding_util::PartiallyReplicateTiledShardingOnAllDimsExcept(
                  common_sharding, it->second);
          HloSharding sharding = member->sharding();
          if (hlo_sharding_util::MergeShardingIfCompatible(
                  partial_replicated, sharding.NumTiles() + 1, &sharding)) {
            member->set_sharding(sharding);
          }
        }
      }
      member->set_sharding(common_sharding);
    }
  }

  //  If a ShardBarrierFrom custom-call op is in a shard as group, and relay
  // the shard as sharding to its original op, do not relay shardings for
  // ShardbarrierTo op. Then run sharding propagation once more at highest
  // aggressiveness without propagating shard group.
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->IsCustomCall(spmd::kShardBarrierFrom) &&
          instruction_to_shard_group_id.contains(instruction) &&
          shard_group_id_to_shard_as_group.contains(
              instruction_to_shard_group_id.at(instruction))) {
        HloSharding sharding = instruction->sharding();
        hlo_sharding_util::MergeShardingIfCompatible(
            instruction->mutable_operand(0)->sharding(), sharding.NumTiles(),
            &sharding);
        instruction->mutable_operand(0)->set_sharding(std::move(sharding));
      }
    }
  }
  TF_RETURN_IF_ERROR(
      run_to_fix_point(/*aggressiveness=*/3, /*propagate_shard_group=*/false));

  // Post-process to remove all "shard-barrier-from" and "shard-barrier-to"
  // custom-calls.
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      // If a ShardBarrierFrom custom-call op is in a shard as group, and relay
      // the shard as sharding to its original op, do not relay shardings for
      // ShardbarrierTo op.
      if (instruction->IsCustomCall(spmd::kShardBarrierFrom) &&
          instruction_to_shard_group_id.contains(instruction) &&
          shard_group_id_to_shard_as_group.contains(
              instruction_to_shard_group_id.at(instruction))) {
        HloSharding sharding = instruction->sharding();
        hlo_sharding_util::MergeShardingIfCompatible(
            instruction->mutable_operand(0)->sharding(), sharding.NumTiles(),
            &sharding);
        instruction->mutable_operand(0)->set_sharding(std::move(sharding));
      }
      if (instruction->IsCustomCall(spmd::kShardBarrierFrom) ||
          instruction->IsCustomCall(spmd::kShardBarrierTo)) {
        TF_ASSIGN_OR_RETURN(std::ignore,
                            computation->ReplaceInstruction(
                                instruction, instruction->mutable_operand(0),
                                /*preserve_sharding=*/false,
                                /*relay_control_dependency=*/false,
                                /*remove_unused_operands=*/false));
      }
    }
  }

  // Post-process for CSE prevention.
  if (cse_prevention_only_) {
    for (auto computation : module->computations(execution_threads)) {
      for (auto instruction : computation->instructions()) {
        if (!instruction->has_sharding()) {
          continue;
        }
        if (IsCSEPreventionTarget(instruction) && instruction->has_sharding()) {
          if (!(*original_sharding).contains(instruction)) {
            // Mark the propagated sharding as for CSE prevention.
            instruction->set_sharding(
                SetCSEPreventionSharding(instruction->sharding()));
          }
          continue;
        }
        auto it = (*original_sharding).find(instruction);
        if (it != (*original_sharding).end()) {
          // Revert sharding.
          instruction->set_sharding(it->second);
        } else {
          // Clear sharding.
          instruction->clear_sharding();
        }
      }
    }
  }
  HloInstruction* root_instruction =
      module->entry_computation()->root_instruction();
  if (saved_root_shardings.size() ==
          allow_spmd_sharding_propagation_to_output_vector_.size() &&
      root_instruction->has_sharding()) {
    HloSharding root_sharding = root_instruction->sharding();
    for (int i = 0; i < saved_root_shardings.size(); ++i) {
      if (!allow_spmd_sharding_propagation_to_output_vector_[i] &&
          !saved_root_shardings[i].IsUnknown()) {
        root_sharding.tuple_elements()[i] = saved_root_shardings[i];
      }
    }
    root_instruction->set_sharding(std::move(root_sharding));
  }
  auto params = module->entry_computation()->parameter_instructions();
  if (allow_spmd_sharding_propagation_to_parameters_) {
    if (allow_spmd_sharding_propagation_to_parameters_vector_.size() ==
        params.size()) {
      for (int64_t i = 0; i < params.size(); ++i) {
        if (!allow_spmd_sharding_propagation_to_parameters_vector_[i]) {
          if (saved_parameter_shardings.contains(i) &&
              !saved_parameter_shardings.at(i).IsUnknown()) {
            params[i]->set_sharding(saved_parameter_shardings.at(i));
          } else {
            params[i]->clear_sharding();
          }
        }
      }
    } else if (params.size() == 1 && saved_parameter_shardings.size() == 1 &&
               params[0]->shape().IsTuple() &&
               params[0]->shape().tuple_shapes_size() ==
                   allow_spmd_sharding_propagation_to_parameters_vector_
                       .size()) {
      // There is a single parameter which is a tuple with many elements.
      HloSharding param_sharding = params[0]->sharding();
      for (int64_t i = 0; i < params[0]->shape().tuple_shapes_size(); ++i) {
        HloSharding saved_subsharding =
            saved_parameter_shardings.at(0).GetSubSharding(params[0]->shape(),
                                                           {i});
        if (!allow_spmd_sharding_propagation_to_parameters_vector_[i] &&
            !saved_subsharding.IsUnknown()) {
          param_sharding.tuple_elements()[i] = saved_subsharding;
        }
      }
      params[0]->set_sharding(std::move(param_sharding));
    }
  }
  // Replicate the parameter/output sharding if the propagated sharding does not
  // evenly partition the parameter/output.
  std::function<bool(const Shape&, const HloSharding&)> evenly_partitions =
      [&evenly_partitions](const Shape& shape,
                           const HloSharding& sharding) -> bool {
    if (!sharding.IsTiled()) {
      return true;
    }
    if (sharding.IsTileMaximal()) {
      return sharding.IsReplicated();
    }
    if (sharding.IsTuple()) {
      for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
        if (!evenly_partitions(ShapeUtil::GetTupleElementShape(shape, i),
                               sharding.GetSubSharding(shape, {i}))) {
          return false;
        }
      }
    }
    for (int64_t i = 0; i < shape.dimensions_size(); ++i) {
      if (shape.dimensions(i) % sharding.tile_assignment().dim(i) != 0) {
        return false;
      }
    }
    return true;
  };
  if (allow_spmd_sharding_propagation_to_output_ &&
      root_instruction->has_sharding()) {
    if (root_instruction->shape().IsTuple() &&
        allow_spmd_sharding_propagation_to_output_vector_.size() ==
            root_instruction->shape().tuple_shapes_size()) {
      // The output shape is a tuple and sharding propagation is allowed for at
      // least one of its elements.
      HloSharding root_sharding = root_instruction->sharding();
      for (int64_t i = 0; i < root_instruction->shape().tuple_shapes_size();
           ++i) {
        if (allow_spmd_sharding_propagation_to_output_vector_[i] &&
            !evenly_partitions(root_instruction->shape().tuple_shapes(i),
                               root_sharding.tuple_elements()[i])) {
          root_sharding.tuple_elements()[i] = HloSharding::Replicate();
        }
      }
      root_instruction->set_sharding(std::move(root_sharding));
    } else if (!root_instruction->shape().IsTuple()) {
      // The output shape is not tuple and sharding propagation is allowed.
      if (!evenly_partitions(root_instruction->shape(),
                             root_instruction->sharding())) {
        root_instruction->set_sharding(HloSharding::Replicate());
      }
    }
  }
  if (allow_spmd_sharding_propagation_to_parameters_) {
    // Sharding propagation is allowed for at least one parameter.
    if (allow_spmd_sharding_propagation_to_parameters_vector_.size() ==
        params.size()) {
      for (int64_t i = 0; i < params.size(); ++i) {
        if (params[i]->has_sharding() &&
            allow_spmd_sharding_propagation_to_parameters_vector_[i] &&
            !evenly_partitions(params[i]->shape(), params[i]->sharding())) {
          params[i]->set_sharding(HloSharding::Replicate());
        }
      }
    } else if (params.size() == 1 && params[0]->shape().IsTuple() &&
               params[0]->has_sharding() &&
               params[0]->shape().tuple_shapes_size() ==
                   allow_spmd_sharding_propagation_to_parameters_vector_
                       .size()) {
      HloSharding param_sharding = params[0]->sharding();
      for (int64_t i = 0; i < params[0]->shape().tuple_shapes_size(); ++i) {
        if (allow_spmd_sharding_propagation_to_parameters_vector_[i] &&
            !evenly_partitions(
                ShapeUtil::GetSubshapeOneIndex(params[0]->shape(), i),
                params[0]->sharding().GetSubSharding(params[0]->shape(),
                                                     {i}))) {
          param_sharding.tuple_elements()[i] = HloSharding::Replicate();
        }
      }
      params[0]->set_sharding(std::move(param_sharding));
    }
  }
  TF_RETURN_IF_ERROR(CanonicalizeLayouts(module));

  VLOG(1) << "Sharding propagation completed after " << iterations
          << " iterations";
  return any_changed;
}

}  // namespace xla
