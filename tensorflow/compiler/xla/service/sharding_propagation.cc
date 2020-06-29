/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/sharding_propagation.h"

#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/dot_as_convolution_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using ComputationMap =
    absl::flat_hash_map<const HloComputation*, HloInstruction*>;

// Returns true iff the specified hlo or sharding has a spatially partitioned
// sharding (tiled or replicated) what can be propagated by sharding
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

// Returns true if the lhs sharding is preferable over the rhs sharding.
// The most specific sharding is tile maximal followed by single device tile
// maximal and finally replicated. This order aims to primarily reduce memory
// usage and secondly reduce total compute.
// Note: This does NOT provide a total ordering as we can have 2 different
// sharding with same preference level.
bool IsShardingMoreSpecific(const HloSharding& lhs, const HloSharding& rhs) {
  CHECK_EQ(lhs.IsTuple(), rhs.IsTuple());
  if (lhs.IsTuple()) {
    // For tuples we consider lhs to have a better sharding if none of the
    // elements are worse and at least one element is better then in rhs
    // sharding.
    const auto& lhs_shardings = lhs.tuple_elements();
    const auto& rhs_shardings = rhs.tuple_elements();
    CHECK_EQ(lhs_shardings.size(), rhs_shardings.size());
    bool is_better = false;
    for (int64 i = 0; i < lhs_shardings.size(); ++i) {
      if (IsShardingMoreSpecific(rhs_shardings[i], lhs_shardings[i])) {
        return false;
      }
      if (IsShardingMoreSpecific(lhs_shardings[i], rhs_shardings[i])) {
        is_better = true;
      }
    }
    return is_better;
  }
  if (!rhs.IsTileMaximal()) {
    // If we already have a non-tile-maximal sharding then we can't improve
    // that.
    return false;
  } else if (!rhs.IsReplicated()) {
    // If we are not replicated then only tiled (not tile maximal) shardings
    // can improve us.
    return !lhs.IsTileMaximal();
  } else {
    // If we are replicated then any non-replicated sharding can improve us.
    return !lhs.IsReplicated();
  }
}

// Returns a sharding where each tuple element is chosen as the more specific
// one of the corresponding elements in a and b. Requires a an b to have the
// same tuple nesting.
HloSharding MergeForMoreSpecificSharding(const HloSharding& a,
                                         const HloSharding& b) {
  if (a.IsTuple()) {
    HloSharding result = a;
    CHECK(b.IsTuple());
    CHECK_EQ(a.tuple_elements().size(), b.tuple_elements().size());
    for (int64 i = 0; i < result.tuple_elements().size(); ++i) {
      result.tuple_elements()[i] = MergeForMoreSpecificSharding(
          a.tuple_elements()[i], b.tuple_elements()[i]);
    }
    return result;
  }
  return IsShardingMoreSpecific(a, b) ? a : b;
}

// Updates the sharding of the specified instruction with the specified sharding
// if it is better than the current one and returns true if a new sharding have
// been applied.
bool MaybeImproveInstructionSharding(const HloSharding& sharding,
                                     HloInstruction* instruction) {
  // We don't want to propagate tile maximal shardings.
  if (!IsSpatiallyPartitioned(sharding)) {
    return false;
  }
  // Any sharding is better then no sharding.
  if (!instruction->has_sharding()) {
    instruction->set_sharding(sharding);
    return true;
  }
  if (IsShardingMoreSpecific(sharding, instruction->sharding())) {
    instruction->set_sharding(sharding);
    return true;
  }
  return false;
}

// Sets the sharding for every element within a tuple to replicated (default
// sharding). This is necessary because there is no way to represent a tuple
// sharding when only some of the elements are sharded.
void SetDefaultTupleSharding(HloInstruction* instruction) {
  instruction->set_sharding(
      HloSharding::SingleTuple(instruction->shape(), HloSharding::Replicate()));
}

// We consider a convolution kernel to be small iff it is smaller along all
// spatial dimensions then the output of the convolution. The rational is that
// we can either shard the kernel or the output and we want to shard the larger
// one for better efficiency.
bool IsConvolutionKernelSmall(const HloInstruction* instruction) {
  CHECK_EQ(instruction->opcode(), HloOpcode::kConvolution);
  const HloInstruction* rhs = instruction->operand(1);
  const auto& dnums = instruction->convolution_dimension_numbers();
  for (int64 i = 0; i < dnums.input_spatial_dimensions().size(); ++i) {
    int64 kernel_dim =
        rhs->shape().dimensions(dnums.kernel_spatial_dimensions(i));
    int64 output_dim =
        instruction->shape().dimensions(dnums.output_spatial_dimensions(i));
    if (kernel_dim >= output_dim) {
      return false;
    }
  }
  return true;
}

// Return the operand which is the most suitable for determining the sharding
// for the specified instruction or nullptr if there isn't any suitable operand.
const HloInstruction* PickRepresentativeOperand(
    const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kMap:
    case HloOpcode::kPad:
    case HloOpcode::kPower:
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
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kDivide:
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
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSort:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTanh:
    case HloOpcode::kTupleSelect:
    case HloOpcode::kWhile:
    case HloOpcode::kXor: {
      // For these opcodes the output sharding can be determined by any operand
      // so we find the operand with the most specific sharding.
      const HloInstruction* best_operand = nullptr;
      for (const HloInstruction* operand : instruction->operands()) {
        if (operand->has_sharding() &&
            (best_operand == nullptr ||
             IsShardingMoreSpecific(operand->sharding(),
                                    best_operand->sharding()))) {
          best_operand = operand;
        }
      }
      return best_operand;
    }

    // There is no suitable operand for the rest of the opcodes.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
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
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kDot:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
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
    case HloOpcode::kTrace:
    case HloOpcode::kTranspose:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kTuple:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
      return nullptr;
  }
}

bool SupportSpatialPartitioning(const HloInstruction* instruction,
                                const ComputationMap& computation_map,
                                bool is_spmd) {
  if (instruction->parent()->root_instruction() == instruction &&
      computation_map.find(instruction->parent()) == computation_map.end()) {
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
      return true;
    case HloOpcode::kAllReduce:
      // Only if channel_id is not specified.
      return instruction->channel_id() == absl::nullopt;
    case HloOpcode::kParameter:
      return computation_map.find(instruction->parent()) !=
             computation_map.end();
    case HloOpcode::kReverse:
      return is_spmd;
    default:
      return false;
  }
}

// Convolution handling for InferShardingFromOperands().
bool InferConvolutionShardingFromOperands(HloInstruction* instruction,
                                          bool aggressive_prop) {
  const auto& dnums = instruction->convolution_dimension_numbers();
  const HloInstruction* lhs = instruction->operand(0);
  const HloInstruction* rhs = instruction->operand(1);
  auto get_tiled_sharding_based_on_lhs = [&] {
    CHECK(!lhs->sharding().IsTileMaximal());
    std::vector<int64> output_to_lhs_indices(instruction->shape().rank());
    output_to_lhs_indices[dnums.output_batch_dimension()] =
        dnums.input_batch_dimension();
    output_to_lhs_indices[dnums.output_feature_dimension()] =
        dnums.input_feature_dimension();
    for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
      output_to_lhs_indices[dnums.output_spatial_dimensions(i)] =
          dnums.input_spatial_dimensions(i);
    }
    return hlo_sharding_util::TransposeSharding(lhs->sharding(),
                                                output_to_lhs_indices);
  };
  auto get_tiled_sharding_based_on_rhs = [&] {
    CHECK(!rhs->sharding().IsTileMaximal());
    std::vector<int64> output_to_rhs_indices(instruction->shape().rank());
    output_to_rhs_indices[dnums.output_batch_dimension()] =
        dnums.kernel_input_feature_dimension();
    output_to_rhs_indices[dnums.output_feature_dimension()] =
        dnums.kernel_output_feature_dimension();
    for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
      output_to_rhs_indices[dnums.output_spatial_dimensions(i)] =
          dnums.kernel_spatial_dimensions(i);
    }
    return hlo_sharding_util::TransposeSharding(rhs->sharding(),
                                                output_to_rhs_indices);
  };
  if (auto dot_dims = dot_as_convolution_util::ParseDotGeneralFromConvolution(
          instruction)) {
    // lhs_or_rhs: lhs is 0 and rhs is 1. Skips dimensions with size 1.
    auto partitioned_only_along_non_trivial_dims =
        [&](const HloSharding& sharding,
            std::vector<dot_as_convolution_util::
                            DotGeneralAsConvolutionDimsInfo::DimNums>& dims,
            int64 lhs_or_rhs) {
          if (sharding.IsTileMaximal()) {
            return false;
          }
          int64 partition_count = 1;
          for (const auto& dim : dims) {
            if (lhs_or_rhs == 0) {
              if (lhs->shape().dimensions(dim.lhs) == 1) {
                continue;
              }
              partition_count *= sharding.tile_assignment().dim(dim.lhs);
            } else {
              if (rhs->shape().dimensions(dim.rhs) == 1) {
                continue;
              }
              CHECK_EQ(lhs_or_rhs, 1);
              partition_count *= sharding.tile_assignment().dim(dim.rhs);
            }
          }
          return partition_count == sharding.tile_assignment().num_elements();
        };
    // If LHS/RHS is partitioned only along the batch dimensions, propagate
    // the sharding to the output, since batch dimensions are the easiest to
    // partition.
    if (IsSpatiallyPartitioned(lhs) &&
        partitioned_only_along_non_trivial_dims(lhs->sharding(),
                                                dot_dims->batch_dims, 0)) {
      return MaybeImproveInstructionSharding(get_tiled_sharding_based_on_lhs(),
                                             instruction);
    }
    if (IsSpatiallyPartitioned(rhs) &&
        partitioned_only_along_non_trivial_dims(rhs->sharding(),
                                                dot_dims->batch_dims, 1)) {
      return MaybeImproveInstructionSharding(get_tiled_sharding_based_on_rhs(),
                                             instruction);
    }
    if (aggressive_prop) {
      // If LHS/RHS is partitioned only along the non-contracting
      // dimensions, propagate the sharding to the output.
      const bool can_propagate_from_lhs =
          IsSpatiallyPartitioned(lhs) &&
          partitioned_only_along_non_trivial_dims(
              lhs->sharding(), dot_dims->lhs_non_contracting_dims, 0);
      const bool can_propagate_from_rhs =
          IsSpatiallyPartitioned(rhs) &&
          partitioned_only_along_non_trivial_dims(
              rhs->sharding(), dot_dims->rhs_non_contracting_dims, 1);
      // If we can propagate from both operands, choose the larger one which
      // should help us reduce communications.
      if (can_propagate_from_lhs && can_propagate_from_rhs) {
        if (Product(lhs->shape().dimensions()) >=
            Product(rhs->shape().dimensions())) {
          return MaybeImproveInstructionSharding(
              get_tiled_sharding_based_on_lhs(), instruction);
        } else {
          return MaybeImproveInstructionSharding(
              get_tiled_sharding_based_on_rhs(), instruction);
        }
      }
      if (can_propagate_from_lhs) {
        return MaybeImproveInstructionSharding(
            get_tiled_sharding_based_on_lhs(), instruction);
      }
      if (can_propagate_from_rhs) {
        return MaybeImproveInstructionSharding(
            get_tiled_sharding_based_on_rhs(), instruction);
      }
    }
  }

  if (!IsSpatiallyPartitioned(lhs)) {
    return false;
  }
  if (lhs->sharding().IsReplicated()) {
    return MaybeImproveInstructionSharding(HloSharding::Replicate(),
                                           instruction);
  }

  if (IsConvolutionKernelSmall(instruction)) {
    // If the kernel is small compared to the input then we can generate an
    // output what is sharded the same way as the input.
    const auto& tile_assignment = lhs->sharding().tile_assignment();
    if (tile_assignment.dim(dnums.input_feature_dimension()) > 1) {
      return false;
    }
    return MaybeImproveInstructionSharding(get_tiled_sharding_based_on_lhs(),
                                           instruction);
  }
  // If the kernel is large (e.g backward convolution) then we only support
  // replicated output.
  return MaybeImproveInstructionSharding(HloSharding::Replicate(), instruction);
}

// Tries to update the sharding of the specified instruction based on its
// operands and returns true if the sharding of the instruction have been
// changed and false otherwise.
bool InferShardingFromOperands(HloInstruction* instruction,
                               const ComputationMap& computation_map,
                               bool is_spmd, bool aggressive_prop) {
  if (!SupportSpatialPartitioning(instruction, computation_map, is_spmd)) {
    // If an array shaped HLO doesn't support spatial partitioning but at least
    // one of its operand is replicated then we make the HLO replicated as well.
    if (instruction->shape().IsTuple() || instruction->operand_count() == 0 ||
        instruction == instruction->parent()->root_instruction() ||
        instruction->HasSideEffect()) {
      return false;
    }
    if (absl::c_any_of(instruction->operands(), [](const HloInstruction* op) {
          return op->has_sharding() && op->sharding().IsReplicated();
        })) {
      return MaybeImproveInstructionSharding(HloSharding::Replicate(),
                                             instruction);
    }
    return false;
  }

  switch (instruction->opcode()) {
    case HloOpcode::kGetTupleElement: {
      const HloInstruction* operand = instruction->operand(0);
      if (!IsSpatiallyPartitioned(operand)) {
        return false;
      }
      HloSharding new_sharding = operand->sharding().GetSubSharding(
          operand->shape(), {instruction->tuple_index()});
      return MaybeImproveInstructionSharding(new_sharding, instruction);
    }
    case HloOpcode::kTuple: {
      if (absl::c_none_of(instruction->operands(),
                          [](const HloInstruction* hlo) {
                            return IsSpatiallyPartitioned(hlo);
                          })) {
        // None of the operands have a spatially partitioned sharding.
        return false;
      }
      bool changed = false;
      if (!instruction->has_sharding()) {
        // Set the sharding for all elements in the tuple because it isn't
        // possible to set a partial sharding.
        SetDefaultTupleSharding(instruction);
        changed = true;
      }
      // Go through each operand and if the operand has a sharding that is
      // better than the current sharding for that tuple element then update
      // it.
      const Shape& shape = instruction->shape();
      std::vector<HloSharding> sub_shardings =
          instruction->sharding().tuple_elements();
      int64 sub_sharding_index = 0;
      for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
        const HloInstruction* operand = instruction->operand(i);
        if (operand->has_sharding()) {
          if (operand->shape().IsTuple()) {
            for (int64 i = 0, e = ShapeUtil::GetLeafCount(operand->shape());
                 i < e; ++i) {
              if (IsShardingMoreSpecific(
                      operand->sharding().tuple_elements()[i],
                      sub_shardings[sub_sharding_index + i])) {
                sub_shardings[sub_sharding_index + i] =
                    operand->sharding().tuple_elements()[i];
              }
            }
          } else {
            if (IsShardingMoreSpecific(operand->sharding(),
                                       sub_shardings[sub_sharding_index])) {
              sub_shardings[sub_sharding_index] = operand->sharding();
            }
          }
        }
        sub_sharding_index += ShapeUtil::GetLeafCount(operand->shape());
      }

      HloSharding new_sharding = HloSharding::Tuple(shape, sub_shardings);
      if (new_sharding != instruction->sharding()) {
        instruction->set_sharding(new_sharding);
        return true;
      }
      return changed;
    }
    case HloOpcode::kReduce: {
      // Reduce could have a tuple shape, where the first half of operands are
      // the arrays to reduce, and the second half of operands are the init
      // values.
      bool changed = false;
      for (int64 operand_id = 0; operand_id < instruction->operand_count() / 2;
           ++operand_id) {
        const HloInstruction* operand = instruction->operand(operand_id);
        if (!IsSpatiallyPartitioned(operand)) {
          continue;
        }
        auto get_maybe_tuple_sharding = [&](const HloSharding& sharding) {
          if (instruction->operand_count() == 2) {
            return sharding;
          }
          std::vector<HloSharding> tuple(instruction->operand_count() / 2,
                                         sharding);
          return HloSharding::Tuple(instruction->shape(), tuple);
        };
        if (operand->sharding().IsReplicated()) {
          changed |= MaybeImproveInstructionSharding(
              get_maybe_tuple_sharding(HloSharding::Replicate()), instruction);
          continue;
        }
        if (absl::c_any_of(instruction->dimensions(), [operand](int64 dim) {
              return operand->sharding().tile_assignment().dim(dim) > 1;
            })) {
          // We are reducing along one of the sharded dimensions. We don't
          // support tiled sharding in this case.
          changed |= MaybeImproveInstructionSharding(
              get_maybe_tuple_sharding(HloSharding::Replicate()), instruction);
        } else {
          // We are reducing along some of the non-sharded dimensions. The
          // result sharding should be the same as the operand sharding with the
          // reduction dimensions removed as they are removed from the result
          // shape.
          std::vector<int64> target_tile_assignment_dimensions;
          const auto& dimensions = instruction->dimensions();
          for (int64 i = 0; i < operand->shape().rank(); ++i) {
            if (absl::c_find(dimensions, i) == dimensions.end()) {
              target_tile_assignment_dimensions.push_back(
                  operand->sharding().tile_assignment().dim(i));
            }
          }
          Array<int64> new_tile_assignment =
              operand->sharding().tile_assignment();
          new_tile_assignment.Reshape(target_tile_assignment_dimensions);
          // Use the same sharding for all tuple elements, because they are part
          // of the same reduce instruction.
          HloSharding new_sharding =
              get_maybe_tuple_sharding(HloSharding::Tile(new_tile_assignment));
          changed |= MaybeImproveInstructionSharding(new_sharding, instruction);
        }
      }
      return changed;
    }
    case HloOpcode::kBroadcast: {
      const HloInstruction* op = instruction->operand(0);
      if (!IsSpatiallyPartitioned(op) || op->sharding().IsReplicated()) {
        return false;
      }
      // Heuristic: If an operand is more than 8 times fewer elements than its
      // output, do not propagate sharding.
      if (ShapeUtil::ElementsIn(instruction->shape()) >
          8 * ShapeUtil::ElementsIn(op->shape())) {
        return false;
      }
      // The output will be tiled along the broadcasted dimension the same way
      // as the input for the broadcast while the other dimensions are kept
      // non-tiled.
      std::vector<int64> target_tile_assignment_dimensions;
      const auto& dimensions = instruction->dimensions();
      for (int64 i = 0; i < instruction->shape().rank(); ++i) {
        auto it = absl::c_find(dimensions, i);
        if (it == dimensions.end()) {
          target_tile_assignment_dimensions.push_back(1);
        } else {
          const int64 source_dim = std::distance(dimensions.begin(), it);
          target_tile_assignment_dimensions.push_back(
              op->sharding().tile_assignment().dim(source_dim));
        }
      }
      Array<int64> new_tile_assignment = op->sharding().tile_assignment();
      new_tile_assignment.Reshape(target_tile_assignment_dimensions);
      HloSharding new_sharding = HloSharding::Tile(new_tile_assignment);
      return MaybeImproveInstructionSharding(new_sharding, instruction);
    }
    case HloOpcode::kConvolution:
      return InferConvolutionShardingFromOperands(instruction, aggressive_prop);
    case HloOpcode::kTranspose: {
      const HloInstruction* input = instruction->operand(0);
      if (!IsSpatiallyPartitioned(input)) {
        return false;
      }
      HloSharding sharding = hlo_sharding_util::TransposeSharding(
          input->sharding(), instruction->dimensions());
      return MaybeImproveInstructionSharding(sharding, instruction);
    }
    case HloOpcode::kReduceWindow: {
      const HloInstruction* lhs = instruction->operand(0);
      if (!IsSpatiallyPartitioned(lhs)) {
        return false;
      }

      auto has_dilation = [](const WindowDimension& dimensions) {
        return dimensions.base_dilation() > 1 ||
               dimensions.window_dilation() > 1;
      };
      if (absl::c_any_of(instruction->window().dimensions(), has_dilation)) {
        VLOG(2) << "Not applying sharding to reduce window because dilatation "
                   "isn't supported yet: "
                << instruction->ToString();
        return false;
      }
      return MaybeImproveInstructionSharding(lhs->sharding(), instruction);
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
      return MaybeImproveInstructionSharding(lhs->sharding(), instruction);
    }
    case HloOpcode::kReshape: {
      if (!IsSpatiallyPartitioned(instruction->operand(0))) {
        return false;
      }
      absl::optional<HloSharding> new_sharding =
          hlo_sharding_util::ReshapeSharding(
              instruction->operand(0)->shape(), instruction->shape(),
              instruction->operand(0)->sharding());
      if (new_sharding.has_value()) {
        return MaybeImproveInstructionSharding(new_sharding.value(),
                                               instruction);
      }
      return false;
    }
    case HloOpcode::kReverse: {
      if (!IsSpatiallyPartitioned(instruction->operand(0))) {
        return false;
      }
      return MaybeImproveInstructionSharding(
          hlo_sharding_util::ReverseSharding(
              instruction->operand(0)->sharding(), instruction->dimensions()),
          instruction);
    }
    case HloOpcode::kDot: {
      auto& dot_dim_numbs = instruction->dot_dimension_numbers();
      // Batch dimensions are the same for lhs and rhs on dot operations.
      int64 num_batch_dims = dot_dim_numbs.lhs_batch_dimensions_size();
      std::vector<int64> contracting_dims(2);
      contracting_dims[0] = dot_dim_numbs.lhs_contracting_dimensions(0);
      contracting_dims[1] = dot_dim_numbs.rhs_contracting_dimensions(0);
      std::vector<const HloSharding*> ops_sharding(2, nullptr);
      for (int64 op_num = 0; op_num < 2; ++op_num) {
        const HloInstruction* op = instruction->operand(op_num);
        if (IsSpatiallyPartitioned(op)) {
          ops_sharding[op_num] = &op->sharding();
        }
      }
      if (ops_sharding[0] == nullptr && ops_sharding[1] == nullptr) {
        return false;
      }

      // Select representative operand.
      int64 representative_op = -1;
      if (ops_sharding[0] == nullptr) {
        representative_op = 1;
      } else if (ops_sharding[1] == nullptr) {
        representative_op = 0;
      } else if (ops_sharding[0]->IsReplicated() &&
                 ops_sharding[1]->IsReplicated()) {
        // Both replicated -> replicate
        return MaybeImproveInstructionSharding(HloSharding::Replicate(),
                                               instruction);
      } else if (!ops_sharding[0]->IsReplicated() &&
                 !ops_sharding[1]->IsReplicated()) {
        // Both tile sharded. The dot spatial partitioning implementation
        // replicates the operand corresponding to the non-tiled dimension:
        // dot(lhs, rhs), sharding={devices=[1, ..., n, 1]} replicates rhs
        // dot(lhs, rhs), sharding={devices=[1, ..., 1, n]} replicates lhs
        // so set sharding in order to replicate the smaller of lhs and rhs
        representative_op =
            ShapeUtil::ByteSizeOf(instruction->operand(0)->shape()) <
                    ShapeUtil::ByteSizeOf(instruction->operand(1)->shape())
                ? 1
                : 0;
      } else {
        // One is replicated and the other is tiled - pick the tiled one.
        representative_op = ops_sharding[0]->IsReplicated() ? 1 : 0;
      }

      if (ops_sharding[representative_op]->IsReplicated()) {
        return MaybeImproveInstructionSharding(HloSharding::Replicate(),
                                               instruction);
      } else {
        // Tile-shard instruction according to representative op.
        auto sharding = *ops_sharding[representative_op];
        if (instruction->shape().dimensions_size() !=
            sharding.tile_assignment().num_dimensions()) {
          // It is necessarily the case of a matrix x vector, with
          // representative_op being the matrix, because the vector op has the
          // same shape as instruction.
          CHECK_EQ(sharding.tile_assignment().num_dimensions(),
                   instruction->shape().dimensions_size() + 1);
          // Reshape sharding so that last dimension is 1, and then remove
          // last dimension.
          std::vector<int64> non_batch_dims(
              sharding.tile_assignment().num_dimensions() - num_batch_dims);
          absl::c_iota(non_batch_dims, num_batch_dims);
          sharding = hlo_sharding_util::ReshapeToTileDimension(
              sharding, num_batch_dims, non_batch_dims);
          auto tile_assignment = sharding.tile_assignment();
          auto dimensions = tile_assignment.dimensions();
          CHECK_EQ(dimensions.back(), 1);
          dimensions.pop_back();
          tile_assignment.Reshape(dimensions);
          sharding = HloSharding::Tile(tile_assignment);
        }
        return MaybeImproveInstructionSharding(sharding, instruction);
      }
    }
    case HloOpcode::kParameter: {
      auto parent_it = computation_map.find(instruction->parent());
      if (parent_it == computation_map.end()) {
        return false;
      }
      const HloInstruction* parent = parent_it->second;
      switch (parent->opcode()) {
        case HloOpcode::kConditional: {
          for (int64 i = 1; i < parent->operand_count(); ++i) {
            if (parent->called_computations()[i - 1] == instruction->parent()) {
              if (parent->operand(i)->has_sharding()) {
                return MaybeImproveInstructionSharding(
                    parent->operand(i)->sharding(), instruction);
              }
              return false;
            }
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

      if (!operand->sharding().IsTileMaximal() &&
          operand->sharding().tile_assignment().dim(
              instruction->dimensions(0)) != 1) {
        // Doesn't support sharding the sorting dimension.
        return false;
      }

      if (instruction->shape().IsTuple()) {
        return MaybeImproveInstructionSharding(
            HloSharding::SingleTuple(instruction->shape(), operand->sharding()),
            instruction);
      } else {
        return MaybeImproveInstructionSharding(operand->sharding(),
                                               instruction);
      }
    }
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice: {
      auto propagate_slicing = [instruction]() {
        const HloInstruction* operand =
            instruction->opcode() == HloOpcode::kDynamicSlice
                ? instruction->operand(0)
                : instruction->operand(1);
        if (!IsSpatiallyPartitioned(operand)) {
          return false;
        }

        if (operand->sharding().IsReplicated()) {
          return MaybeImproveInstructionSharding(HloSharding::Replicate(),
                                                 instruction);
        }

        const auto& tile_assignment = operand->sharding().tile_assignment();
        for (int64 i = 0; i < instruction->shape().rank(); ++i) {
          if (tile_assignment.dim(i) > 1 &&
              instruction->shape().dimensions(i) !=
                  operand->shape().dimensions(i)) {
            return false;
          }
        }
        return MaybeImproveInstructionSharding(operand->sharding(),
                                               instruction);
      };
      auto propagate_base = [instruction]() {
        if (instruction->opcode() != HloOpcode::kDynamicUpdateSlice) {
          return false;
        }
        if (!IsSpatiallyPartitioned(instruction->operand(0))) {
          return false;
        }
        return MaybeImproveInstructionSharding(
            instruction->operand(0)->sharding(), instruction);
      };
      return propagate_slicing() || propagate_base();
    }
    case HloOpcode::kGather: {
      if (!IsSpatiallyPartitioned(instruction->operand(1))) {
        return false;
      }
      HloSharding new_sharding = hlo_sharding_util::GatherOutputSharding(
          instruction->operand(1)->sharding(), instruction);
      return MaybeImproveInstructionSharding(new_sharding, instruction);
    }
    case HloOpcode::kScatter: {
      if (!IsSpatiallyPartitioned(instruction->operand(1)) &&
          !IsSpatiallyPartitioned(instruction->operand(2))) {
        return false;
      }
      return MaybeImproveInstructionSharding(HloSharding::Replicate(),
                                             instruction);
    }
    case HloOpcode::kWhile: {
      if (!instruction->operand(0)->has_sharding()) {
        return false;
      }
      auto sharding = instruction->operand(0)->sharding();
      if (instruction->has_sharding()) {
        sharding =
            MergeForMoreSpecificSharding(sharding, instruction->sharding());
      }
      return MaybeImproveInstructionSharding(sharding, instruction);
    }
    default: {
      const HloInstruction* operand = PickRepresentativeOperand(instruction);
      if (!operand || !IsSpatiallyPartitioned(operand)) {
        return false;
      }
      return MaybeImproveInstructionSharding(operand->sharding(), instruction);
    }
  }
  return false;
}

// Return the sharding that should be propagated from user to instruction.
absl::optional<HloSharding> GetShardingFromUser(
    const HloInstruction& instruction, const HloInstruction& user,
    bool aggressive_prop, bool is_spmd) {
  if (!IsSpatiallyPartitioned(&user)) {
    return absl::nullopt;
  }
  switch (user.opcode()) {
    case HloOpcode::kBroadcast: {
      if (user.sharding().IsReplicated()) {
        return user.sharding();
      }
      // Only support when none of the partitioned dimensions in the broadcast
      // output belong to new dimensions.
      for (int64 i = 0; i < user.shape().rank(); ++i) {
        if (user.sharding().tile_assignment().dim(i) > 1 &&
            absl::c_count(user.dimensions(), i) == 0) {
          return absl::nullopt;
        }
      }

      // The instruction (operand of broadcast) will be tiled the same way
      // as the output.
      std::vector<int64> target_tile_assignment_dimensions;
      for (int64 output_dim : user.dimensions()) {
        target_tile_assignment_dimensions.push_back(
            user.sharding().tile_assignment().dim(output_dim));
      }
      Array<int64> new_tile_assignment = user.sharding().tile_assignment();
      new_tile_assignment.Reshape(target_tile_assignment_dimensions);
      return HloSharding::Tile(new_tile_assignment);
    }
    case HloOpcode::kConcatenate: {
      if (user.sharding().IsReplicated()) {
        return user.sharding();
      }

      const int64 cdim = user.concatenate_dimension();
      const Array<int64>& tile_assignment = user.sharding().tile_assignment();
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
      int64 start_offset = 0;
      for (HloInstruction* op : user.operands()) {
        if (op == &instruction) {
          break;
        }
        start_offset += op->shape().dimensions(cdim);
      }
      const int64 tile_shape = CeilOfRatio(user.shape().dimensions(cdim),
                                           tile_assignment.dimensions()[cdim]);
      std::vector<int64> start_indices(tile_assignment.num_dimensions());
      std::vector<int64> end_indices = tile_assignment.dimensions();
      start_indices[cdim] = start_offset / tile_shape;
      end_indices[cdim] = CeilOfRatio(
          start_offset + instruction.shape().dimensions(cdim), tile_shape);
      auto new_tile_assignment =
          tile_assignment.Slice(start_indices, end_indices);
      if (new_tile_assignment.num_elements() == 1) {
        return HloSharding::AssignDevice(*new_tile_assignment.begin());
      }
      return HloSharding::Tile(new_tile_assignment);
    }
    case HloOpcode::kConvolution: {
      if (auto dot_dims =
              dot_as_convolution_util::ParseDotGeneralFromConvolution(&user)) {
        const auto& dnums = user.convolution_dimension_numbers();
        auto partitioned_only_along_non_trivial_dims =
            [&](const HloSharding& sharding,
                std::vector<dot_as_convolution_util::
                                DotGeneralAsConvolutionDimsInfo::DimNums>&
                    dims) {
              if (sharding.IsTileMaximal()) {
                return false;
              }
              int64 partition_count = 1;
              for (const auto& dim : dims) {
                if (user.shape().dimensions(dim.output) == 1) {
                  continue;
                }
                partition_count *= sharding.tile_assignment().dim(dim.output);
              }
              return partition_count ==
                     sharding.tile_assignment().num_elements();
            };
        // If output is partitioned only along the batch dimensions, or only
        // along the non-contracting dimensions, propagate the sharding to the
        // operand.
        if (&instruction == user.operand(0) &&
            (partitioned_only_along_non_trivial_dims(user.sharding(),
                                                     dot_dims->batch_dims) ||
             partitioned_only_along_non_trivial_dims(
                 user.sharding(), dot_dims->lhs_non_contracting_dims))) {
          std::vector<int64> lhs_to_output_indices(user.shape().rank());
          lhs_to_output_indices[dnums.input_batch_dimension()] =
              dnums.output_batch_dimension();
          lhs_to_output_indices[dnums.input_feature_dimension()] =
              dnums.output_feature_dimension();
          for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
            lhs_to_output_indices[dnums.input_spatial_dimensions(i)] =
                dnums.output_spatial_dimensions(i);
          }
          return hlo_sharding_util::TransposeSharding(user.sharding(),
                                                      lhs_to_output_indices);
        }
        if (&instruction == user.operand(1) &&
            (partitioned_only_along_non_trivial_dims(user.sharding(),
                                                     dot_dims->batch_dims) ||
             partitioned_only_along_non_trivial_dims(
                 user.sharding(), dot_dims->rhs_non_contracting_dims))) {
          std::vector<int64> rhs_to_output_indices(user.shape().rank());
          rhs_to_output_indices[dnums.kernel_input_feature_dimension()] =
              dnums.output_batch_dimension();
          rhs_to_output_indices[dnums.kernel_output_feature_dimension()] =
              dnums.output_feature_dimension();
          for (int64 i = 0; i < dnums.input_spatial_dimensions_size(); ++i) {
            rhs_to_output_indices[dnums.kernel_spatial_dimensions(i)] =
                dnums.output_spatial_dimensions(i);
          }
          return hlo_sharding_util::TransposeSharding(user.sharding(),
                                                      rhs_to_output_indices);
        }
      }
      return absl::nullopt;
    }
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice: {
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
        return absl::nullopt;
      }

      const auto& tile_assignment = user.sharding().tile_assignment();
      for (int64 i = 0; i < user.shape().rank(); ++i) {
        if (tile_assignment.dim(i) > 1 &&
            user.shape().dimensions(i) != operand->shape().dimensions(i)) {
          return absl::nullopt;
        }
      }
      return user.sharding();
    }
    case HloOpcode::kReduceWindow: {
      if (&instruction != user.operand(0)) {
        return absl::nullopt;
      }
      return user.sharding();
    }
    case HloOpcode::kReshape: {
      return hlo_sharding_util::ReshapeSharding(
          user.shape(), instruction.shape(), user.sharding());
    }
    case HloOpcode::kTranspose: {
      // Calculate the dimension numbers for reversing the current transpose
      // and then use TransposeSharding to convert the output sharding to an
      // input sharding.
      std::vector<int64> reverse_dimensions(user.dimensions().size());
      for (int64 i = 0; i < user.dimensions().size(); ++i) {
        reverse_dimensions[user.dimensions(i)] = i;
      }
      return hlo_sharding_util::TransposeSharding(user.sharding(),
                                                  reverse_dimensions);
    }
    case HloOpcode::kTuple: {
      return user.sharding().GetSubSharding(user.shape(),
                                            {user.operand_index(&instruction)});
    }
    case HloOpcode::kGetTupleElement: {
      HloSharding new_sharding =
          instruction.has_sharding()
              ? instruction.sharding()
              : HloSharding::SingleTuple(instruction.shape(),
                                         HloSharding::Replicate());
      int64 sharding_index = 0;
      for (int64 i = 0; i < instruction.shape().tuple_shapes_size(); ++i) {
        if (i == user.tuple_index()) {
          break;
        }
        if (instruction.shape().tuple_shapes(i).IsArray()) {
          sharding_index += 1;
        } else {
          sharding_index +=
              instruction.shape().tuple_shapes(i).tuple_shapes_size();
        }
      }
      if (user.shape().IsArray()) {
        new_sharding.tuple_elements()[sharding_index] = user.sharding();
      }
      for (int64 i = 0; i < user.sharding().tuple_elements().size(); ++i) {
        new_sharding.tuple_elements()[sharding_index + i] =
            user.sharding().tuple_elements()[i];
      }
      return new_sharding;
    }
    case HloOpcode::kDot: {
      if (user.sharding().IsReplicated()) {
        return user.sharding();
      }
      auto& dim_numbers = user.dot_dimension_numbers();
      int64 op_idx = user.operand_index(&instruction);
      // Batch dimensions are the same on lhs and rhs for dot operations.
      int64 num_batch_dims = dim_numbers.lhs_batch_dimensions_size();
      int64 num_spatial_dims =
          instruction.shape().dimensions_size() - num_batch_dims;
      if (num_spatial_dims == 1) {
        // This is the vector of a matrix x vector operation -> replicate,
        // since tiling on the vector would necessarily be on the contracting
        // dimension, which we don't support.
        CHECK_EQ(op_idx, 1);
        return HloSharding::Replicate();
      }
      // Instruction is necessarily a matrix because it is one of the operands
      // of a matrix x matrix operation.
      CHECK_EQ(num_spatial_dims, 2);
      // Propagate tile sharding to the bigger operand, and replicate the other.
      auto other_op = user.operand(op_idx ^ 1);
      if (ShapeUtil::ByteSizeOf(instruction.shape()) >
          ShapeUtil::ByteSizeOf(other_op->shape())) {
        return user.sharding();
      } else {
        return HloSharding::Replicate();
      }
    }
    case HloOpcode::kReduce: {
      if (instruction.shape().rank() == 0) {
        return absl::nullopt;
      }
      auto user_sharding =
          user.shape().IsTuple()
              ? user.sharding().GetSubSharding(
                    user.shape(), {user.operand_index(&instruction)})
              : user.sharding();
      if (user_sharding.IsTileMaximal()) {
        return user_sharding;
      }
      std::vector<int64> target_tile_assignment_dimensions(
          instruction.shape().rank());
      const auto& dimensions = user.dimensions();
      int64 next_output_dim = 0;
      for (int64 i = 0; i < instruction.shape().rank(); ++i) {
        if (absl::c_find(dimensions, i) == dimensions.end()) {
          target_tile_assignment_dimensions[i] =
              user_sharding.tile_assignment().dim(next_output_dim++);
        } else {
          target_tile_assignment_dimensions[i] = 1;
        }
      }
      auto tile_assignment = user_sharding.tile_assignment();
      tile_assignment.Reshape(target_tile_assignment_dimensions);
      return HloSharding::Tile(tile_assignment);
    }
    case HloOpcode::kSort: {
      if (user.sharding().IsTuple()) {
        return user.sharding().GetSubSharding(
            user.shape(), {user.operand_index(&instruction)});
      } else {
        return user.sharding();
      }
    }
    case HloOpcode::kReverse: {
      return hlo_sharding_util::ReverseSharding(user.sharding(),
                                                user.dimensions());
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
      return absl::nullopt;
    }
  }
}

// Tries to update the sharding of the specified instruction based on its users
// and returns true if the sharding of the instruction have been changed and
// false otherwise.
bool InferShardingFromUsers(HloInstruction* instruction,
                            const ComputationMap& computation_map,
                            bool aggressive_prop, bool is_spmd) {
  if (!SupportSpatialPartitioning(instruction, computation_map, is_spmd)) {
    return false;
  }
  bool improved_sharding = false;
  for (const HloInstruction* user : instruction->users()) {
    absl::optional<HloSharding> user_sharding =
        GetShardingFromUser(*instruction, *user, aggressive_prop, is_spmd);
    if (user_sharding) {
      improved_sharding |=
          MaybeImproveInstructionSharding(*user_sharding, instruction);
    }
  }
  return improved_sharding;
}

// Remove Sharding custom-call instruction by folding the sharding attribute
// to its operand. If the operand alreayd has a different sharding, insert a
// copy node for reshard.
StatusOr<bool> ProcessShardingInstruction(HloModule* module) {
  bool changed = false;

  for (HloComputation* computation : module->computations()) {
    auto instructions = computation->MakeInstructionPostOrder();
    std::reverse(instructions.begin(), instructions.end());
    for (HloInstruction* instruction : instructions) {
      if (instruction->opcode() != HloOpcode::kCustomCall) {
        continue;
      }
      if (instruction->custom_call_target() != "Sharding") {
        continue;
      }
      TF_RET_CHECK(instruction->has_sharding())
          << "Sharding instruction must have a sharding attribute";
      const HloSharding& sharding = instruction->sharding();

      // If the operand has a different sharding from the current sharding
      // instruction, create a copy node. Otherwise, just remove the sharding
      // instruction and set the operand sharding.
      if (instruction->operand(0)->has_sharding() &&
          instruction->operand(0)->sharding() != sharding) {
        auto copy = computation->AddInstruction(
            HloInstruction::CreateUnary(instruction->shape(), HloOpcode::kCopy,
                                        instruction->mutable_operand(0)));
        TF_RETURN_IF_ERROR(computation->ReplaceInstruction(instruction, copy));
        copy->set_sharding(sharding);
      } else {
        instruction->mutable_operand(0)->set_sharding(sharding);
        TF_RETURN_IF_ERROR(
            instruction->ReplaceAllUsesWith(instruction->mutable_operand(0)));
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
      }
      changed = true;
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
  auto bad_status = [](HloInstruction* instruction, int64 device,
                       HloInstruction* channel_instruction,
                       int64 correct_device) {
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
  std::map<int64, HloInstruction*> devices_to_instructions;
  absl::optional<int64> unique_device = absl::nullopt;
  HloInstruction* channel_instruction = nullptr;

  for (HloInstruction* instruction : while_body->instructions()) {
    if (instruction->sharding_unique_device()) {
      auto opcode = instruction->opcode();
      int64 device = *instruction->sharding_unique_device();
      if (unique_device.has_value()) {
        if (*unique_device != device) {
          return bad_status(instruction, device, channel_instruction,
                            *unique_device);
        }
      } else if (opcode == HloOpcode::kSend || opcode == HloOpcode::kRecv ||
                 // Cross-replica AllReduces don't have a channel_id, and we
                 // don't enforce any invariant about their device assignment.
                 (opcode == HloOpcode::kAllReduce &&
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
  return Status::OK();
}

}  // namespace

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
        for (HloInstruction* domain : domain.exit_domains) {
          domain->mutable_operand(0)->set_sharding(*sharding);
        }
        return Status::OK();
      }
    }
  }
  return ShardingMetadata::NormalizeShardingDomain(domain, metadata);
}

StatusOr<bool> ShardingPropagation::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(bool any_changed, ProcessShardingInstruction(module));

  // Association of partitionable embedded computations with their parent
  // instruction.
  ComputationMap computation_map;

  // Instructions that are related through a computation and need to share the
  // same sharding.
  auto get_related_instructions = [](HloInstruction* inst) {
    if (inst->opcode() == HloOpcode::kWhile) {
      return std::vector<HloInstruction*>{
          inst, inst->while_body()->root_instruction(),
          inst->while_body()->parameter_instruction(0),
          inst->while_condition()->parameter_instruction(0)};
    } else if (inst->opcode() == HloOpcode::kConditional) {
      std::vector<HloInstruction*> comps{inst};
      for (HloComputation* c : inst->called_computations()) {
        comps.push_back(c->root_instruction());
      }
      return comps;
    } else {
      CHECK(false);
    }
  };

  // If instruction is a while, or the root or a parameter of a while body,
  // then propagate its sharding to the while instruction, to its body root,
  // and to its condition parameter.
  std::function<void(HloInstruction*)> maybe_computation_propagation =
      [&](HloInstruction* instruction) {
        auto propagate_to_instruction = [&](HloInstruction* search_inst) {
          auto related_instructions = get_related_instructions(search_inst);
          if (absl::c_count(related_instructions, instruction)) {
            for (HloInstruction* inst : related_instructions) {
              if (!inst->has_sharding() ||
                  inst->sharding() != instruction->sharding()) {
                VLOG(2) << "Add computation sharding: " << inst->name();
                inst->set_sharding(instruction->sharding());
                maybe_computation_propagation(inst);
              }
            }
          }
        };

        if (instruction->opcode() == HloOpcode::kConditional ||
            instruction->opcode() == HloOpcode::kWhile) {
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

  for (auto computation : module->computations()) {
    for (auto instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        TF_RETURN_IF_ERROR(
            CheckAndUpdateDeviceAssignmentsInWhileBody(instruction));
      }
    }
  }

  // Populate computation_map in order to associate while bodies to their
  // while instructions.
  for (auto computation : module->computations()) {
    for (auto instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kWhile ||
          instruction->opcode() == HloOpcode::kConditional) {
        // Check if any of the related instructions has sharding, in which case
        // propagate it to the other instructions, so they all share the same
        // sharding, in case the user didn't shard all of them. We don't check
        // that user shardings are consistent, because such check is already
        // done by HloShardingVerifier.
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
            inst->set_sharding(sharded_inst->sharding());
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
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* inst : computation->instructions()) {
      if (inst->has_sharding()) {
        provided_shardings.insert(inst);
      }
    }
  }

  // Consider the root instruction of the entry module as one with provided
  // sharding as its sharding have to match with the one expected by the host.
  provided_shardings.insert(module->entry_computation()->root_instruction());

  // Iterate to a fixpoint that is guaranteed to be reached because we only
  // strictly improve the sharding of the graph and it can't be improved
  // indefinitely.
  int64 iterations = 0;
  auto run_to_fix_point = [&](bool aggressive_prop) {
    bool changed = true;
    while (changed) {
      changed = false;
      int64 inferred_from_operand_counter = 0;
      int64 inferred_from_user_counter = 0;
      int64 instruction_counter = 0;
      int64 already_sharded_counter = 0;
      for (const HloComputation* computation : module->computations()) {
        std::vector<HloInstruction*> instructions =
            computation->MakeInstructionPostOrder();

        instruction_counter += instructions.size();
        for (const HloInstruction* instruction : instructions) {
          already_sharded_counter += (instruction->has_sharding() ? 1 : 0);
        }

        // Remove the instructions where the sharding was provided from the
        // outside so we don't modify them.
        instructions.erase(
            std::remove_if(instructions.begin(), instructions.end(),
                           [&](HloInstruction* instruction) {
                             return provided_shardings.contains(instruction);
                           }),
            instructions.end());

        // First iterate the HLO graph in post order taking shardings from
        // operands.
        for (HloInstruction* instruction : instructions) {
          if (InferShardingFromOperands(instruction, computation_map, is_spmd_,
                                        aggressive_prop)) {
            ++inferred_from_operand_counter;
            changed = true;
            VLOG(2) << "Add sharding (forward-pass): "
                    << instruction->ToString();
            maybe_computation_propagation(instruction);
          }
        }

        // Then iterate the HLO graph in reverse post order taking shardings
        // from users.
        for (auto it = instructions.rbegin(); it != instructions.rend(); ++it) {
          if (InferShardingFromUsers(*it, computation_map, aggressive_prop,
                                     is_spmd_)) {
            ++inferred_from_user_counter;
            changed = true;
            VLOG(2) << "Add sharding (backward-pass): " << (*it)->ToString();
            maybe_computation_propagation(*it);
          }
        }
      }
      any_changed |= changed;
      VLOG(1) << "Sharding propagation iteration " << iterations << ";";
      VLOG(1) << "  total instructions: " << instruction_counter;
      VLOG(1) << "  instructions already sharded: " << already_sharded_counter;
      VLOG(1) << "  shardings inferred from operands: "
              << inferred_from_operand_counter;
      VLOG(1) << "  shardings inferred from users: "
              << inferred_from_user_counter;
      ++iterations;
    }
  };
  run_to_fix_point(false);
  run_to_fix_point(true);

  VLOG(1) << "Sharding propagation completed after " << iterations
          << " iterations";
  return any_changed;
}

}  // namespace xla
