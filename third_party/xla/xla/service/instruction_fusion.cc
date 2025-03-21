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

#include "xla/service/instruction_fusion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
// The source_location.h is not available in open source.
#if defined(PLATFORM_GOOGLE)
#include "absl/types/source_location.h"
#endif  // PLATFORM_GOOGLE
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/map_util.h"
#include "xla/service/fusion_queue.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace xla {

#if defined(PLATFORM_GOOGLE)
FusionDecision::FusionDecision(bool decision,
                               absl::SourceLocation source_location) {
  if (!decision) {
    explanation_ =
        absl::StrCat("Not fusing: due to ", source_location.file_name(), ":",
                     source_location.line());
  }
}
#endif  // PLATFORM_GOOGLE

namespace {

// These nodes can always be duplicated into consumers, even if
// InstructionFusion::may_duplicate_ is false.
//
// In general these should be nodes that get *cheaper* the more they're
// duplicated (and fused into consumers).
//
// TODO(jlebar): Duplicating instructions when we have a variable called "may
// duplicate" that's equal to false is not pretty.
bool IsAlwaysDuplicable(const HloInstruction& instruction) {
  // We are always willing to duplicate a widening type-conversion instruction
  // if it means we can fuse the convert into a consumer.  This allows the
  // consumer to read less memory, which is almost always a performance win.
  return (instruction.opcode() == HloOpcode::kConvert &&
          ShapeUtil::ByteSizeOf(instruction.operand(0)->shape()) <
              ShapeUtil::ByteSizeOf(instruction.shape())) ||
         instruction.opcode() == HloOpcode::kBroadcast;
}
}  // namespace

/*static*/ bool InstructionFusion::IsExpensive(
    const HloInstruction& instruction) {
  namespace m = match;

  switch (instruction.opcode()) {
    // Cheap instructions.
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConstant:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kFloor:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kImag:
    case HloOpcode::kInfeed:
    case HloOpcode::kIota:
    case HloOpcode::kIsFinite:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kOutfeed:
    case HloOpcode::kPad:
    case HloOpcode::kPartitionId:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kReplicaId:
    case HloOpcode::kReshape:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSlice:
    case HloOpcode::kStochasticConvert:
    case HloOpcode::kSubtract:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
      return false;

    // Cheap instructions for reals, but expensive for complex.
    case HloOpcode::kAbs:
    case HloOpcode::kCos:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kTan:
      return ShapeUtil::ElementIsComplex(instruction.shape());

    // We say that integer div/mod by a constant is cheap because it gets
    // compiled down to multiplies and shifts, and we consider those to be
    // cheap.
    case HloOpcode::kDivide:
    case HloOpcode::kRemainder:
      return !ShapeUtil::ElementIsIntegral(instruction.shape()) ||
             !Match(instruction.operand(1),
                    m::AnyOf<const HloInstruction>(
                        m::ConstantEffectiveScalar(),
                        m::Broadcast(m::ConstantEffectiveScalar())));

    // Expensive instructions or unusual instructions for which fusion is
    // nonsensical.
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    case HloOpcode::kAtan2:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kCall:
    case HloOpcode::kCholesky:
    case HloOpcode::kConditional:
    case HloOpcode::kConvolution:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kDot:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFft:
    case HloOpcode::kFusion:
    case HloOpcode::kGather:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kLogistic:
    case HloOpcode::kMap:
    case HloOpcode::kParameter:
    case HloOpcode::kPower:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kRng:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kRsqrt:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kSort:
    case HloOpcode::kTopK:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kTanh:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kWhile:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
      return true;
  }

  return false;
}

// An "effectively at most unary" operation is one that has at most one "large"
// input with the others being negligible in terms of memory usage.
// We use "has a smaller true rank than the output" as a heuristic
// for "negligible" memory usage.
bool InstructionFusion::EffectivelyAtMostUnary(HloInstruction* hlo) {
  int64_t output_rank = 0;
  ShapeUtil::ForEachSubshape(
      hlo->shape(),
      [&output_rank](const Shape& subshape, const ShapeIndex& shape_index) {
        if (subshape.IsArray()) {
          output_rank = std::max(output_rank, ShapeUtil::TrueRank(subshape));
        }
      });
  return absl::c_count_if(
             hlo->operands(), [output_rank](HloInstruction* operand) {
               if (operand->opcode() == HloOpcode::kBroadcast ||
                   operand->opcode() == HloOpcode::kIota) {
                 return false;
               }
               if (operand->opcode() == HloOpcode::kConstant &&
                   ShapeUtil::IsEffectiveScalar(operand->shape())) {
                 return false;
               }
               return ShapeUtil::TrueRank(operand->shape()) >= output_rank;
             }) <= 1;
}

bool InstructionFusion::CanFuseOnAllPaths(
    HloInstruction* producer, HloInstruction* consumer,
    const HloInstructionSet& do_not_fuse,
    const HloReachabilityMap& reachability,
    absl::flat_hash_map<std::pair<HloInstruction*, HloInstruction*>, bool>*
        result_cache) {
  if (consumer == producer) {
    return true;
  }
  if (!consumer->IsFusible()) {
    return false;
  }
  auto cache_it = result_cache->find(std::make_pair(producer, consumer));
  if (cache_it != result_cache->end()) {
    return cache_it->second;
  }
  bool result = true;
  for (int64_t i = 0, e = consumer->operand_count(); i < e; ++i) {
    auto* consumer_operand = consumer->mutable_operand(i);
    // If the operand is not on a path to the producer, it doesn't matter
    // whether it's fusible.
    if (!reachability.IsReachable(producer, consumer_operand)) {
      continue;
    }
    if (do_not_fuse.count(consumer_operand) > 0 || !ShouldFuse(consumer, i)) {
      result = false;
      break;
    }
    // The producer is reachable from consumer_operand which means we need
    // to be able to fuse consumer_operand into consumer in order for
    // producer to be fusible into consumer on all paths.
    // Perform the recursive step: make sure producer can be fused into
    // consumer_operand on all paths.
    if (!CanFuseOnAllPaths(producer, consumer_operand, do_not_fuse,
                           reachability, result_cache)) {
      result = false;
      break;
    }
  }
  result_cache->emplace(std::make_pair(producer, consumer), result);
  return result;
}

InstructionFusion::HloInstructionSet
InstructionFusion::ComputeGloballyUnfusible(
    absl::Span<HloInstruction* const> post_order,
    const HloReachabilityMap& reachability) {
  // Forbid fusion of producers that:
  // a) Need to be duplicated, unless they can be fused into all consumers
  //    via all paths.
  // b) Are more than unary, that is, fusing them would likely lead to an
  //    increase in memory bandwidth use.
  //
  // Note that if we allow fusion by these global rules, we may still forbid
  // fusing operations that require duplication later depending on
  // is_expensive_().
  HloInstructionSet do_not_duplicate;
  absl::flat_hash_map<std::pair<HloInstruction*, HloInstruction*>, bool>
      can_fuse_on_all_paths_result_cache;
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    HloInstruction* producer = *it;
    // If the producer is effectively not more than unary, duplicating it
    // will not increase the number of relevant inputs read, as the fusion
    // node will only need to read at most 1 relevant input (the input of
    // the producer). In that case, we do not forbid fusion of the operation
    // here.
    if (EffectivelyAtMostUnary(producer)) {
      continue;
    }

    // If the total size of the inputs is less than or equal to the total size
    // of the outputs for the producer then duplicating it won't increase the
    // memory traffic. In that case, we do not forbid fusion of the operation
    // here.
    auto total_size = [](const Shape& shape) {
      int64_t size = 0;
      ShapeUtil::ForEachSubshape(
          shape, [&size](const Shape& subshape, const ShapeIndex& shape_index) {
            if (subshape.IsArray()) {
              size += ShapeUtil::ElementsIn(subshape);
            }
          });
      return size;
    };
    int64_t operands_size = 0;
    for (const HloInstruction* op : producer->unique_operands()) {
      operands_size += total_size(op->shape());
    }
    if (operands_size <= total_size(producer->shape())) {
      continue;
    }

    // Otherwise we will forbid fusing the op unless we can fuse it into
    // all of its consumers on all paths.
    //
    // That means, that for:
    // A --> B (fusible)
    //   \-> C (non-fusible)
    // A will be not allowed to be fused into B, as it cannot be fused into C.
    //
    // Similarly, for:
    // A -------------> B
    //   \-> C -> D -/
    // If:
    // - A is fusible into B and C, and D is fusible into B
    // - C is *not* fusible into D
    // A will be not allowed to be fused into B, as it cannot be fused via
    // all paths.
    if (producer->IsFusible() &&
        absl::c_all_of(producer->users(), [&](HloInstruction* consumer) {
          return CanFuseOnAllPaths(producer, consumer, do_not_duplicate,
                                   reachability,
                                   &can_fuse_on_all_paths_result_cache);
        })) {
      continue;
    }
    do_not_duplicate.insert(producer);
  }

  return do_not_duplicate;
}

namespace {

// A FusionQueue that uses reverse post order.
//
// We want to be able to remove arbitrary instructions from the post order and
// also compare positions of instructions in the post order. To make this
// possible, create vector of instructions in post order and create a map from
// HloInstruction* to the instruction's index in the vector. An instruction is
// "removed" from the vector by setting it's element to nullptr.
class ReversePostOrderFusionQueue : public FusionQueue {
 public:
  explicit ReversePostOrderFusionQueue(HloComputation* computation) {
    post_order_ = computation->MakeInstructionPostOrder();

    for (size_t i = 0; i < post_order_.size(); ++i) {
      InsertOrDie(&post_order_index_, post_order_[i], i);
    }
  }

  std::pair<HloInstruction*, std::vector<int64_t>>
  DequeueNextInstructionAndOperandsToFuseInOrder() override {
    // Instructions are "removed" from the post order by nulling out the element
    // in the vector, so if the pointer is null, continue to the next
    // instruction in the sort.
    while (!post_order_.empty() && post_order_.back() == nullptr) {
      post_order_.pop_back();
    }
    if (post_order_.empty()) {
      return std::pair<HloInstruction*, std::vector<int64_t>>{nullptr, {}};
    }
    // We want to iterate in reverse post order, so remove from the back of the
    // vector.
    HloInstruction* instruction = post_order_.back();
    post_order_.pop_back();

    CHECK(instruction != nullptr);
    // Remove instruction from the index map to ensure the vector and map stay
    // consistent.
    post_order_index_.erase(instruction);

    // Consider each operand of this instruction for fusion into this
    // instruction. We want to consider the operands in a particular order to
    // avoid creating duplicate instruction clones in the fusion instruction.
    // For example, consider the following expression:
    //
    //   A = ...
    //   B = op(A)
    //   C = op(A, B)
    //
    // If we are considering the operands of C for fusion into C. We might
    // fuse A or B first. If we fuse A first, we get:
    //
    //   A = ...
    //   B = op(A)
    //   C_fusion = { A' = ...
    //                C' = op(A', B) }
    //
    // Where A' and C' are clones of A and C, respectively. Now only B is an
    // operand of the fusion instruction C_fusion, so then we fuse B:
    //
    //   A = ...
    //   B = op(A)
    //   C_fusion = { A' = ...
    //                B' = op(A)
    //                C' = op(A', B') }
    //
    // Now A is an operand of C_fusion again, so we then fuse A (again!):
    //
    //   A = ...
    //   B = op(A)
    //   C_fusion = { A' = ...
    //                A" = ..
    //                B' = op(A")
    //                C' = op(A', B') }
    //
    // We prevent this duplication by considering the operands in the order
    // they appear int the queue. In the example, this ensures that B will be
    // considered before A.
    //
    // We store the original indices of the operands to pass to ShouldFuse.
    std::vector<int64_t> sorted_operand_numbers;
    sorted_operand_numbers.reserve(instruction->operands().size());
    for (int i = 0; i < instruction->operands().size(); ++i) {
      // This will happen if we have two possible instructions to fuse the
      // same operand into; once the operand is fused into one instruction,
      // the other instruction will get a new get-tuple-element as its
      // operand, which is not in the queue.
      // TODO(tjoerg): Look into fusing past these multi-output fuse points.
      if (!ContainsKey(post_order_index_, instruction->mutable_operand(i))) {
        continue;
      }
      sorted_operand_numbers.push_back(i);
    }
    absl::c_sort(sorted_operand_numbers, [&](int64_t i, int64_t j) {
      // Instructions with higher priority in the queue come first.
      return (FindOrDie(post_order_index_, instruction->mutable_operand(i)) >
              FindOrDie(post_order_index_, instruction->mutable_operand(j)));
    });
    return std::make_pair(instruction, sorted_operand_numbers);
  }

  void OnFusingInstruction(HloInstruction* fusion,
                           HloInstruction* original_producer,
                           HloInstruction* original_consumer) override {
    // Fusing an instruction into a fusion instruction can change the operand
    // set of the fusion instruction. For simplicity just re-enqueue the
    // instruction and reconsider it for further fusion in the next iteration.
    InsertOrDie(&post_order_index_, fusion, post_order_.size());
    post_order_.push_back(fusion);
  }

  void RemoveInstruction(HloInstruction* instruction) override {
    post_order_[FindOrDie(post_order_index_, instruction)] = nullptr;
    post_order_index_.erase(instruction);
  }

  const std::vector<bool>* FusionConfiguration() override {
    return &fusion_config_;
  }

 private:
  std::vector<HloInstruction*> post_order_;
  absl::flat_hash_map<HloInstruction*, int> post_order_index_;
  std::vector<bool> fusion_config_;
};

bool MultiOutputFusionCreatesCycle(HloInstruction* producer,
                                   HloInstruction* consumer,
                                   const HloReachabilityMap& reachability) {
  absl::flat_hash_set<int> operands;
  auto insert = [&](const HloInstruction* operand) {
    if (operand == producer) {
      return false;
    }

    // If the reachability map already contains the producer and the operand of
    // the consumer, and the producer can reach the operand, then we know for
    // sure MultiOutputFusion would create a cycle. If not, we need to do a DFS
    // traversal of the computation to verify that this multioutput fusion would
    // not create a cycle.
    if (reachability.IsPresent(producer) && reachability.IsPresent(operand) &&
        reachability.IsReachable(producer, operand)) {
      return true;
    }
    operands.insert(operand->unique_id());
    return false;
  };

  for (const HloInstruction* operand : consumer->operands()) {
    if (insert(operand)) {
      return true;
    }
  }
  for (const HloInstruction* predecessor : consumer->control_predecessors()) {
    if (insert(predecessor)) {
      return true;
    }
  }

  // Do a DFS on the producer to see if any of the other consumer operands are
  // reachable in the current state of the graph.
  std::vector<HloInstruction*> worklist = producer->users();
  worklist.insert(worklist.end(), producer->control_successors().begin(),
                  producer->control_successors().end());
  absl::flat_hash_set<int> visits;
  while (!worklist.empty()) {
    const HloInstruction* user = worklist.back();
    worklist.pop_back();
    if (operands.count(user->unique_id()) != 0) {
      return true;
    }
    if (visits.count(user->unique_id()) == 0) {
      visits.insert(user->unique_id());
      worklist.insert(worklist.end(), user->users().begin(),
                      user->users().end());
      worklist.insert(worklist.end(), user->control_successors().begin(),
                      user->control_successors().end());
    }
  }
  return false;
}

}  // namespace

std::vector<HloComputation*> InstructionFusion::GetNonFusionComputations(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return module->MakeNonfusionComputations(execution_threads);
}

std::unique_ptr<FusionQueue> InstructionFusion::GetFusionQueue(
    HloComputation* computation) {
  return std::make_unique<ReversePostOrderFusionQueue>(computation);
}

absl::StatusOr<bool> InstructionFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  int64_t fuse_count = 0;
  std::vector<std::vector<bool>> fusion_config;

  bool dump_fusion =
      module->config().debug_options().xla_dump_fusion_visualization();

  for (auto* computation :
       GetNonFusionComputations(module, execution_threads)) {
    CHECK(!computation->IsFusionComputation());
    std::unique_ptr<HloReachabilityMap> reachability =
        HloReachabilityMap::Build(computation);

    HloInstructionSet do_not_duplicate;
    // If we allow duplications, we need to compute which instructions we do not
    // want to duplicate based on a global analysis of the graph.
    if (may_duplicate_) {
      do_not_duplicate = ComputeGloballyUnfusible(
          computation->MakeInstructionPostOrder(), *reachability);
    }
    auto fusion_queue = GetFusionQueue(computation);

    // Instruction fusion effectively fuses edges in the computation graph
    // (producer instruction -> consumer instruction) so we iterate over all
    // edges. When we fuse an edge, we create a copy of the producer inside the
    // fusion instruction.
    while (true) {
      std::pair<HloInstruction*, std::vector<int64_t>> next_entry =
          fusion_queue->DequeueNextInstructionAndOperandsToFuseInOrder();
      HloInstruction* instruction = next_entry.first;
      if (instruction == nullptr) {
        break;
      }

      if (!instruction->IsFusible() &&
          instruction->opcode() != HloOpcode::kFusion) {
        continue;
      }

      std::vector<int64_t>& sorted_operand_numbers = next_entry.second;

      for (int64_t i : sorted_operand_numbers) {
        HloInstruction* operand = instruction->mutable_operand(i);
        VLOG(5) << "Considering fusion of: " << instruction->ToString()
                << " with operand " << operand->name();

        if (!operand->IsFusible()) {
          VLOG(3) << "Operand (" << operand->ToString() << ") is not fusible";
          continue;
        }

        // Consumes a unit of compiler fuel and returns true if we should
        // continue with the transformation.
        auto consume_fuel = [&] {
          return ConsumeFuel(name(), /*ran_out_of_fuel_msg=*/[&] {
            return absl::StrFormat("Not fusing operand %d of %s, namely, %s", i,
                                   instruction->ToString(),
                                   operand->ToString());
          });
        };

        HloInstruction* fusion_instruction = nullptr;

        // Try "regular" fusion if the operand may be duplicated. Otherwise,
        // perform multi-output fusion, unless this creates a cycle.
        FusionDecision use_regular_fusion(do_not_duplicate.count(operand) == 0,
                                          "operand can not be duplicated");
        if (use_regular_fusion) {
          use_regular_fusion =
              use_regular_fusion.And(ShouldFuse(instruction, i));
        }

        if (use_regular_fusion && consume_fuel()) {
          if (dump_fusion) {
            DumpPreFusionState(computation, /*consumer=*/instruction,
                               /*producer=*/operand);
          }

          fusion_queue->PreFusion(operand, instruction);
          fusion_instruction = Fuse(operand, instruction, computation);
        } else {
          VLOG(3) << "not fusing operand " << operand->ToString() << " because "
                  << use_regular_fusion.Explain();
        }

        FusionDecision use_mof = FusionDecision::Allow();
        if (!use_regular_fusion) {
          use_mof = ShouldFuseIntoMultiOutput(instruction, i);
          if (use_mof) {
            use_mof = use_mof.And(
                FusionDecision{!MultiOutputFusionCreatesCycle(
                                   operand, instruction, *reachability),
                               "multi-output fusion creates a cycle"});
          }
          if (use_mof && consume_fuel()) {
            if (dump_fusion) {
              DumpPreFusionState(computation, /*consumer=*/instruction,
                                 /*producer=*/operand, /*is_mof=*/true);
            }

            fusion_queue->PreFusion(operand, instruction);
            fusion_instruction =
                FuseIntoMultiOutput(operand, instruction, computation);
          }
        }

        if (fusion_instruction == nullptr) {
          FusionDecision fusion_decision = use_regular_fusion.Or(use_mof);
          CHECK(!fusion_decision.CanFuse());
          if (dump_fusion) {
            VLOG(2) << "Not fusing " << operand->ToShortString() << "| into |"
                    << instruction->ToShortString() << "| as "
                    << fusion_decision.Explain();

            DumpNotFusingState(computation,
                               /*consumer=*/instruction,
                               /*producer=*/operand,
                               /*decision=*/fusion_decision);
          }

          fusion_queue->NotFusingInstruction(operand, instruction);
          continue;
        }

        // Saving name to use after the instruction is removed.
        std::string producer_name(operand->name());
        fusion_queue->OnFusingInstruction(fusion_instruction, operand,
                                          instruction);
        changed = true;
        ++fuse_count;

        if (operand->user_count() == 0) {
          do_not_duplicate.erase(operand);
          // Operand is now dead. Remove from queue.
          fusion_queue->RemoveInstruction(operand);
          // Remove from computation.
          TF_RETURN_IF_ERROR(computation->RemoveInstruction(operand));
        }

        if (dump_fusion) {
          DumpStateAfterFusion(computation, fusion_instruction, producer_name);
        }

        if (fusion_instruction != instruction) {
          do_not_duplicate.erase(instruction);
        }
        break;
      }
    }

    if (config_collection_mode_ != FusionConfigCollection::kOff) {
      const std::vector<bool>* comp_fusion_config =
          fusion_queue->FusionConfiguration();
      if (comp_fusion_config && !comp_fusion_config->empty()) {
        fusion_config.push_back(*comp_fusion_config);
      }
    }
  }

  if (config_collection_mode_ != FusionConfigCollection::kOff) {
    if (VLOG_IS_ON(1)) {
      int64_t fused_count = 0;
      for (auto& config_per_computation : fusion_config) {
        fused_count += std::count(config_per_computation.begin(),
                                  config_per_computation.end(), true);
      }
      VLOG(1) << "There are " << fused_count << " fused bits that cause "
              << fuse_count << " fusion actions.";
    }
    module->mutable_config().set_fusion_config(std::move(fusion_config));
  }

  VLOG(1) << "Fusion count: " << fuse_count;

  return changed;
}

HloInstruction* InstructionFusion::AddFusionInstruction(
    HloInstruction* producer, HloInstruction* consumer,
    HloComputation* computation) {
  HloInstruction* fusion_instruction;
  auto kind = ChooseKind(producer, consumer);
  if (consumer->opcode() == HloOpcode::kFusion) {
    fusion_instruction = consumer;
    if (kind != fusion_instruction->fusion_kind()) {
      fusion_instruction->set_fusion_kind(kind);
    }
  } else {
    fusion_instruction =
        computation->AddInstruction(HloInstruction::CreateFusion(
            consumer->shape(), kind, consumer,
            absl::StrCat(HloOpcodeString(producer->opcode()), "_",
                         HloOpcodeString(consumer->opcode()), "_")));
    TF_CHECK_OK(computation->ReplaceInstruction(consumer, fusion_instruction));
  }
  fusion_instruction->set_called_computations_execution_thread(
      computation->execution_thread(),
      /*skip_async_execution_thread_overwrite=*/false);
  return fusion_instruction;
}

HloInstruction* InstructionFusion::FuseInstruction(
    HloInstruction* fusion_instruction, HloInstruction* producer) {
  return fusion_instruction->FuseInstruction(producer);
}

void InstructionFusion::UpdateReusedOperandsForFusion(
    HloInstruction* producer, HloInstruction* fusion_instruction) {
  // Find or compute the existing fusion reused operands. Note these reflect the
  // state *before* the current fusion has taken place, although if we have
  // replaced the consumer with a new single-element fusion, we will compute
  // the new single-element fusion's reused operands here.
  absl::flat_hash_set<const HloInstruction*>& fusion_reused_operands =
      ReusedOperandsOf(fusion_instruction);

  // If the producer is reused, replace it with its operands.
  if (fusion_reused_operands.erase(producer)) {
    fusion_reused_operands.insert(producer->operands().begin(),
                                  producer->operands().end());
  } else {
    const absl::flat_hash_set<const HloInstruction*>& producer_reused_operands =
        ReusedOperandsOf(producer);
    // Otherwise add the producer's reused operands to the set.
    fusion_reused_operands.insert(producer_reused_operands.begin(),
                                  producer_reused_operands.end());
  }
}

HloInstruction* InstructionFusion::Fuse(HloInstruction* producer,
                                        HloInstruction* consumer,
                                        HloComputation* computation) {
  VLOG(2) << "Fusing " << producer->ToString() << " into "
          << consumer->ToString();
  HloInstruction* fusion_instruction =
      AddFusionInstruction(producer, consumer, computation);
  UpdateReusedOperandsForFusion(producer, fusion_instruction);
  FuseInstruction(fusion_instruction, producer);
  if (fusion_instruction != producer && fusion_instruction != consumer) {
    VLOG(2) << "       created new fusion: " << fusion_instruction->ToString();
  }
  return fusion_instruction;
}

HloInstruction* InstructionFusion::FuseIntoMultiOutput(
    HloInstruction* producer, HloInstruction* consumer,
    HloComputation* computation) {
  VLOG(2) << "Multi-output fusing " << producer->ToString() << " into "
          << consumer->ToString();
  HloInstruction* fusion_instruction =
      AddFusionInstruction(producer, consumer, computation);
  UpdateReusedOperandsForFusion(producer, fusion_instruction);
  fusion_instruction->FuseInstructionIntoMultiOutput(producer);
  return fusion_instruction;
}

namespace {

// Extracts instructions from the fusion that satisfy the filter.
std::vector<const HloInstruction*> ExtractInstructions(
    const HloInstruction* hlo, const HloPredicate& filter) {
  if (filter(hlo)) {
    return {hlo};
  }
  if (hlo->opcode() != HloOpcode::kFusion) {
    return {};
  }
  std::vector<const HloInstruction*> matches;
  for (HloInstruction* inst :
       hlo->fused_instructions_computation()->instructions()) {
    if (filter(inst)) {
      matches.push_back(inst);
    }
  }
  return matches;
}

std::vector<const HloInstruction*> ExtractInstructions(
    const HloInstruction* hlo, HloOpcode opcode) {
  return ExtractInstructions(hlo, [opcode](const HloInstruction* inst) {
    return inst->opcode() == opcode;
  });
}

// Returns true if fusing a slice or dynamic slice in producer into a dynamic
// update slice fusion in consumer is safe. It is not safe to fuse the slice and
// DUS when fusing will cause another non-elementwise op to share operands with
// the DUS in-place buffer.
bool IsSafeToFuseSliceIntoDusFusion(const HloInstruction* producer,
                                    const HloInstruction* consumer,
                                    const HloInstruction* dus) {
  // Use a memoization map to avoid exponential runtime.
  absl::flat_hash_map<const HloInstruction*, bool> nonelementwise_memo;
  // Recursively check if the instruction or its users (or their users) are
  // non-elementwise with the exception of the DUS. We have already verified
  // that the slice and DUS are compatible since their indices match.
  HloPredicate has_nonelementwise_uses_except_dus =
      [&](const HloInstruction* instruction) {
        auto record_and_return = [&](bool val) {
          nonelementwise_memo[instruction] = val;
          return val;
        };
        auto nonelementwise_memo_it = nonelementwise_memo.find(instruction);
        if (nonelementwise_memo_it != nonelementwise_memo.end()) {
          return nonelementwise_memo_it->second;
        }
        if (instruction != dus && !instruction->IsElementwise() &&
            instruction->opcode() != HloOpcode::kParameter) {
          return record_and_return(true);
        }
        return record_and_return(absl::c_any_of(
            instruction->users(), has_nonelementwise_uses_except_dus));
      };
  for (int i = 0; i < consumer->operand_count(); ++i) {
    if (consumer->operand(i) == producer &&
        has_nonelementwise_uses_except_dus(consumer->fused_parameter(i))) {
      VLOG(4) << "Found a different elementwise";
      return false;
    }
  }
  return true;
}

}  // namespace

/*static*/ FusionDecision InstructionFusion::ShouldFuseInPlaceOp(
    const HloInstruction* producer, const HloInstruction* consumer,
    std::optional<const InPlaceFusionOptions> in_place_fusion_options) {
  // Don't fuse if the producer is a non-elementwise op that has the same
  // operand as an in-place operand of the consumer. The consumer will modify
  // the buffer in-place, which will cause producer's operand to change if we
  // allow them to fuse.
  std::vector<std::pair<HloOperandIndex, ShapeIndex>>
      in_place_input_output_pairs =
          HloDataflowAnalysis::GetInPlaceInputOutputPairs(
              const_cast<HloInstruction*>(consumer));
  for (auto& pair : in_place_input_output_pairs) {
    int operand_number = pair.first.operand_number;
    VLOG(4) << "in/out pair: " << operand_number << " "
            << pair.first.operand_index.ToString() << " "
            << pair.second.ToString();
    // Check if the consumer also has an additional operand that has the same
    // value as the in-place buffer. If so, it's unsafe to fuse.
    for (int i = 0; i < consumer->operand_count(); ++i) {
      if (i != operand_number &&
          consumer->operand(operand_number) == consumer->operand(i)) {
        return FusionDecision::Forbid(
            "The consumer is an in-place operation that has an additional "
            "operand that has the same value as the in-place buffer");
      }
    }
    if (consumer->operand(operand_number) == producer ||
        absl::c_find(producer->operands(), consumer->operand(operand_number)) !=
            producer->operands().end()) {
      VLOG(4) << "Found producer that uses the same operand of "
                 "an in-place consumer";
      auto get_real_operand = [](const HloInstruction* op,
                                 const HloInstruction* operand) {
        if (op->opcode() == HloOpcode::kFusion &&
            operand->opcode() == HloOpcode::kParameter) {
          return op->operand(operand->parameter_number());
        }
        return operand;
      };

      auto get_constant_operand =
          [](const HloInstruction* operand) -> std::optional<int> {
        if (operand->IsConstant()) {
          return operand->literal().GetFirstInteger();
        }
        return std::nullopt;
      };
      // A common special case is a slice or dynamic-slice and a
      // dynamic-update-slice that use the same indices. This pattern is safe.

      auto is_nonelementwise_op = [](const HloInstruction* inst) {
        return inst->opcode() != HloOpcode::kFusion && !inst->IsElementwise() &&
               inst->opcode() != HloOpcode::kBitcast &&
               inst->opcode() != HloOpcode::kParameter;
      };
      auto producer_nonelementwise_ops =
          ExtractInstructions(producer, [&](const HloInstruction* inst) {
            return is_nonelementwise_op(inst);
          });
      if ((producer_nonelementwise_ops.size() > 1) &&
          !(in_place_fusion_options.has_value() &&
            in_place_fusion_options->relax_multiple_non_elementwise_ops)) {
        return FusionDecision::Forbid(
            "Producer fusion has multiple non-elementwise ops, bailing.");
      }
      // If the producer has only elementwise ops or bitcasts, we can fuse.
      if (producer_nonelementwise_ops.empty()) {
        if (consumer->opcode() != HloOpcode::kFusion) {
          return FusionDecision::Allow();
        }
        // If the consumer fusion has both elementwise and non-elementwise ops,
        // and ops of the two groups access the same buffer of the producer, we
        // have a potential conflict even if all the ops in the producer are
        // elementwise.
        auto inplace_instr_and_index = FollowTupleIndirection(
            consumer->fused_expression_root(), pair.second);
        auto consumer_nonelementwise_ops =
            ExtractInstructions(consumer, [&](const HloInstruction* inst) {
              // Exclude instructions which are same as inplace fusion roots.
              return is_nonelementwise_op(inst) &&
                     inst != inplace_instr_and_index.first;
            });
        auto* fused_computation = consumer->fused_instructions_computation();
        std::unique_ptr<HloReachabilityMap> reachability =
            HloReachabilityMap::Build(fused_computation);
        auto inplace_consumer_parameter =
            fused_computation->parameter_instruction(
                consumer->operand_index(producer));
        bool inplace_conflict_after_fusion = absl::c_any_of(
            consumer_nonelementwise_ops, [&](const HloInstruction* instr) {
              return reachability->IsReachable(inplace_consumer_parameter,
                                               instr);
            });
        return inplace_conflict_after_fusion
                   ? FusionDecision::Forbid(
                         "Non-elementwise ops in consumer lead to inplace "
                         "conflict after fusion.")
                   : FusionDecision::Allow();
      }
      auto dus_ops =
          ExtractInstructions(consumer, HloOpcode::kDynamicUpdateSlice);
      // Check whether we have a DUS, in which case we should not fuse to make
      // sure we can compute the fusion in place.
      // TODO(akuegel): Are there other ops than dynamic update slice where we
      // have a special emitter if it can be done in-place?
      if (dus_ops.empty()) {
        return FusionDecision::Allow();
      }
      if (dus_ops.size() > 1) {
        return FusionDecision::Forbid("multiple dus ops, bailing.");
      }
      auto dus = dus_ops[0];
      auto producer_nonelementwise = producer_nonelementwise_ops[0];
      if (producer_nonelementwise->opcode() == HloOpcode::kSlice) {
        if (producer_nonelementwise->shape() != dus->operand(1)->shape()) {
          return FusionDecision::Forbid(
              "Slice op has a different shape than the update shape of the "
              "dus op, bailing.");
        }
        for (int i = 0; i < dus->shape().dimensions_size(); ++i) {
          const HloInstruction* dus_operand =
              get_real_operand(consumer, dus->operand(2 + i));
          auto constant_operand = get_constant_operand(dus_operand);
          if (!constant_operand ||
              *constant_operand != producer_nonelementwise->slice_starts(i) ||
              producer_nonelementwise->slice_strides(i) != 1) {
            return FusionDecision::Forbid("DUS and slice index mismatch");
          }
        }
        VLOG(4) << "DUS and slice index match";
        if (consumer->opcode() == HloOpcode::kFusion &&
            !IsSafeToFuseSliceIntoDusFusion(producer, consumer, dus)) {
          return FusionDecision::Forbid(
              "Fusing slice into DUS will also fuse another non-elementwise "
              "op with shared operand as DUS.");
        }
        return FusionDecision::Allow();
      }
      if (producer_nonelementwise->opcode() == HloOpcode::kDynamicSlice) {
        if (producer_nonelementwise->shape() != dus->operand(1)->shape()) {
          return FusionDecision::Forbid(
              "Dynamic slice op has a different shape than the update shape "
              "of the dus op, bailing.");
        }
        for (int i = 0; i < dus->shape().dimensions_size(); ++i) {
          const HloInstruction* ds_operand = get_real_operand(
              producer, producer_nonelementwise->operand(1 + i));
          const HloInstruction* dus_operand =
              get_real_operand(consumer, dus->operand(2 + i));
          auto constant_ds_operand = get_constant_operand(ds_operand);
          auto constant_dus_operand = get_constant_operand(dus_operand);
          if (constant_ds_operand != constant_dus_operand ||
              (!constant_ds_operand && ds_operand != dus_operand)) {
            return FusionDecision::Forbid("DUS and DS index mismatch");
          }
        }
        VLOG(4) << "DUS and DS index match";
        if (consumer->opcode() == HloOpcode::kFusion &&
            !IsSafeToFuseSliceIntoDusFusion(producer, consumer, dus)) {
          return FusionDecision::Forbid(
              "Fusing DS into DUS will also fuse another non-elementwise op "
              "with shared operand as DUS.");
        }
        return FusionDecision::Allow();
      }
      return FusionDecision::Forbid(
          "unrecognized inplace update non-elementwise output pair");
    }
  }
  return FusionDecision::Allow();
}

FusionDecision InstructionFusion::ShouldFuse(HloInstruction* consumer,
                                             int64_t operand_index) {
  return ShouldFuse(consumer, operand_index, ShouldFuseInPlaceOp);
}

FusionDecision InstructionFusion::ShouldFuse(
    HloInstruction* consumer, int64_t operand_index,
    std::function<FusionDecision(const HloInstruction*, const HloInstruction*,
                                 std::optional<const InPlaceFusionOptions>)>
        inplace_op_fusion_decider) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

  // Don't fuse across a root instruction.
  if (producer == producer->parent()->root_instruction()) {
    return FusionDecision::Forbid(
        "not fusing into the output of the root instruction");
  }

  // Cost condition: don't duplicate expensive instructions.
  if (FusionWouldDuplicate(*producer, *consumer) &&
      (!may_duplicate_ || is_expensive_(*producer)) &&
      !IsAlwaysDuplicable(*producer)) {
    return FusionDecision::Forbid(may_duplicate_
                                      ? "expensive producer would be duplicated"
                                      : "fusion pass cannot duplicate");
  }
  return inplace_op_fusion_decider(producer, consumer, std::nullopt);
}

HloInstruction::FusionKind InstructionFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  return HloInstruction::FusionKind::kLoop;
}

absl::flat_hash_set<const HloInstruction*>& InstructionFusion::ReusedOperandsOf(
    const HloInstruction* instruction) {
  std::unique_ptr<absl::flat_hash_set<const HloInstruction*>>& reused_operands =
      reused_fusion_operands_[instruction];
  if (reused_operands != nullptr) {
    return *reused_operands;
  }
  reused_operands =
      std::make_unique<absl::flat_hash_set<const HloInstruction*>>();

  for (int64_t i = 0; i < instruction->operand_count(); ++i) {
    bool reuses = instruction->ReusesOperandElements(i);
    if (reuses) {
      // We cache the operand corresponding to the fusion parameter, because the
      // parameter numbers would be invalidated after the next fusion.
      reused_operands->insert(instruction->operand(i));
    }
  }
  return *reused_operands;
}

bool InstructionFusion::ReusesOperandElements(const HloInstruction* consumer,
                                              int64_t operand_index) {
  auto operand = consumer->operand(operand_index);
  return ReusedOperandsOf(consumer).contains(operand);
}

void InstructionFusion::DumpPreFusionState(HloComputation* computation,
                                           HloInstruction* consumer,
                                           HloInstruction* producer,
                                           bool is_mof) {
  RegisterFusionState(
      *computation,
      absl::StrCat(
          "About to ", is_mof ? "MOF-fuse" : "fuse", " |", producer->name(),
          "| into |", consumer->name(),
          "| inside InstructionFusion with may_duplicate=", may_duplicate_),
      *consumer, producer);
}

void InstructionFusion::DumpNotFusingState(HloComputation* computation,
                                           HloInstruction* consumer,
                                           HloInstruction* producer,
                                           FusionDecision decision) {
  // Readability optimizations: lack of fusion for tuple accesses
  // generates a lot of noise.
  if (producer->opcode() == HloOpcode::kGetTupleElement ||
      consumer->opcode() == HloOpcode::kGetTupleElement) {
    return;
  }

  RegisterFusionState(
      *computation,
      absl::StrCat("Not fusing |", producer->name(), "| into |",
                   consumer->name(), "| as ", decision.Explain()),
      *consumer, producer);
}

void InstructionFusion::DumpStateAfterFusion(HloComputation* computation,
                                             HloInstruction* fusion_instruction,
                                             const std::string& producer_name) {
  RegisterFusionState(
      *computation,
      absl::StrCat(
          "Fused |", producer_name, "| into |", fusion_instruction->name(),
          "| inside InstructionFusion with may_duplicate=", may_duplicate_),
      *fusion_instruction);
}

}  // namespace xla
