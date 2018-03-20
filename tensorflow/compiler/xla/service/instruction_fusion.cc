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

#include "tensorflow/compiler/xla/service/instruction_fusion.h"

#include <algorithm>
#include <list>
#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
/*static*/ bool InstructionFusion::IsExpensive(
    const HloInstruction& instruction) {
  switch (instruction.opcode()) {
    // Cheap instructions.
    case HloOpcode::kAdd:
    case HloOpcode::kAnd:
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
    case HloOpcode::kComplex:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConstant:
    case HloOpcode::kConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kEq:
    case HloOpcode::kFloor:
    case HloOpcode::kGe:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kGt:
    case HloOpcode::kImag:
    case HloOpcode::kInfeed:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNe:
    case HloOpcode::kNegate:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kOutfeed:
    case HloOpcode::kPad:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSlice:
    case HloOpcode::kSubtract:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
      return false;

    // Cheap instructions for reals, but expensive for complex.
    case HloOpcode::kAbs:
    case HloOpcode::kCos:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
      return ShapeUtil::ElementIsComplex(instruction.shape());

    // Expensive instructions.
    case HloOpcode::kAtan2:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kConvolution:
    case HloOpcode::kCrossReplicaSum:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDivide:
    case HloOpcode::kDot:
    case HloOpcode::kExp:
    case HloOpcode::kFft:
    case HloOpcode::kFusion:
    case HloOpcode::kGather:
    case HloOpcode::kHostCompute:
    case HloOpcode::kLog:
    case HloOpcode::kMap:
    case HloOpcode::kParameter:
    case HloOpcode::kPower:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kRemainder:
    case HloOpcode::kRng:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kSort:
    case HloOpcode::kTanh:
    case HloOpcode::kTrace:
    case HloOpcode::kWhile:
      return true;
  }

  return false;
}

// An "effectively unary" operation is one that has one "large"
// input with the others being negligible in terms of memory usage.
// We use "has a smaller true rank than the output" as a heuristic
// for "negligible" memory usage.
bool InstructionFusion::EffectivelyUnary(HloInstruction* hlo) {
  int64 output_rank = 0;
  ShapeUtil::ForEachSubshape(
      hlo->shape(),
      [&output_rank](const Shape& subshape, const ShapeIndex& shape_index) {
        if (ShapeUtil::IsArray(subshape)) {
          output_rank = std::max(output_rank, ShapeUtil::TrueRank(subshape));
        }
      });
  return std::count_if(hlo->operands().begin(), hlo->operands().end(),
                       [output_rank](HloInstruction* operand) {
                         if (operand->opcode() == HloOpcode::kBroadcast) {
                           return false;
                         }
                         if (operand->opcode() == HloOpcode::kConstant &&
                             ShapeUtil::IsEffectiveScalar(operand->shape())) {
                           return false;
                         }
                         return ShapeUtil::TrueRank(operand->shape()) >=
                                output_rank;
                       }) <= 1;
}

bool InstructionFusion::CanFuseOnAllPaths(
    const HloReachabilityMap& reachability_map, HloInstruction* producer,
    HloInstruction* consumer, DoNotFuseSet* do_not_fuse) {
  auto could_fuse_on_all_paths = [&] {
    // First check to see if we have already marked this producer as infeasible
    // to fuse into consumer.
    if (do_not_fuse->count(producer) > 0) {
      return false;
    }
    // Make sure it is possible for producer and consumer to exist in a fusion
    // node.
    if (!producer->IsFusable() || !consumer->IsFusable()) {
      return false;
    }
    // We do an upward walk of the graph from consumer towards all paths which
    // lead to producer to find any unfusable paths.
    for (int64 i = 0, e = consumer->operand_count(); i < e; ++i) {
      auto* consumer_operand = consumer->mutable_operand(i);
      if (consumer_operand == producer) {
        // This is the base case: our upward crawl ends but we need to make sure
        // that fusion from consumer can happen.
        if (!ShouldFuse(consumer, i)) {
          return false;
        }
      } else if (reachability_map.IsReachable(producer, consumer_operand)) {
        // The reachability map told us that consumer_operand is a node on the
        // path to producer. We need to further investigate from
        // consumer_operand.

        // First check if we have already ruled out fusing producer into
        // consumer_operand.
        if (do_not_fuse->count(consumer_operand) > 0) {
          return false;
        }
        // Make sure it is possible for consumer_operand to exist in a fusion
        // node.
        if (!consumer_operand->IsFusable()) {
          return false;
        }
        // The producer is reachable from consumer_operand which means we need
        // to be able to fuse consumer_operand into consumer in order for
        // producer to be fusable into consumer on all paths.
        if (!ShouldFuse(consumer, i)) {
          return false;
        }
        // Perform the recursive step: make sure producer can be fused into
        // consumer_operand on all paths.
        if (!CanFuseOnAllPaths(reachability_map, producer, consumer_operand,
                               do_not_fuse)) {
          return false;
        }
      }
    }
    return true;
  };
  if (could_fuse_on_all_paths()) {
    return true;
  }
  // We couldn't fuse on all paths, record this result.
  do_not_fuse->insert(producer);
  return false;
}

StatusOr<bool> InstructionFusion::Run(HloModule* module) {
  VLOG(2) << "Before instruction fusion:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  module_ = module;
  for (auto* computation : module->MakeNonfusionComputations()) {
    CHECK(!computation->IsFusionComputation());
    computation_ = computation;

    // We want to be able to remove arbitrary instructions from the post order
    // and also compare positions of instructions in the post order. To make
    // this possible, create vector of instructions in post order and create a
    // map from HloInstruction* to the instruction's index in the vector. An
    // instruction is "removed" from the vector by setting it's element to
    // nullptr.
    std::list<HloInstruction*> post_order_list =
        computation_->MakeInstructionPostOrder();
    std::vector<HloInstruction*> post_order(post_order_list.begin(),
                                            post_order_list.end());

    tensorflow::gtl::FlatMap<HloInstruction*, int> post_order_index;
    for (size_t i = 0; i < post_order.size(); ++i) {
      InsertOrDie(&post_order_index, post_order[i], i);
    }

    DoNotFuseSet do_not_fuse;
    auto reachability = computation->ComputeReachability();

    auto cheap_to_duplicate = [this](HloInstruction* producer) {
      if (producer->opcode() == HloOpcode::kBroadcast) {
        return true;
      }
      if (producer->opcode() == HloOpcode::kConstant &&
          ShapeUtil::IsEffectiveScalar(producer->shape())) {
        return true;
      }
      if (EffectivelyUnary(producer)) {
        return true;
      }
      return false;
    };

    for (HloInstruction* consumer : post_order) {
      for (HloInstruction* producer : consumer->operands()) {
        if (cheap_to_duplicate(producer)) {
          continue;
        }
        if (CanFuseOnAllPaths(*reachability, producer, consumer,
                              &do_not_fuse)) {
          CHECK_EQ(do_not_fuse.count(producer), 0);
        } else {
          CHECK_GT(do_not_fuse.count(producer), 0);
        }
      }
    }

    // Instruction fusion effectively fuses edges in the computation graph
    // (producer instruction -> consumer instruction) so we iterate over all
    // edges. When we fuse an edge, we create a copy of the producer inside the
    // fusion instruction.
    while (!post_order.empty()) {
      // We want to iterate in reverse post order, so remove from the back of
      // the vector.
      HloInstruction* instruction = post_order.back();
      post_order.pop_back();

      // Instructions are "removed" from the post order by nulling out the
      // element in the vector, so if the pointer is null, continue to the next
      // instruction in the sort.
      if (instruction == nullptr) {
        continue;
      }

      // Remove instruction from the index map to ensure the vector and map stay
      // consistent.
      post_order_index.erase(instruction);

      if (!instruction->IsFusable() &&
          instruction->opcode() != HloOpcode::kFusion) {
        continue;
      }

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
      // We prevent this duplication by considering the operands in the reverse
      // order they appear in the instruction post order. In the example, this
      // ensures that B will be considered before A.
      //
      // We store the original indices of the operands to pass to ShouldFuse.
      std::vector<int64> sorted_operand_numbers(instruction->operands().size());
      std::iota(std::begin(sorted_operand_numbers),
                std::end(sorted_operand_numbers), 0);
      std::sort(
          sorted_operand_numbers.begin(), sorted_operand_numbers.end(),
          [&](int64 i, int64 j) {
            // Instructions with higher indices in the post order come
            // first.
            return (
                FindOrDie(post_order_index, instruction->mutable_operand(i)) >
                FindOrDie(post_order_index, instruction->mutable_operand(j)));
          });

      for (int64 i : sorted_operand_numbers) {
        HloInstruction* operand = instruction->mutable_operand(i);

        if (!operand->IsFusable()) {
          continue;
        }
        if (!ShouldFuse(instruction, i)) {
          continue;
        }
        if (do_not_fuse.count(operand) > 0) {
          continue;
        }
        HloInstruction* fusion_instruction = Fuse(operand, instruction);

        // Fusing an instruction into a fusion instruction can change the
        // operand set of the fusion instruction. For simplicity just push the
        // instruction to the top of the post_order and reconsider it for
        // further fusion in the next iteration of the outer loop.
        post_order.push_back(fusion_instruction);
        InsertOrDie(&post_order_index, fusion_instruction,
                    post_order.size() - 1);
        changed = true;

        if (operand->user_count() == 0) {
          // Operand is now dead. Remove from post order by setting its
          // location to nullptr.
          post_order[FindOrDie(post_order_index, operand)] = nullptr;
          post_order_index.erase(operand);

          // Remove from computation.
          TF_RETURN_IF_ERROR(computation_->RemoveInstruction(operand));
        }
        break;
      }
    }
  }

  VLOG(2) << "After instruction fusion:";
  XLA_VLOG_LINES(2, module->ToString());

  return changed;
}

HloInstruction* InstructionFusion::Fuse(HloInstruction* producer,
                                        HloInstruction* consumer) {
  HloInstruction* fusion_instruction;

  VLOG(2) << "Fusing " << producer->ToString() << " into "
          << consumer->ToString();
  auto kind = ChooseKind(producer, consumer);
  if (consumer->opcode() == HloOpcode::kFusion) {
    fusion_instruction = consumer;
    if (kind != fusion_instruction->fusion_kind()) {
      fusion_instruction->set_fusion_kind(kind);
    }
  } else {
    fusion_instruction = computation_->AddInstruction(
        HloInstruction::CreateFusion(consumer->shape(), kind, consumer));
    TF_CHECK_OK(computation_->ReplaceInstruction(consumer, fusion_instruction));
  }

  fusion_instruction->FuseInstruction(producer);
  return fusion_instruction;
}

bool InstructionFusion::ShouldFuse(HloInstruction* consumer,
                                   int64 operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);
  // Cost condition: don't duplicate expensive instructions.
  if (FusionWouldDuplicate(*producer, *consumer) &&
      (is_expensive_(*producer) || !may_duplicate_)) {
    return false;
  }

  if (consumer->opcode() == HloOpcode::kFusion &&
      consumer->fusion_kind() != HloInstruction::FusionKind::kLoop &&
      consumer->fusion_kind() != HloInstruction::FusionKind::kInput &&
      consumer->fusion_kind() != HloInstruction::FusionKind::kOutput) {
    return false;
  }

  if (producer->CouldBeBitcast() &&
      // We can't fuse parameters anyhow, so we leave the user unfused to become
      // a bitcast. If the operand is not a parameter, we would break a
      // potential fusion to make it a bitcast, which is not so clear a win.
      producer->operand(0)->opcode() == HloOpcode::kParameter) {
    return false;
  }

  return true;
}

HloInstruction::FusionKind InstructionFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  return HloInstruction::FusionKind::kLoop;
}

}  // namespace xla
