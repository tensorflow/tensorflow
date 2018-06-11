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
  return instruction.opcode() == HloOpcode::kConvert &&
         ShapeUtil::ByteSizeOf(instruction.operand(0)->shape()) <
             ShapeUtil::ByteSizeOf(instruction.shape());
}
}  // namespace

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
    case HloOpcode::kClz:
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
    case HloOpcode::kDomain:
    case HloOpcode::kDot:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFft:
    case HloOpcode::kFusion:
    case HloOpcode::kGather:
    case HloOpcode::kHostCompute:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
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

// An "effectively at most unary" operation is one that has at most one "large"
// input with the others being negligible in terms of memory usage.
// We use "has a smaller true rank than the output" as a heuristic
// for "negligible" memory usage.
bool InstructionFusion::EffectivelyAtMostUnary(HloInstruction* hlo) {
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
    HloInstruction* producer, HloInstruction* consumer,
    const HloInstructionSet& do_not_duplicate) {
  if (consumer == producer) {
    return true;
  }
  if (!consumer->IsFusable()) {
    return false;
  }
  for (int64 i = 0, e = consumer->operand_count(); i < e; ++i) {
    auto* consumer_operand = consumer->mutable_operand(i);
    // If the operand is not on a path to the producer, it doesn't matter
    // whether it's fusable.
    if (!reachability_->IsReachable(producer, consumer_operand)) {
      continue;
    }
    if (do_not_duplicate.count(consumer_operand) > 0 ||
        !ShouldFuse(consumer, i)) {
      return false;
    }
    // The producer is reachable from consumer_operand which means we need
    // to be able to fuse consumer_operand into consumer in order for
    // producer to be fusable into consumer on all paths.
    // Perform the recursive step: make sure producer can be fused into
    // consumer_operand on all paths.
    if (!CanFuseOnAllPaths(producer, consumer_operand, do_not_duplicate)) {
      return false;
    }
  }
  return true;
}

InstructionFusion::HloInstructionSet
InstructionFusion::ComputeGloballyUnfusable(
    tensorflow::gtl::ArraySlice<HloInstruction*> post_order) {
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
  for (HloInstruction* consumer : post_order) {
    for (HloInstruction* producer : consumer->operands()) {
      if (do_not_duplicate.count(producer) > 0) {
        continue;
      }

      // If the producer is effectively not more than unary, duplicating it
      // will not increase the number of relevant inputs read, as the fusion
      // node will only need to read at most 1 relevant input (the input of
      // the producer). In that case, we do not forbid fusion of the operation
      // here.
      if (EffectivelyAtMostUnary(producer)) {
        continue;
      }
      // Otherwise we will forbid fusing the op unless we can fuse it into
      // all of its consumers on all paths.
      //
      // That means, that for:
      // A --> B (fusable)
      //   \-> C (non-fusable)
      // A will be not allowed to be fused into B, as it cannot be fused into C.
      //
      // Similarly, for:
      // A -------------> B
      //   \-> C -> D -/
      // If:
      // - A is fusable into B and C, and D is fusable into B
      // - C is *not* fusable into D
      // A will be not allowed to be fused into B, as it cannot be fused via
      // all paths.
      if (producer->IsFusable() &&
          CanFuseOnAllPaths(producer, consumer, do_not_duplicate)) {
        continue;
      }
      do_not_duplicate.insert(producer);
    }
  }

  return do_not_duplicate;
}

StatusOr<bool> InstructionFusion::Run(HloModule* module) {
  VLOG(2) << "Before instruction fusion:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  module_ = module;
  for (auto* computation : module->MakeNonfusionComputations()) {
    CHECK(!computation->IsFusionComputation());
    computation_ = computation;
    reachability_ = computation_->ComputeReachability();

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

    HloInstructionSet do_not_duplicate = ComputeGloballyUnfusable(post_order);

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
      std::vector<int64> sorted_operand_numbers;
      sorted_operand_numbers.reserve(instruction->operands().size());
      for (int i = 0; i < instruction->operands().size(); ++i) {
        // This will happen if we have two possible instructions to fuse the
        // same operand into; once the operand is fused into one instruction,
        // the other instruction will get a new get-tuple-element as its
        // operand, which is not in the post-order index.
        // TODO(tjoerg): Look into fusing past these multi-output fuse points.
        if (post_order_index.find(instruction->mutable_operand(i)) ==
            post_order_index.end()) {
          continue;
        }
        sorted_operand_numbers.push_back(i);
      }
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

        HloInstruction* fusion_instruction;
        // Try "regular" fusion if the operand may be duplicated. Otherwise,
        // perform multi-output fusion, unless this creates a cycle.
        // TODO(tjoerg): Consider making multi-output fusion the default.
        if (ShouldFuse(instruction, i) &&
            do_not_duplicate.count(operand) == 0) {
          fusion_instruction = Fuse(operand, instruction);
        } else if (ShouldFuseIntoMultiOutput(instruction, i) &&
                   !MultiOutputFusionCreatesCycle(operand, instruction)) {
          fusion_instruction = FuseIntoMultiOutput(operand, instruction);
        } else {
          continue;
        }

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

HloInstruction* InstructionFusion::AddFusionInstruction(
    HloInstruction* producer, HloInstruction* consumer) {
  HloInstruction* fusion_instruction;
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
  return fusion_instruction;
}

HloInstruction* InstructionFusion::Fuse(HloInstruction* producer,
                                        HloInstruction* consumer) {
  VLOG(2) << "Fusing " << producer->ToString() << " into "
          << consumer->ToString();
  HloInstruction* fusion_instruction = AddFusionInstruction(producer, consumer);
  fusion_instruction->FuseInstruction(producer);
  return fusion_instruction;
}

HloInstruction* InstructionFusion::FuseIntoMultiOutput(
    HloInstruction* producer, HloInstruction* consumer) {
  VLOG(2) << "Multi-output fusing " << producer->ToString() << " into "
          << consumer->ToString();
  HloInstruction* fusion_instruction = AddFusionInstruction(producer, consumer);
  fusion_instruction->FuseInstructionIntoMultiOutput(producer);
  return fusion_instruction;
}

bool InstructionFusion::MultiOutputFusionCreatesCycle(
    HloInstruction* producer, HloInstruction* consumer) {
  return c_any_of(
      consumer->operands(), [&](const HloInstruction* consumer_operand) {
        // The fusion algorithm traverses the HLO graph in reverse post order.
        // Thus `cosumers` is visited before its operands (including
        // `producer`). Therefore, consumer operands cannot have been fused yet.
        // It is thus safe to use the pre-computed reachability map.
        return consumer_operand != producer &&
               reachability_->IsReachable(producer, consumer_operand);
      });
}

bool InstructionFusion::ShouldFuse(HloInstruction* consumer,
                                   int64 operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);

  // Cost condition: don't duplicate expensive instructions.
  if (FusionWouldDuplicate(*producer, *consumer) &&
      (!may_duplicate_ || is_expensive_(*producer)) &&
      !IsAlwaysDuplicable(*producer)) {
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
