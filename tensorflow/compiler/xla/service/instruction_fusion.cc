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

bool IsExpensive(const HloInstruction& instruction) {
  switch (instruction.opcode()) {
    // Cheap instructions.
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCeil:
    case HloOpcode::kClamp:
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
    case HloOpcode::kInfeed:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLe:
    case HloOpcode::kLogicalAnd:
    case HloOpcode::kLogicalNot:
    case HloOpcode::kLogicalOr:
    case HloOpcode::kLt:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNe:
    case HloOpcode::kNegate:
    case HloOpcode::kOutfeed:
    case HloOpcode::kPad:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSelect:
    case HloOpcode::kSign:
    case HloOpcode::kSlice:
    case HloOpcode::kSubtract:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
      return false;

    // Expensive instructions.
    case HloOpcode::kCall:
    case HloOpcode::kConvolution:
    case HloOpcode::kCrossReplicaSum:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDivide:
    case HloOpcode::kDot:
    case HloOpcode::kExp:
    case HloOpcode::kFusion:
    case HloOpcode::kIndex:
    case HloOpcode::kLog:
    case HloOpcode::kMap:
    case HloOpcode::kParameter:
    case HloOpcode::kPower:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kRemainder:
    case HloOpcode::kRng:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSort:
    case HloOpcode::kTanh:
    case HloOpcode::kTrace:
    case HloOpcode::kUpdate:
    case HloOpcode::kWhile:
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
      return true;
  }

  return false;
}

bool FusionWouldDuplicate(HloInstruction* producer, HloInstruction* consumer) {
  return !(producer->users().size() == 1 && consumer->IsUserOf(producer));
}

StatusOr<bool> InstructionFusion::Run(HloModule* module) {
  bool changed = false;
  for (auto& computation : module->computations()) {
    computation_ = computation.get();

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
    for (std::vector<HloInstruction*>::size_type i = 0; i < post_order.size();
         ++i) {
      InsertOrDie(&post_order_index, post_order[i], i);
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
      // avoid created duplicate instruction clones in the fusion instruction.
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
        if (operand->IsFusable() && ShouldFuse(instruction, i)) {
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
            // Operand is now dead. Remove from post order by setting it's
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
  }
  return changed;
}

HloInstruction* InstructionFusion::Fuse(HloInstruction* producer,
                                        HloInstruction* consumer) {
  HloInstruction* fusion_instruction;

  VLOG(2) << "Fusing " << producer << " into " << consumer;

  if (consumer->opcode() == HloOpcode::kFusion) {
    fusion_instruction = consumer;
  } else {
    fusion_instruction =
        computation_->AddInstruction(HloInstruction::CreateFusion(
            consumer->shape(), ChooseKind(producer, consumer), consumer));
    TF_CHECK_OK(computation_->ReplaceInstruction(consumer, fusion_instruction));
  }
  fusion_instruction->FuseInstruction(producer);

  return fusion_instruction;
}

bool InstructionFusion::ShouldFuse(HloInstruction* consumer,
                                   int64 operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);
  // Cost condition: don't duplicate expensive instructions.
  if (FusionWouldDuplicate(producer, consumer) &&
      (IsExpensive(*producer) || !may_duplicate_)) {
    return false;
  }

  if (consumer->opcode() == HloOpcode::kFusion &&
      consumer->fusion_kind() != HloInstruction::FusionKind::kLoop &&
      consumer->fusion_kind() != HloInstruction::FusionKind::kInput) {
    return false;
  }

  // Cost condition: not fuse (expensive producers) and (consumers who reuse
  // operand elements).
  if (consumer->ReusesOperandElements(operand_index) &&
      IsExpensive(*producer)) {
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
