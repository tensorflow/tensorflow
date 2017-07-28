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

#include "tensorflow/compiler/xla/service/call_inliner.h"

#include <deque>

#include "tensorflow/core/lib/core/errors.h"

namespace xla {

StatusOr<bool> CallInliner::Run(HloModule* module) {
  std::deque<HloInstruction*> work_queue;

  // Seed the work queue with call instructions from the main computation.
  TF_RETURN_IF_ERROR(
      module->entry_computation()->Accept([&](HloInstruction* hlo) {
        if (hlo->opcode() == HloOpcode::kCall) {
          work_queue.push_back(hlo);
        }
        return Status::OK();
      }));

  VLOG(1) << "Work queue seeded with " << work_queue.size() << " entries.";

  bool mutated = false;
  while (!work_queue.empty()) {
    mutated = true;
    HloInstruction* call = work_queue.front();
    work_queue.pop_front();
    TF_RETURN_IF_ERROR(ReplaceWithInlinedBody(call, &work_queue));
  }
  return mutated;
}

// Traverses the callee computation, inlining cloned nodes into the caller
// computation and connecting them to producers/consumers appropriately.
// When the traversal has completed, the provided call instruction is entriely
// replaced in the caller's graph, and any calls encountered in the callee
// computation have been added to the work_queue.
class SubcomputationInsertionVisitor : public DfsHloVisitorWithDefault {
 public:
  SubcomputationInsertionVisitor(HloInstruction* call,
                                 std::deque<HloInstruction*>* work_queue)
      : call_(call), outer_(call->parent()), work_queue_(work_queue) {}

  // Resolves the operands to the HLO instruction in the inlined (caller) graph,
  // and clones the HLO instruction into that graph with the new operands.
  // If the instruction is a call, it is added to the work queue.
  Status DefaultAction(HloInstruction* hlo) override {
    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* operand : hlo->operands()) {
      TF_ASSIGN_OR_RETURN(HloInstruction * new_operand, Resolve(operand));
      new_operands.push_back(new_operand);
    }
    VLOG(1) << "Cloning HLO and adding to caller: " << hlo->ToString();
    auto new_hlo = hlo->CloneWithNewOperands(hlo->shape(), new_operands);
    HloInstruction* new_hlo_pointer =
        outer_->AddInstruction(std::move(new_hlo));
    TF_RETURN_IF_ERROR(NoteMapping(hlo, new_hlo_pointer));

    // Account for control edges.
    for (HloInstruction* control_predecessor : hlo->control_predecessors()) {
      TF_ASSIGN_OR_RETURN(HloInstruction * new_control_predecessor,
                          Resolve(control_predecessor));
      TF_RETURN_IF_ERROR(
          new_control_predecessor->AddControlDependencyTo(new_hlo_pointer));
    }

    if (new_hlo_pointer->opcode() == HloOpcode::kCall) {
      VLOG(1) << "Adding new call HLO to work queue.";
      // Call instructions we observe in the subcomputation are added to the
      // inliner work queue.
      work_queue_->push_back(new_hlo_pointer);
    }
    return Status::OK();
  }

  // Does not create new nodes for the parameter; rather, notes the mapping from
  // the subcomputation parameter node to the call operands in the caller
  // computation.
  Status HandleParameter(HloInstruction* parameter) override {
    TF_RETURN_IF_ERROR(NoteMapping(
        parameter, call_->mutable_operand(parameter->parameter_number())));
    return Status::OK();
  }

  // Wires the consumers of the call to instead point at the newly created root,
  // replacing the call operation in the caller computation.
  Status FinishVisit(HloInstruction* root) override {
    TF_ASSIGN_OR_RETURN(HloInstruction * new_root, Resolve(root));
    VLOG(1) << "Replacing all uses of " << call_->ToString()
            << " with new root " << new_root->ToString();
    return outer_->ReplaceInstruction(call_, new_root);
  }

 private:
  // Resolves the callee subcomputation_hlo to the new (inline) HLO in the
  // caller computation, or returns a NotFound error if that subcomputation HLO
  // has not been mapped.
  StatusOr<HloInstruction*> Resolve(HloInstruction* subcomputation_hlo) {
    auto it = subcomputation_hlo_to_new_hlo_.find(subcomputation_hlo);
    if (it == subcomputation_hlo_to_new_hlo_.end()) {
      return NotFound(
          "Could not find mapping from subcomputation HLO %s to a cloned HLO.",
          subcomputation_hlo->ToString().c_str());
    }
    return it->second;
  }

  // Notes that the given subcomputation_hlo in the callee has been mapped to
  // the (inline) new_hlo in the caller computation.
  //
  // Returns an error status if the subcomputation_hlo is mapped more than
  // once.
  Status NoteMapping(HloInstruction* subcomputation_hlo,
                     HloInstruction* new_hlo) {
    auto result = subcomputation_hlo_to_new_hlo_.insert(
        std::make_pair(subcomputation_hlo, new_hlo));
    TF_RET_CHECK(result.second)
        << "A mapping for the subcomputation HLO is already present.";
    return Status::OK();
  }

  HloInstruction* call_;
  HloComputation* outer_;
  std::unordered_map<HloInstruction*, HloInstruction*>
      subcomputation_hlo_to_new_hlo_;
  std::deque<HloInstruction*>* work_queue_;
};

Status CallInliner::ReplaceWithInlinedBody(
    HloInstruction* call, std::deque<HloInstruction*>* work_queue) {
  TF_RET_CHECK(call->opcode() == HloOpcode::kCall);
  TF_RET_CHECK(call->called_computations().size() == 1);
  HloComputation* called = call->called_computations()[0];
  VLOG(1) << "Replacing call " << call->ToString() << " with inlined body of "
          << called->name();

  SubcomputationInsertionVisitor visitor(call, work_queue);
  return called->Accept(&visitor);
}

}  // namespace xla
