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

#include "tensorflow/compiler/plugin/poplar/driver/passes/scheduler.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {

class SequenceCosts {
 public:
  SequenceCosts() : max_root_cost(0) {}

  int64 FindCosts(HloInstruction* inst) {
    int64 cost = 0;
    if (costs.find(inst) != costs.end()) {
      return costs.at(inst);
    }

    for (auto o : inst->operands()) {
      int64 c = FindCosts(o);
      cost = std::max(cost, c);
    }
    cost += 1;

    costs[inst] = cost;
    max_root_cost = std::max(max_root_cost, cost);

    return cost;
  }

  void AddDisconnectedInstructions(const std::vector<HloInstruction*> insts) {
    int64 cost = max_root_cost + 1;
    for (auto* inst : insts) {
      if (costs.find(inst) == costs.end()) {
        costs[inst] = cost++;
      }
    }
  }

  bool Compare(const HloInstruction* a, const HloInstruction* b) const {
    if (costs.at(a) > max_root_cost || costs.at(b) > max_root_cost) {
      return costs.at(a) < costs.at(b);
    }

    unsigned int types = ((a->opcode() == HloOpcode::kParameter) ? 0x2 : 0x0) +
                         ((b->opcode() == HloOpcode::kParameter) ? 0x1 : 0x0);

    switch (types) {
      case 0x0:
        return costs.at(a) < costs.at(b);
      case 0x1:
        return true;
      case 0x2:
        return false;
      case 0x3:
        return costs.at(a) > costs.at(b);
    }

    return false;
  }

 private:
  std::map<const HloInstruction*, int64> costs;
  int64 max_root_cost;
};

}  // namespace

StatusOr<bool> Scheduler::Run(HloModule* module) {
  HloSchedule schedule(module);

  for (auto* comp : module->MakeNonfusionComputations()) {
    SequenceCosts costs;
    costs.FindCosts(comp->root_instruction());
    costs.AddDisconnectedInstructions(comp->MakeInstructionPostOrder());

    HloInstructionSequence sequence;
    FunctionVisitor visitor([&sequence](HloInstruction* hlo) {
      sequence.push_back(hlo);
      return Status::OK();
    });
    TF_RETURN_IF_ERROR(comp->AcceptWithOperandOrder(
        &visitor, [&costs](const HloInstruction* a, const HloInstruction* b) {
          return costs.Compare(a, b);
        }));

    CHECK_EQ(sequence.size(), comp->instruction_count());
    schedule.set_sequence(comp, sequence);
  }

  module->set_schedule(schedule);
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
