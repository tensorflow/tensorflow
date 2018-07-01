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

#include "tensorflow/compiler/plugin/poplar/driver/scheduler.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {

class DepthFinder {
 public:
  DepthFinder(const std::vector<HloInstruction*> insts) {
    for (auto i : insts) {
      distance[i] = 0;
    }
  }

  int64 FindDepths(HloInstruction* inst) {
    int64 cost = 0;
    if (distance.find(inst) != distance.end()) {
      return distance.at(inst);
    }

    for (auto o : inst->operands()) {
      int64 c = FindDepths(o);
      cost = std::max(cost, c);
    }
    cost += 1;

    distance[inst] = cost;
    return cost;
  }

  bool Compare(const HloInstruction* a, const HloInstruction* b) const {
    return distance.at(a) < distance.at(b);
  }

 private:
  std::map<const HloInstruction*, int64> distance;
};

}  // namespace

StatusOr<std::vector<const HloInstruction*>> Scheduler::schedule(
    HloComputation* comp) {
  DepthFinder depths(comp->MakeInstructionPostOrder());
  depths.FindDepths(comp->root_instruction());

  std::vector<const HloInstruction*> sequence;
  FunctionVisitor visitor([&sequence](HloInstruction* hlo) {
    sequence.push_back(hlo);
    return Status::OK();
  });
  TF_RETURN_IF_ERROR(comp->AcceptWithOperandOrder(
      &visitor, [&depths](const HloInstruction* a, const HloInstruction* b) {
        return depths.Compare(a, b);
      }));

  CHECK_EQ(sequence.size(), comp->instruction_count());
  return sequence;
}

}  // namespace poplarplugin
}  // namespace xla
