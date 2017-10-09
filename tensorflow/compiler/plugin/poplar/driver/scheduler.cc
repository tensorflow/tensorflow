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

class FindAllInstructions : public DfsHloVisitorWithDefault {
public:
  FindAllInstructions() {}

  ~FindAllInstructions() override = default;

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    inst.push_back(hlo_instruction);
    return Status::OK();
  }

  std::list<HloInstruction*> inst;
};

}

std::vector<const HloInstruction*> Scheduler::schedule(HloComputation* comp) {
  FindAllInstructions all;
  comp->Accept(&all);

  std::vector<const HloInstruction*> out(all.inst.begin(), all.inst.end());
  return out;
}

}
}
