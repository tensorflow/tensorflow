/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/despecializer.h"

#include "tensorflow/compiler/xla/service/bfloat16_normalization.h"
#include "tensorflow/compiler/xla/service/defuser.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/implicit_broadcast_remover.h"

namespace xla {

namespace {

// Pass which strips control dependencies from all instructions in the module.
class ControlDepRemover : public HloModulePass {
 public:
  ControlDepRemover() = default;
  absl::string_view name() const override { return "control-dep-remover"; }

  StatusOr<bool> Run(HloModule* module) override {
    bool changed = false;
    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        changed = changed || !instruction->control_predecessors().empty();
        TF_RETURN_IF_ERROR(instruction->DropAllControlDeps());
      }
    }
    return changed;
  }
};

}  // namespace

Despecializer::Despecializer() : pipeline_("despecializer") {
  // TODO(b/70588125): Also deal with window reversal in a fast way.
  pipeline_.AddPass<HloDescheduler>();
  pipeline_.AddPass<ControlDepRemover>();
  pipeline_.AddPass<Defuser>();
  pipeline_.AddPass<ImplicitBroadcastRemover>();
  pipeline_.AddPass<BFloat16MixedPrecisionRemoval>();
}

StatusOr<bool> Despecializer::Run(HloModule* module) {
  return pipeline_.Run(module);
}

}  // namespace xla
