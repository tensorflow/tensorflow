/* Copyright 2018 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/while_loop_to_repeat_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/while_loop_util.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/while_loop_analysis.h"

#include <stdlib.h>
#include <map>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {
namespace {
// Allow user to change the upper bound of the brute force method
int64 GetMaxLoopTripCount() {
  if (const char* env_c = std::getenv("TF_POPLAR_MAX_WHILE_LOOP_TRIP_COUNT")) {
    std::string env(env_c);
    return std::stoll(env);
  }
  return 128;
}
}  // namespace

WhileLoopToRepeatSimplify::WhileLoopToRepeatSimplify(
    CompilerAnnotations& annotations)
    : while_loop_num_iterations(annotations.while_loop_num_iterations) {}

StatusOr<bool> WhileLoopToRepeatSimplify::Run(HloModule* module) {
  std::vector<HloInstruction*> while_insts;
  for (auto* comp : module->computations()) {
    for (auto* inst : comp->instructions()) {
      if (inst->opcode() == HloOpcode::kWhile) {
        while_insts.push_back(inst);
      }
    }
  }

  for (auto* while_inst : while_insts) {
    // For each while loop, try and simplify the logic to convert the loop into
    // a repeat
    auto statusor = WhileLoopUtil::CanConvertWhileToRepeat(while_inst);
    uint64 count = 0;
    bool simplified = false;
    if (statusor.ok()) {
      simplified = true;
      count = statusor.ValueOrDie();
    } else {
      // Try the brute force method
      auto op_count =
          ComputeWhileLoopTripCount(while_inst, GetMaxLoopTripCount());
      if (op_count) {
        simplified = true;
        count = *op_count;
      }
    }

    if (simplified) {
      VLOG(1) << "Simplified while loop " << while_inst->name()
              << " with a repeat of count " << count;
      while_loop_num_iterations[while_inst] = count;
    }
  }
  // This is an analysis pass
  return false;
}

}  // namespace poplarplugin
}  // namespace xla
