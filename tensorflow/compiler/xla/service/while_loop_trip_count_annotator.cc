/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/while_loop_trip_count_annotator.h"
#include "tensorflow/compiler/xla/service/while_loop_analysis.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

StatusOr<bool> WhileLoopTripCountAnnotator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() != HloOpcode::kWhile) {
        continue;
      }
      if (auto trip_count = ComputeWhileLoopTripCount(instr)) {
        WhileLoopBackendConfig config;
        config.mutable_known_trip_count()->set_n(*trip_count);
        TF_RETURN_IF_ERROR(instr->set_backend_config(config));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
