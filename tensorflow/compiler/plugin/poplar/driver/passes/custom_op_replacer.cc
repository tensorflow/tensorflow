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

#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

#include <set>

namespace xla {
namespace poplarplugin {

StatusOr<bool> CustomOpReplacer::Run(HloModule* module) {
  std::vector<HloCustomCallInstruction*> custom_calls;

  for (auto comp : module->MakeNonfusionComputations()) {
    for (auto inst : comp->instructions()) {
      if (inst->opcode() == HloOpcode::kCustomCall) {
        auto custom_call = Cast<HloCustomCallInstruction>(inst);

        if (HloPoplarInstructionFactory::IsCreatable(custom_call)) {
          custom_calls.push_back(custom_call);
        }
      }
    }
  }

  for (auto custom_call : custom_calls) {
    TF_ASSIGN_OR_RETURN(auto poplar_inst,
                        HloPoplarInstructionFactory::Create(custom_call));

    if (custom_call->has_sharding()) {
      poplar_inst->set_sharding(custom_call->sharding());
    }

    auto comp = custom_call->parent();
    TF_RETURN_IF_ERROR(
        comp->ReplaceWithNewInstruction(custom_call, std::move(poplar_inst)));
  }

  return !custom_calls.empty();
}

}  // namespace poplarplugin
}  // namespace xla
