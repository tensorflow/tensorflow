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

#include "tensorflow/compiler/plugin/poplar/driver/passes/replication_factor_to_constant.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_factor.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {

ReplicationFactorToConstant::ReplicationFactorToConstant(
    int32 replication_factor)
    : replication_factor_(replication_factor) {}

StatusOr<bool> ReplicationFactorToConstant::Run(HloModule* module) {
  bool changed = false;
  for (auto* comp : module->computations()) {
    const auto instructions = comp->MakeInstructionPostOrder();

    for (auto* inst : instructions) {
      if (inst->opcode() == HloOpcode::kCustomCall &&
          IsPoplibsHloCustomOp(inst) &&
          inst->custom_call_target() ==
              GetPoplibsCustomOpTargetString(PoplibsOp::Poputil,
                                             PoplibsOp::ReplicationFactor)) {
        auto replacement = comp->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32>(replication_factor_)));

        inst->ReplaceAllUsesWith(replacement);
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
