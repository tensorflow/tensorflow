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

#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/find_all_users.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> ShardingPass::Run(HloModule* module) {
  if (!HaveSharding(module)) {
    return false;
  }

  for (auto* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // Remove unsupported sharding
    for (auto* inst : comp->instructions()) {
      if (inst->has_sharding()) {
        auto sharding = inst->sharding();
        if (!IsSupportedSharding(sharding)) {
          LOG(INFO) << "Instruction " << inst->name()
                    << " has unsupported sharding " << sharding.ToString()
                    << " which will be ignored.";
          inst->clear_sharding();
        }
      }
    }

    if (!HaveSharding(comp)) {
      continue;
    }

    // First apply sharding to any simple ops which do not have it
    bool done = false;
    while (!done) {
      done = true;
      bool made_progress = false;
      for (auto* inst : comp->MakeInstructionPostOrder()) {
        if (!inst->has_sharding()) {
          for (auto* u : inst->users()) {
            if (u->has_sharding()) {
              inst->set_sharding(u->sharding());
              made_progress = true;
              break;
            }
          }
        }
        if (!inst->has_sharding()) {
          for (auto* u : inst->operands()) {
            if (u->has_sharding()) {
              inst->set_sharding(u->sharding());
              made_progress = true;
              break;
            }
          }
        }
        if (!inst->has_sharding()) {
          done = false;
        }
      }
      if (!done && !made_progress) {
        return xla::FailedPrecondition(
            "Could not apply sharding information to the %s computation.",
            comp->name().c_str());
      }
    }
  }

  return true;
}

ShardingPass::ShardingPass() {}

}  // namespace poplarplugin
}  // namespace xla
