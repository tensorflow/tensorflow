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

#include "tensorflow/compiler/plugin/poplar/driver/sharding_pass.h"
#include "tensorflow/compiler/plugin/poplar/driver/find_all_users.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

static bool HaveSharding(HloComputation* comp) {
  for (auto* inst : comp->instructions()) {
    if (inst->has_sharding()) {
      return true;
    }
  }
  return false;
}

static bool HaveSharding(HloModule* module) {
  for (auto* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // If there is no sharding information, no need to continue
    if (HaveSharding(comp)) {
      return true;
    }
  }
  return false;
}

StatusOr<bool> ShardingPass::Run(HloModule* module) {
  if (!HaveSharding(module)) {
    return true;
  }

  for (auto* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    if (!HaveSharding(comp)) {
      continue;
    }

    // First apply sharding to any simple ops which do not have it
    bool done = false;
    while (!done) {
      done = true;
      for (auto* inst : comp->MakeInstructionPostOrder()) {
        if (!inst->has_sharding()) {
          for (auto* u : inst->users()) {
            if (u->has_sharding()) {
              inst->set_sharding(u->sharding());
              break;
            }
          }
        }
        if (!inst->has_sharding()) {
          for (auto* u : inst->operands()) {
            if (u->has_sharding()) {
              inst->set_sharding(u->sharding());
              break;
            }
          }
        }
        if (!inst->has_sharding()) {
          done = false;
        }
      }
    }
  }

  bool added = false;

  for (auto* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // Now add InterIpuCopy instructions between nodes which are on different
    // devices
    auto original_insts = comp->MakeInstructionPostOrder();
    for (auto* inst : original_insts) {
      auto opcode = inst->opcode();
      if (opcode == HloOpcode::kTuple ||
          opcode == HloOpcode::kGetTupleElement) {
        continue;
      }

      if (!inst->has_sharding()) {
        continue;
      }

      const auto& src_sharding = inst->sharding();
      if (!src_sharding.HasUniqueDevice()) {
        return xla::FailedPrecondition("No unique IPU number on %s",
                                       inst->name());
      }

      int src_ipu = src_sharding.GetUniqueDevice();

      FindAllUsers finder;
      finder.Find(inst);
      auto paths = finder.Paths();

      std::multimap<int, HloInstruction*> ipu_map;
      std::set<int> ipu_nums;
      for (const auto& path : paths) {
        auto user = path.back();
        auto next = path.front();

        if (inst->parent() != next->parent()) {
          return xla::FailedPrecondition(
              "Instructions on different computations %s, %s", inst->name(),
              next->name());
        }

        const auto& dst_sharding = user->sharding();
        if (!dst_sharding.HasUniqueDevice()) {
          return xla::FailedPrecondition("No unique IPU number on %s",
                                         user->name());
        }

        int dst_ipu = dst_sharding.GetUniqueDevice();

        if (src_ipu != dst_ipu) {
          ipu_map.insert(std::make_pair(dst_ipu, next));
          ipu_nums.insert(dst_ipu);
        }
      }

      for (auto ipu : ipu_nums) {
        added = true;
        auto range = ipu_map.equal_range(ipu);
        auto copy_inst = comp->AddInstruction(HloInstruction::CreateCustomCall(
            inst->shape(), {inst}, "inter_ipu_copy", ""));
        copy_inst->set_sharding(HloSharding::AssignDevice(ipu));

        for (auto user = range.first; user != range.second; ++user) {
          auto* u = user->second;
          for (int operand = 0; operand < u->operand_count(); operand++) {
            if (u->operand(operand) == inst) {
              u->ReplaceOperandWith(operand, copy_inst);
            }
          }
        }
      }
    }
  }

  return added;
}

ShardingPass::ShardingPass() {}

}  // namespace poplarplugin
}  // namespace xla
