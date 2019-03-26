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

#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/find_all_users.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

using UserAndParam = std::pair<HloInstruction*, int>;

StatusOr<bool> InterIpuCopyInserter::Run(HloModule* module) {
  if (!HaveSharding(module)) {
    return false;
  }

  bool added = false;

  for (auto* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // Add InterIpuCopy instructions between nodes which are on different
    // devices
    auto original_insts = comp->MakeInstructionPostOrder();
    for (auto* inst : original_insts) {
      if (!inst->has_sharding()) {
        continue;
      }

      if (inst->opcode() == HloOpcode::kAfterAll) {
        continue;
      }

      const auto& src_sharding = GetShardingOfOutputTensor(inst);

      // Construct a map from the sharding of the tensor users' operand inputs
      // (represented as a vector of int64 values, to all users and operand
      // indices.
      std::multimap<std::vector<int64>, UserAndParam> dst_sharding_map;
      std::set<std::vector<int64>> dst_shardings;
      std::map<std::vector<int64>, HloSharding> sharding_map;
      for (const auto& user : inst->users()) {
        if (user->opcode() == HloOpcode::kAfterAll) {
          continue;
        }

        if (user->opcode() == HloOpcode::kGetTupleElement) {
          // GTEs should always have the same sharding as the tuple
          const auto& s = inst->sharding();
          const auto& tuple_sub_sharding =
              s.IsTuple()
                  ? s.GetSubSharding(inst->shape(), {user->tuple_index()})
                  : s;
          if (tuple_sub_sharding != user->sharding()) {
            return InternalError(
                "Different sharding on Tuple and GTE: %s != %s",
                inst->ToString(), user->ToString());
          }
          continue;
        }

        for (int operand = 0; operand < user->operand_count(); operand++) {
          if (user->operand(operand) == inst) {
            const auto& dst_sharding = GetShardingForOperand(user, operand);

            std::vector<int64> sharding_vector =
                GetShardingDeviceIdVector(dst_sharding);

            if (src_sharding != dst_sharding) {
              auto u = std::make_pair(user, operand);
              dst_sharding_map.insert(std::make_pair(sharding_vector, u));
              dst_shardings.insert(sharding_vector);
              sharding_map.insert(
                  std::make_pair(sharding_vector, dst_sharding));
            }
          }
        }
      }

      // For each unique destination sharding that is not the same as the
      // sharding of the source of the tensors, add an inter-ipu copy to move
      // the tensors to the other devices.
      for (auto s : dst_shardings) {
        added = true;
        auto range = dst_sharding_map.equal_range(s);
        HloInstruction* new_inst;
        if (inst->opcode() == HloOpcode::kConstant ||
            IsPopOpsFusion(inst, "wide_const")) {
          new_inst = comp->AddInstruction(inst->Clone());
        } else {
          new_inst = comp->AddInstruction(HloInstruction::CreateCustomCall(
              inst->shape(), {inst}, "inter_ipu_copy", ""));
        }

        new_inst->set_sharding(sharding_map.at(s));

        for (auto user = range.first; user != range.second; ++user) {
          auto* u = user->second.first;
          auto o = user->second.second;
          u->ReplaceOperandWith(o, new_inst);
        }
      }
    }
  }

  return added;
}

InterIpuCopyInserter::InterIpuCopyInserter() {}

}  // namespace poplarplugin
}  // namespace xla
