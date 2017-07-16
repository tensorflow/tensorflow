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

#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

bool FuseOps::ShouldFuse(HloInstruction* consumer, int64 operand_index) {
  HloInstruction* producer = consumer->mutable_operand(operand_index);
  if (producer->IsConstant() &&
      consumer->opcode() == HloOpcode::kDynamicUpdateSlice &&
      operand_index == 2) {
    return true;
  }

  if (producer->IsConstant() &&
      consumer->opcode() == HloOpcode::kDynamicSlice &&
      operand_index == 1) {
    return true;
  }

  if (producer->opcode() == HloOpcode::kRng &&
      consumer->opcode() == HloOpcode::kWhile &&
      consumer->while_condition()->name().substr(0, 16) == "truncated_normal") {
    return true;
  }

  if (producer->opcode() == HloOpcode::kConstant &&
      consumer->opcode() == HloOpcode::kMaximum &&
      producer->literal().IsAll(0)) {
    return true;
  }
  return false;
}

HloInstruction::FusionKind
FuseOps::ChooseKind(const HloInstruction* producer,
                    const HloInstruction* consumer) {
  return HloInstruction::FusionKind::kCustom;
}

}
}
