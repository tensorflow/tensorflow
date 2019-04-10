/* Copyright 2019 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/remap.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

namespace xla {
namespace poplarplugin {

HloRemapInstruction::HloRemapInstruction(HloInstruction* operand)
    : HloPoplarInstruction(
          operand->shape(), {operand},
          GetPoplibsCustomOpTargetString(PoplibsOp::Poputil, PoplibsOp::Remap),
          {}) {}

const HloInstruction* HloRemapInstruction::input() const { return operand(0); }

absl::flat_hash_set<int64> HloRemapInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloRemapInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloRemapInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloRemapInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction> HloRemapInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloRemapInstruction>(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateRemap(HloInstruction* operand) {
  return absl::make_unique<HloRemapInstruction>(operand);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloRemapInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateRemap(call->mutable_operand(0));
}

static HloPoplarInstructionFactory remap_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Poputil, PoplibsOp::Remap),
    HloRemapInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
