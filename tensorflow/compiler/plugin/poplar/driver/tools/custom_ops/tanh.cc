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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/tanh.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

namespace xla {
namespace poplarplugin {

HloTanhGradInstruction::HloTanhGradInstruction(HloInstruction* out,
                                               HloInstruction* grad)
    : HloPoplarInstruction(
          out->shape(), {out, grad},
          GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::TanhGrad),
          {}) {}

const HloInstruction* HloTanhGradInstruction::out() const { return operand(0); }

const HloInstruction* HloTanhGradInstruction::grad() const {
  return operand(1);
}

absl::flat_hash_set<int64> HloTanhGradInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloTanhGradInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloTanhGradInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloTanhGradInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction>
HloTanhGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloTanhGradInstruction>(new_operands[0],
                                                   new_operands[1]);
}

std::unique_ptr<HloInstruction> CreateTanhGrad(HloInstruction* out,
                                               HloInstruction* grad) {
  return absl::make_unique<HloTanhGradInstruction>(out, grad);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloTanhGradInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateTanhGrad(call->mutable_operand(0), call->mutable_operand(1));
}

static HloPoplarInstructionFactory relu_grad_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::TanhGrad),
    HloTanhGradInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
