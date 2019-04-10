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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/sigmoid.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

namespace xla {
namespace poplarplugin {

HloSigmoidInstruction::HloSigmoidInstruction(HloInstruction* operand)
    : HloPoplarInstruction(
          operand->shape(), {operand},
          GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::Sigmoid),
          {}) {}

const HloInstruction* HloSigmoidInstruction::input() const {
  return operand(0);
}

absl::flat_hash_set<int64> HloSigmoidInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloSigmoidInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloSigmoidInstruction::NumberOfInplaceOperands() const { return 1; }

bool HloSigmoidInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction> HloSigmoidInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloSigmoidInstruction>(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateSigmoid(HloInstruction* operand) {
  return absl::make_unique<HloSigmoidInstruction>(operand);
}

HloSigmoidGradInstruction::HloSigmoidGradInstruction(HloInstruction* out,
                                                     HloInstruction* grad)
    : HloPoplarInstruction(out->shape(), {out, grad},
                           GetPoplibsCustomOpTargetString(
                               PoplibsOp::Popnn, PoplibsOp::SigmoidGrad),
                           {}) {}

const HloInstruction* HloSigmoidGradInstruction::out() const {
  return operand(0);
}

const HloInstruction* HloSigmoidGradInstruction::grad() const {
  return operand(1);
}

absl::flat_hash_set<int64> HloSigmoidGradInstruction::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloSigmoidGradInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloSigmoidGradInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloSigmoidGradInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction>
HloSigmoidGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloSigmoidGradInstruction>(new_operands[0],
                                                      new_operands[1]);
}

std::unique_ptr<HloInstruction> CreateSigmoidGrad(HloInstruction* out,
                                                  HloInstruction* grad) {
  return absl::make_unique<HloSigmoidGradInstruction>(out, grad);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloSigmoidInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateSigmoid(call->mutable_operand(0));
}

static HloPoplarInstructionFactory sigmoid_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::Sigmoid),
    HloSigmoidInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloSigmoidGradInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateSigmoidGrad(call->mutable_operand(0), call->mutable_operand(1));
}

static HloPoplarInstructionFactory sigmoid_grad_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::SigmoidGrad),
    HloSigmoidGradInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
