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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/relu.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

namespace xla {
namespace poplarplugin {

HloReluInstruction::HloReluInstruction(HloInstruction* operand)
    : HloPoplarInstruction(
          operand->shape(), {operand},
          GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::Relu),
          {}) {}

const HloInstruction* HloReluInstruction::input() const { return operand(0); }

absl::flat_hash_set<int64> HloReluInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloReluInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloReluInstruction::NumberOfInplaceOperands() const { return 1; }

bool HloReluInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction> HloReluInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloReluInstruction>(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateRelu(HloInstruction* operand) {
  return absl::make_unique<HloReluInstruction>(operand);
}

HloReluGradInstruction::HloReluGradInstruction(HloInstruction* out,
                                               HloInstruction* grad)
    : HloPoplarInstruction(
          out->shape(), {out, grad},
          GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::ReluGrad),
          {}) {}

const HloInstruction* HloReluGradInstruction::out() const { return operand(0); }

const HloInstruction* HloReluGradInstruction::grad() const {
  return operand(1);
}

absl::flat_hash_set<int64> HloReluGradInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloReluGradInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloReluGradInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloReluGradInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction>
HloReluGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloReluGradInstruction>(new_operands[0],
                                                   new_operands[1]);
}

std::unique_ptr<HloInstruction> CreateReluGrad(HloInstruction* out,
                                               HloInstruction* grad) {
  return absl::make_unique<HloReluGradInstruction>(out, grad);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloReluInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateRelu(call->mutable_operand(0));
}

static HloPoplarInstructionFactory relu_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::Relu),
    HloReluInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloReluGradInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return CreateReluGrad(call->mutable_operand(0), call->mutable_operand(1));
}

static HloPoplarInstructionFactory relu_grad_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::ReluGrad),
    HloReluGradInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
