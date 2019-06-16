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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/print_tensor.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

// Constructor.
HloPrintTensor::HloPrintTensor(HloInstruction* input)
    : HloPoplarInstruction(ShapeUtil::MakeTokenShape(), {input},
                           GetPoplibsCustomOpTargetString(
                               PoplibsOp::Poputil, PoplibsOp::PrintTensor)) {}

absl::flat_hash_set<int64> HloPrintTensor::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloPrintTensor::LayoutDependencies() const {
  return {};
}

uint64 HloPrintTensor::NumberOfInplaceOperands() const { return 0; }

bool HloPrintTensor::IsPopOpsElementwise() const { return false; }

bool HloPrintTensor::HasSideEffectNoRecurse() const { return true; }

// Creates an instance of a HloOneHotInstruction
std::unique_ptr<HloInstruction> CreateHloPrintTensor(HloInstruction* input) {
  return absl::make_unique<HloPrintTensor>(input);
}

std::unique_ptr<HloInstruction> HloPrintTensor::CloneWithNewOperandsImpl(
    const Shape&, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateHloPrintTensor(operands[0]);
}

namespace {

static HloPoplarInstructionFactory print_tensor_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Poputil, PoplibsOp::PrintTensor),
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      return CreateHloPrintTensor(call->mutable_operand(0));
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
