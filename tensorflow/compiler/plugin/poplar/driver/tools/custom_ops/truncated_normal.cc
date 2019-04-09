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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/truncated_normal.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

namespace xla {
namespace poplarplugin {

HloTruncatedNormalInstruction::HloTruncatedNormalInstruction(const Shape& shape)
    : HloPoplarInstruction(shape, {},
                           GetPoplibsCustomOpTargetString(
                               PoplibsOp::Poprand, PoplibsOp::TruncatedNormal),
                           {}) {}

absl::flat_hash_set<int64> HloTruncatedNormalInstruction::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloTruncatedNormalInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloTruncatedNormalInstruction::NumberOfInplaceOperands() const {
  return 0;
}

bool HloTruncatedNormalInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloTruncatedNormalInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const>,
    HloCloneContext*) const {
  return absl::make_unique<HloTruncatedNormalInstruction>(shape);
}

std::unique_ptr<HloInstruction> CreateTruncatedNormal(const Shape& shape) {
  return absl::make_unique<HloTruncatedNormalInstruction>(shape);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>>
HloTruncatedNormalInstructionFactoryFunc(HloCustomCallInstruction* call) {
  // Decode opaque here...
  return CreateTruncatedNormal(call->shape());
}

static HloPoplarInstructionFactory truncated_normal_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Poprand,
                                   PoplibsOp::TruncatedNormal),
    HloTruncatedNormalInstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
