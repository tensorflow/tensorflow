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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/onehot.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

namespace xla {
namespace poplarplugin {

// Constructor.
HloOneHotInstruction::HloOneHotInstruction(HloInstruction* indices, int64 depth,
                                           int32 axis, HloInstruction* on,
                                           HloInstruction* off,
                                           const Shape shape)
    : HloPoplarInstruction(
          shape, {indices, on, off},
          GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::OneHot),
          {}),
      depth_(depth),
      axis_(axis) {}

absl::flat_hash_set<int64> HloOneHotInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloOneHotInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloOneHotInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloOneHotInstruction::IsPopOpsElementwise() const { return false; }

// Creates an instance of a HloOneHotInstruction
std::unique_ptr<HloInstruction> CreateOneHot(HloInstruction* indices,
                                             int64 depth, int32 axis,
                                             HloInstruction* on,
                                             HloInstruction* off,
                                             const Shape& shape) {
  return absl::make_unique<HloOneHotInstruction>(indices, depth, axis, on, off,
                                                 shape);
}

std::unique_ptr<HloInstruction> HloOneHotInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  return CreateOneHot(operands[0], Depth(), Axis(), operands[1], operands[2],
                      shape);
}

namespace {

static HloPoplarInstructionFactory onehot_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::OneHot),
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

      TF_ASSIGN_OR_RETURN(int64 depth,
                          attribute_map.GetAttributeAsInt("depth"));

      TF_ASSIGN_OR_RETURN(int64 axis, attribute_map.GetAttributeAsInt("axis"));

      return CreateOneHot(call->mutable_operand(0), depth, axis,
                          call->mutable_operand(1), call->mutable_operand(2),
                          call->shape());
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
