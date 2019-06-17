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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/arg_min_max.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace poplarplugin {

static PoplibsOp_Op SwitchArgMinMax(bool is_arg_min) {
  return is_arg_min ? PoplibsOp::ArgMin : PoplibsOp::ArgMax;
}

// Constructor.
HloArgMinMax::HloArgMinMax(HloInstruction* input, const Shape shape,
                           int64 axis_, bool is_min)
    : HloPoplarInstruction(shape, {input},
                           GetPoplibsCustomOpTargetString(
                               PoplibsOp::Popnn, SwitchArgMinMax(is_min)),
                           axis),
      axis(axis_) {}

absl::flat_hash_set<int64> HloArgMinMax::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloArgMinMax::LayoutDependencies() const {
  return {};
}

uint64 HloArgMinMax::NumberOfInplaceOperands() const { return 0; }

bool HloArgMinMax::IsPopOpsElementwise() const { return false; }

// Creates an instance of a HloOneHotInstruction
std::unique_ptr<HloInstruction> CreateHloArgMinMax(HloInstruction* input,
                                                   const Shape& shape,
                                                   int64 axis, bool is_min) {
  if (is_min) {
    return absl::make_unique<HloArgMin>(input, shape, axis);
  } else {
    return absl::make_unique<HloArgMax>(input, shape, axis);
  }
}

std::unique_ptr<HloInstruction> HloArgMinMax::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext*) const {
  bool isArgMin = DynCast<HloArgMin>(this) != nullptr;

  return CreateHloArgMinMax(operands[0], shape, Axis(), isArgMin);
}

namespace {

static HloPoplarInstructionFactory argmax_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::ArgMax),
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
      TF_ASSIGN_OR_RETURN(int64 axis, attribute_map.GetAttributeAsInt("axis"));

      return CreateHloArgMinMax(call->mutable_operand(0), call->shape(), axis,
                                false);
    });

static HloPoplarInstructionFactory argmin_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn, PoplibsOp::ArgMin),
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
      TF_ASSIGN_OR_RETURN(int64 axis, attribute_map.GetAttributeAsInt("axis"));

      return CreateHloArgMinMax(call->mutable_operand(0), call->shape(), axis,
                                true);
    });

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
