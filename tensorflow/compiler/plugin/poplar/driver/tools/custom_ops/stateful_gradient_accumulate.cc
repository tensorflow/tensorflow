/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

namespace xla {
namespace poplarplugin {
namespace {
PoplibsOp_Op SwitchAllReduce(bool is_all_reduce) {
  return is_all_reduce ? PoplibsOp::StatefulGradientAccumulateAndAllReduce
                       : PoplibsOp::StatefulGradientAccumulate;
}
}  // namespace

HloStatefulGradientAccumulate::HloStatefulGradientAccumulate(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches,
    bool is_all_reduce)
    : HloPoplarInstruction(
          GetHloPoplarInstructionShape(operands), operands,
          GetPoplibsCustomOpTargetString(PoplibsOp::Poputil,
                                         SwitchAllReduce(is_all_reduce)),
          num_mini_batches),
      num_mini_batches_(num_mini_batches) {}

absl::flat_hash_set<int64> HloStatefulGradientAccumulate::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloStatefulGradientAccumulate::LayoutDependencies() const {
  return {};
}

uint64 HloStatefulGradientAccumulate::NumberOfInplaceOperands() const {
  return operand_count();
}

bool HloStatefulGradientAccumulate::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloStatefulGradientAccumulate::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloStatefulGradientAccumulate>(new_operands,
                                                          num_mini_batches_);
}

std::unique_ptr<HloInstruction> CreateStatefulGradientAccumulation(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches) {
  return absl::make_unique<HloStatefulGradientAccumulate>(operands,
                                                          num_mini_batches);
}

HloStatefulGradientAccumulateAndAllReduce::
    HloStatefulGradientAccumulateAndAllReduce(
        absl::Span<HloInstruction* const> operands, int32 num_mini_batches)
    : HloStatefulGradientAccumulate(operands, num_mini_batches, true) {}

std::unique_ptr<HloInstruction>
HloStatefulGradientAccumulateAndAllReduce::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloStatefulGradientAccumulateAndAllReduce>(
      new_operands, num_mini_batches_);
}

std::unique_ptr<HloInstruction> CreateStatefulGradientAccumulateAndAllReduce(
    absl::Span<HloInstruction* const> operands, int32 num_mini_batches) {
  return absl::make_unique<HloStatefulGradientAccumulateAndAllReduce>(
      operands, num_mini_batches);
}

namespace {

StatusOr<std::unique_ptr<HloInstruction>>
HloStatefulGradientAccumulateFactoryFunc(HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  // Get the attribute values
  TF_ASSIGN_OR_RETURN(int32 num_mini_batches,
                      attribute_map.GetAttributeAsInt("num_mini_batches"));

  return CreateStatefulGradientAccumulation(call->operands(), num_mini_batches);
}

static HloPoplarInstructionFactory stateful_gradient_accumulate_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Poputil,
                                   PoplibsOp::StatefulGradientAccumulate),
    HloStatefulGradientAccumulateFactoryFunc);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
