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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/norm.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"
#include "tensorflow/compiler/tf2xla/type_util.h"

namespace xla {
namespace poplarplugin {

HloGroupNormTrainInstruction::HloGroupNormTrainInstruction(
    const Shape& shape, HloInstruction* const operand,
    HloInstruction* const scale, HloInstruction* const offset, int32 num_groups,
    float epsilon, int feature_index)
    : HloNormInstruction(shape, {operand, scale, offset},
                         GetPoplibsCustomOpTargetString(
                             PoplibsOp::Popnn, PoplibsOp::GroupNormTraining),
                         num_groups, epsilon, feature_index) {}

const HloInstruction* HloGroupNormTrainInstruction::operand() const {
  return HloInstruction::operand(0);
}

const HloInstruction* HloGroupNormTrainInstruction::scale() const {
  return HloInstruction::operand(1);
}

const HloInstruction* HloGroupNormTrainInstruction::offset() const {
  return HloInstruction::operand(2);
}

absl::flat_hash_set<int64> HloGroupNormTrainInstruction::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloGroupNormTrainInstruction::LayoutDependencies() const {
  return {{1, 0}, {2, 0}};
}

uint64 HloGroupNormTrainInstruction::NumberOfInplaceOperands() const {
  return 0;
}

bool HloGroupNormTrainInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloGroupNormTrainInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateGroupNormTrain(shape, new_operands[0], new_operands[1],
                              new_operands[2], num_groups(), epsilon(),
                              feature_index());
}

std::unique_ptr<HloInstruction> CreateGroupNormTrain(
    const Shape& shape, HloInstruction* const operand,
    HloInstruction* const scale, HloInstruction* const offset, int32 num_groups,
    float epsilon, int feature_index) {
  return absl::make_unique<HloGroupNormTrainInstruction>(
      shape, operand, scale, offset, num_groups, epsilon, feature_index);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloGroupNormTrainFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(int32 num_groups,
                      attribute_map.GetAttributeAsInt("num_groups"));

  TF_ASSIGN_OR_RETURN(float epsilon,
                      attribute_map.GetAttributeAsFloat("epsilon"));

  TF_ASSIGN_OR_RETURN(int feature_index,
                      attribute_map.GetAttributeAsInt("feature_index"));

  auto args = call->operands();

  return CreateGroupNormTrain(call->shape(), args[0], args[1], args[2],
                              num_groups, epsilon, feature_index);
}

static HloPoplarInstructionFactory group_norm_train_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn,
                                   PoplibsOp::GroupNormTraining),
    HloGroupNormTrainFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
