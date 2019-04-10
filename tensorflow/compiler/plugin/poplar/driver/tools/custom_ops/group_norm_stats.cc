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

HloGroupNormStatsInstruction::HloGroupNormStatsInstruction(
    const Shape& shape, HloInstruction* const operand, int32 num_groups,
    float epsilon, int feature_index)
    : HloNormInstruction(shape, {operand},
                         GetPoplibsCustomOpTargetString(
                             PoplibsOp::Popnn, PoplibsOp::GroupNormStatistics),
                         num_groups, epsilon, feature_index) {}

const HloInstruction* HloGroupNormStatsInstruction::operand() const {
  return HloInstruction::operand(0);
}

absl::flat_hash_set<int64> HloGroupNormStatsInstruction::AllocatingIndices()
    const {
  return {};
}

absl::flat_hash_map<int64, int64>
HloGroupNormStatsInstruction::LayoutDependencies() const {
  return {};
}

uint64 HloGroupNormStatsInstruction::NumberOfInplaceOperands() const {
  return 0;
}

bool HloGroupNormStatsInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloGroupNormStatsInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return CreateGroupNormStats(shape, new_operands[0], num_groups(), epsilon(),
                              feature_index());
}

std::unique_ptr<HloInstruction> CreateGroupNormStats(
    const Shape& shape, HloInstruction* const operand, int32 num_groups,
    float epsilon, int feature_index) {
  return absl::make_unique<HloGroupNormStatsInstruction>(
      shape, operand, num_groups, epsilon, feature_index);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloGroupNormStatsFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);

  TF_ASSIGN_OR_RETURN(int32 num_groups,
                      attribute_map.GetAttributeAsInt("num_groups"));

  TF_ASSIGN_OR_RETURN(float epsilon,
                      attribute_map.GetAttributeAsFloat("epsilon"));

  TF_ASSIGN_OR_RETURN(int feature_index,
                      attribute_map.GetAttributeAsInt("feature_index"));

  auto args = call->operands();

  return CreateGroupNormStats(call->shape(), args[0], num_groups, epsilon,
                              feature_index);
}

static HloPoplarInstructionFactory group_norm_stats_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Popnn,
                                   PoplibsOp::GroupNormStatistics),
    HloGroupNormStatsFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
