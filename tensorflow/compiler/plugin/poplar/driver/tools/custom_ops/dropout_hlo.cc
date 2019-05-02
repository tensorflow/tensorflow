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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/dropout_hlo.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/poplibs_ops.pb.h"

namespace xla {
namespace poplarplugin {

HloDropoutInstruction::HloDropoutInstruction(HloInstruction* X,
                                             HloInstruction* seed, float rate_,
                                             float scale_, int32_t seed_mod,
                                             bool should_use_user_seed)
    : HloPoplarInstruction(
          xla::ShapeUtil::MakeTupleShape({X->shape(), seed->shape()}),
          {X, seed},
          GetPoplibsCustomOpTargetString(PoplibsOp::Poprand,
                                         PoplibsOp::Dropout),
          {}),
      scale(scale_),
      rate(rate_),
      seed_modifier(seed_mod),
      is_user_seed(should_use_user_seed) {}

absl::flat_hash_set<int64> HloDropoutInstruction::AllocatingIndices() const {
  return {};
}

absl::flat_hash_map<int64, int64> HloDropoutInstruction::LayoutDependencies()
    const {
  return {};
}

uint64 HloDropoutInstruction::NumberOfInplaceOperands() const { return 0; }

bool HloDropoutInstruction::IsPopOpsElementwise() const { return true; }

std::unique_ptr<HloInstruction> HloDropoutInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloDropoutInstruction>(
      new_operands[0], new_operands[1], Rate(), Scale(), SeedModifier(),
      IsUserSeed());
}

std::unique_ptr<HloInstruction> CreateDropout(HloInstruction* operand,
                                              HloInstruction* seed, float rate,
                                              float scale,
                                              uint32_t seed_modifier,
                                              bool should_use_user_seed) {
  return absl::make_unique<HloDropoutInstruction>(
      operand, seed, rate, scale, seed_modifier, should_use_user_seed);
}

namespace {

StatusOr<std::unique_ptr<HloInstruction>> HloDropoutInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  // Get the attribute values
  TF_ASSIGN_OR_RETURN(float rate, attribute_map.GetAttributeAsFloat("rate"));
  TF_ASSIGN_OR_RETURN(float scale, attribute_map.GetAttributeAsFloat("scale"));
  TF_ASSIGN_OR_RETURN(int32_t seed_modifier,
                      attribute_map.GetAttributeAsInt("seed_modifier"));
  TF_ASSIGN_OR_RETURN(bool should_use_user_seed,
                      attribute_map.GetAttributeAsBool("is_using_user_seed"));

  return CreateDropout(call->mutable_operand(0), call->mutable_operand(1), rate,
                       scale, seed_modifier, should_use_user_seed);
}

static HloPoplarInstructionFactory dropout_factory(
    GetPoplibsCustomOpTargetString(PoplibsOp::Poprand, PoplibsOp::Dropout),
    HloDropoutInstructionFactoryFunc);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
