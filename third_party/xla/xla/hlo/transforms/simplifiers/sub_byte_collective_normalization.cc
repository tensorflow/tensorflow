/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/sub_byte_collective_normalization.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

HloInstruction* ReshapeAndCastToWiderType(HloInstruction* input,
                                          const PrimitiveType type) {
  const Shape& input_shape = input->shape();
  const int64_t ratio = primitive_util::BitWidth(type) /
                        primitive_util::BitWidth(input_shape.element_type());

  std::vector<int64_t> bitcast_dimensions(input_shape.dimensions().begin(),
                                          input_shape.dimensions().end());
  bitcast_dimensions[input_shape.layout().minor_to_major(0)] /= ratio;
  bitcast_dimensions.push_back(ratio);
  Shape bitcast_shape =
      ShapeUtil::MakeShape(input_shape.element_type(), bitcast_dimensions);
  if (input_shape.has_layout()) {
    *bitcast_shape.mutable_layout() = LayoutUtil::MoveDimToMinor(
        input_shape.layout(), bitcast_dimensions.size() - 1);
  }
  HloInstruction* bitcast = input->parent()->AddInstruction(
      HloInstruction::CreateBitcast(bitcast_shape, input));
  HloInstruction* result =
      input->parent()->AddInstruction(HloInstruction::CreateBitcastConvert(
          ShapeUtil::MakeShape(
              type, std::vector<int64_t>(bitcast_dimensions.begin(),
                                         bitcast_dimensions.end() - 1)),
          bitcast));
  if (input_shape.has_layout()) {
    *result->mutable_shape()->mutable_layout() = input_shape.layout();
    result->mutable_shape()->mutable_layout()->set_element_size_in_bits(0);
  }
  return result;
}

HloInstruction* CastToNarrowerTypeAndReshape(HloInstruction* input,
                                             const Shape& shape) {
  const Shape& input_shape = input->shape();
  const int64_t ratio = primitive_util::BitWidth(input_shape.element_type()) /
                        primitive_util::BitWidth(shape.element_type());

  std::vector<int64_t> convert_dimensions(input_shape.dimensions().begin(),
                                          input_shape.dimensions().end());
  convert_dimensions.push_back(ratio);
  Shape convert_shape =
      ShapeUtil::MakeShape(shape.element_type(), convert_dimensions);
  if (shape.has_layout()) {
    *convert_shape.mutable_layout() = LayoutUtil::MoveDimToMinor(
        input_shape.layout(), convert_dimensions.size() - 1);
    convert_shape.mutable_layout()->set_element_size_in_bits(
        shape.layout().element_size_in_bits());
  }
  HloInstruction* convert = input->parent()->AddInstruction(
      HloInstruction::CreateBitcastConvert(convert_shape, input));
  return input->parent()->AddInstruction(
      HloInstruction::CreateBitcast(shape, convert));
}

bool CanBeRepresentedAs(const Shape& shape, const PrimitiveType casted_type) {
  const int64_t ratio = primitive_util::BitWidth(casted_type) /
                        primitive_util::BitWidth(shape.element_type());
  return primitive_util::IsSubByteNonPredType(shape.element_type()) &&
         shape.layout().element_size_in_bits() ==
             primitive_util::BitWidth(shape.element_type()) &&
         shape.dimensions(shape.layout().minor_to_major(0)) % ratio == 0;
}

class SubByteCollectiveNormalizationVisitor : public DfsHloRewriteVisitor {
 public:
  SubByteCollectiveNormalizationVisitor() = default;
  absl::Status HandleAllGather(HloInstruction* hlo) override;
  absl::Status HandleAllToAll(HloInstruction* hlo) override;
  absl::Status HandleCollectiveBroadcast(HloInstruction* hlo) override;
  absl::Status HandleCollectivePermute(HloInstruction* hlo) override;

 private:
  bool ShouldProcessInstruction(const HloInstruction& hlo) const;
  absl::Status ProcessCollectiveInstruction(HloInstruction& hlo);
  static constexpr PrimitiveType casted_type_ = S8;
};

bool SubByteCollectiveNormalizationVisitor::ShouldProcessInstruction(
    const HloInstruction& hlo) const {
  return hlo.operand_count() == 1 &&
         CanBeRepresentedAs(hlo.operand(0)->shape(), casted_type_);
}

absl::Status SubByteCollectiveNormalizationVisitor::HandleAllGather(
    HloInstruction* hlo) {
  return ProcessCollectiveInstruction(*hlo);
}

absl::Status SubByteCollectiveNormalizationVisitor::HandleAllToAll(
    HloInstruction* hlo) {
  if (!ShouldProcessInstruction(*hlo)) {
    return absl::OkStatus();
  }

  const int64_t ratio = primitive_util::BitWidth(casted_type_) /
                        primitive_util::BitWidth(hlo->shape().element_type());
  const auto* all_to_all = Cast<HloAllToAllInstruction>(hlo);
  if (all_to_all->split_dimension()) {
    TF_ASSIGN_OR_RETURN(const CollectiveOpGroupMode group_mode,
                        GetCollectiveOpGroupMode(all_to_all));
    const int64_t split_dimension_size =
        hlo->shape().dimensions(*all_to_all->split_dimension());
    if (split_dimension_size %
            (GetSubgroupSize(all_to_all, group_mode) * ratio) !=
        0) {
      return absl::OkStatus();
    }
  }
  return ProcessCollectiveInstruction(*hlo);
}

absl::Status SubByteCollectiveNormalizationVisitor::HandleCollectiveBroadcast(
    HloInstruction* hlo) {
  return ProcessCollectiveInstruction(*hlo);
}

absl::Status SubByteCollectiveNormalizationVisitor::HandleCollectivePermute(
    HloInstruction* hlo) {
  return ProcessCollectiveInstruction(*hlo);
}

absl::Status
SubByteCollectiveNormalizationVisitor::ProcessCollectiveInstruction(
    HloInstruction& hlo) {
  if (!ShouldProcessInstruction(hlo)) {
    return absl::OkStatus();
  }

  const int64_t ratio = primitive_util::BitWidth(casted_type_) /
                        primitive_util::BitWidth(hlo.shape().element_type());

  std::vector<int64_t> new_collective_dimensions(
      hlo.shape().dimensions().begin(), hlo.shape().dimensions().end());
  new_collective_dimensions[hlo.shape().layout().minor_to_major(0)] /= ratio;
  Shape new_collective_shape =
      ShapeUtil::MakeShape(casted_type_, new_collective_dimensions);
  if (hlo.shape().has_layout()) {
    *new_collective_shape.mutable_layout() = hlo.shape().layout();
    new_collective_shape.mutable_layout()->set_element_size_in_bits(0);
  }
  HloInstruction* new_collective =
      hlo.parent()->AddInstruction(hlo.CloneWithNewOperands(
          new_collective_shape,
          {ReshapeAndCastToWiderType(hlo.mutable_operand(0), casted_type_)}));
  TF_RETURN_IF_ERROR(hlo.parent()->ReplaceInstructionWithDifferentShape(
      &hlo, CastToNarrowerTypeAndReshape(new_collective, hlo.shape())));

  MarkAsChanged();
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> SubByteCollectiveNormalization::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  SubByteCollectiveNormalizationVisitor visitor;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  }

  return visitor.changed();
}

}  // namespace xla
