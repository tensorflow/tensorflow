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

#include "xla/service/gpu/transforms/collectives/all_gather_major_dimension_rewriter.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

namespace {
bool ShouldBeOptimized(const HloInstruction& all_gather) {
  return all_gather.operand_count() == 1 &&
         all_gather.operand(0)->opcode() == HloOpcode::kCopy &&
         all_gather.operand(0)->user_count() == 1 &&
         all_gather.user_count() == 1 &&
         all_gather.users()[0]->opcode() == HloOpcode::kCopy;
}
}  // namespace

absl::Status AllGatherMajorDimensionRewriter::Visitor::HandleAllGather(
    HloInstruction* all_gather) {
  if (!ShouldBeOptimized(*all_gather)) {
    return absl::OkStatus();
  }

  HloComputation* computation = all_gather->parent();

  const int64_t original_gather_dim =
      Cast<HloAllGatherInstruction>(all_gather)->all_gather_dimension();
  const int64_t shard_count =
      all_gather->shape().dimensions(original_gather_dim) /
      all_gather->operand(0)->shape().dimensions(original_gather_dim);

  HloInstruction* new_input =
      all_gather->mutable_operand(0)->mutable_operand(0);
  Shape new_all_gather_shape = new_input->shape();
  // Always gather the major most dimension.
  const int64_t new_gathered_dim =
      new_input->shape().layout().minor_to_major().back();
  new_all_gather_shape.set_dimensions(
      new_gathered_dim,
      new_all_gather_shape.dimensions(new_gathered_dim) * shard_count);

  HloInstruction* new_all_gather = computation->AddInstruction(
      all_gather->CloneWithNewOperands(new_all_gather_shape, {new_input}));
  Cast<HloAllGatherInstruction>(new_all_gather)
      ->set_all_gather_dimension(new_gathered_dim);

  const absl::Span<const int64_t> first_bitcast_dimensions =
      ShapeUtil::InsertDimensionAtIndex(new_input->shape(), new_gathered_dim,
                                        shard_count)
          .dimensions();

  auto insert_gathered_dimension =
      [&new_gathered_dim](const Layout& layout, const int64_t insert_after) {
        absl::InlinedVector<int64_t, 4> result;
        for (const int64_t dim_idx : layout.minor_to_major()) {
          result.push_back(dim_idx + (dim_idx >= new_gathered_dim));
          if (dim_idx == insert_after) {
            result.push_back(new_gathered_dim);
          }
        }
        return result;
      };

  HloInstruction* first_bitcast = MakeBitcastHlo(
      new_all_gather,
      ShapeUtil::MakeShapeWithDenseLayout(
          all_gather->shape().element_type(), first_bitcast_dimensions,
          insert_gathered_dimension(new_input->shape().layout(),
                                    new_gathered_dim)));

  HloInstruction* original_output_copy = all_gather->users()[0];

  HloInstruction* copy = MakeCopyHlo(
      first_bitcast,
      ShapeUtil::MakeShapeWithDenseLayout(
          all_gather->shape().element_type(), first_bitcast_dimensions,
          insert_gathered_dimension(original_output_copy->shape().layout(),
                                    original_gather_dim)));

  // Bitcast to the original shape.
  return ReplaceInstruction(
      original_output_copy,
      MakeBitcastHlo(copy, original_output_copy->shape()));
}

absl::StatusOr<bool> AllGatherMajorDimensionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  Visitor visitor;
  return visitor.RunOnModule(module, execution_threads);
}

}  // namespace gpu
}  // namespace xla
