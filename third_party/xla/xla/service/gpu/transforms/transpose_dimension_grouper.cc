/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/transpose_dimension_grouper.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

class TransposeDimensionGroupVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleTranspose(HloInstruction *transpose) override {
    VLOG(4) << "Input: " << transpose->ToString();
    absl::InlinedVector<int64_t, 3> permutation;
    auto normalized_dims = ShapeUtil::GetNormalizedLogicalTransposeShape(
        transpose->shape(), transpose->dimensions(), permutation);
    if (!normalized_dims.has_value() ||
        normalized_dims->size() == transpose->shape().rank()) {
      return absl::OkStatus();
    }
    auto normalized_operand_dims =
        ComposePermutations(*normalized_dims, InversePermutation(permutation));
    Shape grouped_operand_shape = ShapeUtil::MakeShapeWithDescendingLayout(
        transpose->shape().element_type(), normalized_operand_dims);
    auto new_operand = transpose->AddInstruction(HloInstruction::CreateBitcast(
        grouped_operand_shape, transpose->mutable_operand(0)));
    Shape grouped_shape = ShapeUtil::MakeShapeWithDescendingLayout(
        transpose->shape().element_type(), *normalized_dims);
    auto new_transpose =
        transpose->AddInstruction(HloInstruction::CreateTranspose(
            grouped_shape, new_operand, permutation));
    VLOG(5) << "Generated new transpose: " << new_transpose->ToString();
    return ReplaceWithNewInstruction(
        transpose,
        HloInstruction::CreateBitcast(transpose->shape(), new_transpose));
  }
};

absl::StatusOr<bool> TransposeDimensionGrouper::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  TF_ASSIGN_OR_RETURN(
      bool changed,
      TransposeDimensionGroupVisitor().RunOnModule(module, execution_threads));
  return changed;
}

}  // namespace gpu
}  // namespace xla
