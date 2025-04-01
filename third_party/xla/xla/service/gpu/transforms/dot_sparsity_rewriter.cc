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

#include "xla/service/gpu/transforms/dot_sparsity_rewriter.h"

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class SparseDotRewriterImpl : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleDot(HloInstruction* instr) override {
    // Only handle sparse dots with a single RHS sparse descriptor.
    HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
    if (dot->sparse_operands() != 1 || dot->sparsity().front().index() != 1) {
      return absl::OkStatus();
    }

    HloInstruction* lhs = dot->mutable_operand(0);
    HloInstruction* rhs = dot->mutable_operand(1);
    HloInstruction* meta = dot->mutable_operand(2);

    // Swap LHS and RHS in the attributes.
    DotDimensionNumbers dnums = dot->dot_dimension_numbers();
    std::swap(*dnums.mutable_lhs_batch_dimensions(),
              *dnums.mutable_rhs_batch_dimensions());
    std::swap(*dnums.mutable_lhs_contracting_dimensions(),
              *dnums.mutable_rhs_contracting_dimensions());

    PrecisionConfig precision_config = dot->precision_config();
    std::swap(precision_config.mutable_operand_precision()->at(0),
              precision_config.mutable_operand_precision()->at(1));

    SparsityDescriptor sparsity = dot->sparsity().front();
    sparsity.set_index(0);

    // Create new dot with LHS and RHS swapped.
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_dot,
        MakeDotHlo(rhs, lhs, dnums, precision_config,
                   dot->shape().element_type(), {std::move(sparsity)}, {meta}));
    dot->SetupDerivedInstruction(new_dot);

    // Result dimensions: <batch>, <rhs_noncontracting>, <lhs_noncontracting>
    int batch_dims = dnums.lhs_batch_dimensions().size();
    int new_lhs_noncontracting = rhs->shape().dimensions_size() - batch_dims -
                                 dnums.lhs_contracting_dimensions().size();
    int new_rhs_noncontracting = lhs->shape().dimensions_size() - batch_dims -
                                 dnums.rhs_contracting_dimensions().size();

    int rank = dot->shape().dimensions_size();
    DimensionVector dimensions(rank);
    for (int i = 0; i < batch_dims; ++i) {
      dimensions[i] = i;
    }
    for (int i = 0; i < new_lhs_noncontracting; ++i) {
      dimensions[i + batch_dims] = i + batch_dims + new_rhs_noncontracting;
    }
    for (int i = 0; i < new_rhs_noncontracting; ++i) {
      dimensions[i + batch_dims + new_lhs_noncontracting] = i + batch_dims;
    }

    // Transpose the result.
    TF_ASSIGN_OR_RETURN(HloInstruction * transpose,
                        MakeTransposeHlo(new_dot, dimensions));
    transpose->set_metadata(dot->metadata());
    *transpose->mutable_shape()->mutable_layout() = dot->shape().layout();

    return ReplaceInstruction(dot, transpose);
  }
};

}  // namespace

absl::StatusOr<bool> DotSparsityRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return SparseDotRewriterImpl().RunOnModule(module, execution_threads);
}

}  // namespace gpu
}  // namespace xla
