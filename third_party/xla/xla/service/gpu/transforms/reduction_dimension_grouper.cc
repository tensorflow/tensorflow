/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/reduction_dimension_grouper.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

class ReduceDimensionGroupVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleReduce(HloInstruction *hlo) override {
    auto reduce = Cast<HloReduceInstruction>(hlo);

    VLOG(4) << "Input: " << reduce->ToString();

    absl::InlinedVector<HloInstruction *, 2> reduce_inputs_grouped;
    std::vector<int64_t> reduced_dims_grouped;

    int idx = -1;
    for (HloInstruction *operand : reduce->inputs()) {
      idx++;
      std::vector<int64_t> new_grouped_dims;
      const Shape &shape = operand->shape();
      CHECK(shape == LayoutUtil::GetWithDefaultLayout(shape))
          << "Default layout should be enforced on reduction operand";
      auto is_reduced = [&](int dim) {
        return absl::c_linear_search(reduce->dimensions(), dim);
      };

      bool changed = false;
      int64_t next_dim_size = 1;

      // Since we have enforced the standard layout, iteration over logical
      // dimensions is equivalent to iteration over the major-to-minor order.
      for (int logical_dim = 0; logical_dim < shape.rank(); logical_dim++) {
        VLOG(5) << "Processing dimension " << logical_dim << " of size "
                << shape.dimensions(logical_dim);
        if (is_reduced(logical_dim) && logical_dim < shape.rank() - 1 &&
            is_reduced(logical_dim + 1)) {
          VLOG(5) << "This and consecutive dimension are reduced, merging";
          changed = true;
          next_dim_size *= shape.dimensions(logical_dim);
          continue;
        }

        if (is_reduced(logical_dim)) {
          new_grouped_dims.push_back(next_dim_size *
                                     shape.dimensions(logical_dim));
          if (idx == 0) {
            // Only populate for first argument.
            reduced_dims_grouped.push_back(new_grouped_dims.size() - 1);
          }
          next_dim_size = 1;
        } else {
          new_grouped_dims.push_back(shape.dimensions(logical_dim));
        }
      }

      if (!changed) {  // Since all inputs have same shape dimensions.
        return absl::OkStatus();
      }

      Shape grouped_shape =
          ShapeUtil::MakeShape(shape.element_type(), new_grouped_dims);
      reduce_inputs_grouped.push_back(reduce->parent()->AddInstruction(
          HloInstruction::CreateBitcast(grouped_shape, operand),
          &operand->metadata()));
      VLOG(5) << "Adding bitcast: " << reduce_inputs_grouped.back()->ToString();
    }

    std::unique_ptr<HloInstruction> new_reduce = HloInstruction::CreateReduce(
        reduce->shape(), reduce_inputs_grouped, reduce->init_values(),
        reduced_dims_grouped, reduce->to_apply());
    VLOG(5) << "Generated new reduction: " << new_reduce->ToString();
    return ReplaceWithNewInstruction(reduce, std::move(new_reduce));
  }
};

absl::StatusOr<bool> ReductionDimensionGrouper::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  TF_ASSIGN_OR_RETURN(bool changed, ReduceDimensionGroupVisitor().RunOnModule(
                                        module, execution_threads));
  return changed;
}

}  // namespace gpu
}  // namespace xla
