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

#include "tensorflow/compiler/xla/service/gpu/reduction_dimension_grouper.h"

#include <algorithm>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {

class ReduceDimensionGroupVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleReduce(HloInstruction *reduce) override {
    VLOG(4) << "Input: " << reduce->ToString();

    if (!reduce->shape().IsArray()) {
      // TODO(cheshire): Handle variadic reduction.
      return Status::OK();
    }

    std::vector<int64> new_grouped_dims;
    std::vector<int64> reduced_dims_grouped;
    HloInstruction *operand = reduce->mutable_operand(0);
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
        reduced_dims_grouped.push_back(new_grouped_dims.size() - 1);
        next_dim_size = 1;
      } else {
        new_grouped_dims.push_back(shape.dimensions(logical_dim));
      }
    }

    if (!changed) {
      return Status::OK();
    }

    Shape grouped_shape =
        ShapeUtil::MakeShape(shape.element_type(), new_grouped_dims);
    HloInstruction *reduce_input_grouped = reduce->parent()->AddInstruction(
        HloInstruction::CreateBitcast(grouped_shape, operand));

    std::unique_ptr<HloInstruction> new_reduce = HloInstruction::CreateReduce(
        reduce->shape(), reduce_input_grouped, reduce->mutable_operand(1),
        reduced_dims_grouped, reduce->to_apply());
    VLOG(5) << "Generated new reduction: " << new_reduce->ToString();
    return ReplaceWithNewInstruction(reduce, std::move(new_reduce));
  }
};

StatusOr<bool> ReductionDimensionGrouper::Run(HloModule *module) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      ReduceDimensionGroupVisitor().RunOnModule(module));
  return changed;
}

}  // namespace gpu
}  // namespace xla
