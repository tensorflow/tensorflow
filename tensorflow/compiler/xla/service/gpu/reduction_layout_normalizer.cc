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

#include "tensorflow/compiler/xla/service/gpu/reduction_layout_normalizer.h"

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

class EnforceMinorToMajorReduceOpVisitor : public DfsHloRewriteVisitor {
  Status HandleReduce(HloInstruction *reduce) override {
    VLOG(5) << "Input: " << reduce->ToString();
    HloInstruction *operand = reduce->mutable_operand(0);
    const Shape &operand_shape = operand->shape();
    const Layout &operand_layout = operand_shape.layout();
    const Shape &reduce_shape = reduce->shape();

    if (!reduce_shape.IsArray()) {
      // TODO(cheshire): Handle variadic reduction.
      return Status::OK();
    }

    std::vector<int64> new_reduce_dimensions;
    std::vector<int64> new_operand_shape_data;
    std::vector<int64> new_reduce_shape_data;

    // The layout order of the reduction output can be different to the
    // ordering of kept dimensions in the input operand, thus we need to
    // calculate the new layout.
    std::vector<int64> new_reduce_shape_layout(reduce_shape.rank());
    std::vector<int64> reduce_shape_logical_to_physical =
        LayoutUtil::MakeLogicalToPhysical(reduce_shape.layout());

    auto to_reduce_logical_dim = [&](int64 op_logical_dim) {
      return op_logical_dim -
             absl::c_count_if(reduce->dimensions(), [&](int64 dim) {
               CHECK(dim != op_logical_dim);
               return dim < op_logical_dim;
             });
    };

    for (int i = 0; i < operand_shape.rank(); i++) {
      // Process the dimensions in the major-to-minor order in order to enforce
      // the default layout.
      int64 major_to_minor_dim_idx = operand_shape.rank() - i - 1;
      int64 logical_dim = operand_layout.minor_to_major(major_to_minor_dim_idx);
      int64 dim_size = operand_shape.dimensions(logical_dim);
      VLOG(5) << "Processing logical dimension " << logical_dim << " of size "
              << dim_size;
      new_operand_shape_data.push_back(dim_size);

      if (absl::c_linear_search(reduce->dimensions(), logical_dim)) {
        new_reduce_dimensions.push_back(i);
      } else {
        new_reduce_shape_data.push_back(dim_size);
        int64 logical_reduce_dim = to_reduce_logical_dim(logical_dim);
        int64 physical_reduce_dim =
            reduce_shape_logical_to_physical[logical_reduce_dim];
        VLOG(5) << "logical_reduce_dim = " << logical_reduce_dim << ", "
                << "physical_reduce_dim = " << physical_reduce_dim;
        new_reduce_shape_layout[reduce_shape.rank() - physical_reduce_dim - 1] =
            new_reduce_shape_data.size() - 1;
      }
    }

    Shape new_operand_shape = ShapeUtil::MakeShape(operand_shape.element_type(),
                                                   new_operand_shape_data);
    if (new_operand_shape == operand_shape) {
      return Status::OK();
    }

    Shape new_reduce_shape = ShapeUtil::MakeShapeWithLayout(
        reduce_shape.element_type(), new_reduce_shape_data,
        new_reduce_shape_layout);
    HloInstruction *canonical_reduce_input = reduce->parent()->AddInstruction(
        HloInstruction::CreateBitcast(new_operand_shape, operand));

    VLOG(5) << "Reduction input: " << canonical_reduce_input->ToString();
    std::unique_ptr<HloInstruction> new_reduce = HloInstruction::CreateReduce(
        new_reduce_shape, canonical_reduce_input, reduce->mutable_operand(1),
        new_reduce_dimensions, reduce->to_apply());
    VLOG(5) << "Generated new reduction: " << new_reduce->ToString();

    if (new_reduce_shape != reduce_shape) {
      HloInstruction *wrapped_reduce =
          reduce->parent()->AddInstruction(std::move(new_reduce));
      new_reduce = HloInstruction::CreateBitcast(reduce_shape, wrapped_reduce);
    }

    VLOG(5) << "Generated output: " << new_reduce->ToString();
    return ReplaceWithNewInstruction(reduce, std::move(new_reduce));
  }
};

StatusOr<bool> ReductionLayoutNormalizer::Run(HloModule *module) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      EnforceMinorToMajorReduceOpVisitor().RunOnModule(module));
  return changed;
}

}  // namespace gpu
}  // namespace xla
