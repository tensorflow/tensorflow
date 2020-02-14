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

#include "tensorflow/compiler/xla/service/gpu/reduction_degenerate_dim_remover.h"

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

class ReductionDegenerateDimRemoverVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleReduce(HloInstruction *instr) override {
    HloInstruction *reduced_op = instr->mutable_operand(0);
    const Shape &input_shape = reduced_op->shape();
    const Shape &reduce_shape = instr->shape();

    if (!instr->shape().IsArray() ||
        !ShapeUtil::HasDegenerateDimensions(reduced_op->shape())) {
      return Status::OK();
    }
    Shape canonical_input_shape =
        ShapeUtil::DropDegenerateDimensions(input_shape);

    Shape canonical_reduce_shape =
        ShapeUtil::DropDegenerateDimensions(reduce_shape);

    const std::vector<int64> &reduced_dimensions = instr->dimensions();
    std::vector<int64> updated_reduced_dimensions;
    int64 shift = 0;

    for (int dim = 0; dim < input_shape.rank(); dim++) {
      if (input_shape.dimensions(dim) == 1) {
        shift++;
      } else {
        if (absl::c_linear_search(reduced_dimensions, dim)) {
          updated_reduced_dimensions.push_back(dim - shift);
        }
      }
    }

    HloInstruction *input_reshape = instr->parent()->AddInstruction(
        HloInstruction::CreateBitcast(canonical_input_shape, reduced_op));

    std::unique_ptr<HloInstruction> new_reduce = HloInstruction::CreateReduce(
        canonical_reduce_shape, input_reshape, instr->mutable_operand(1),
        updated_reduced_dimensions, instr->to_apply());

    if (canonical_reduce_shape != reduce_shape) {
      HloInstruction *wrapped_reduce =
          instr->parent()->AddInstruction(std::move(new_reduce));
      new_reduce = HloInstruction::CreateBitcast(reduce_shape, wrapped_reduce);
    }

    return ReplaceWithNewInstruction(instr, std::move(new_reduce));
  }
};

StatusOr<bool> ReductionDegenerateDimRemover::Run(HloModule *module) {
  TF_ASSIGN_OR_RETURN(
      bool changed, ReductionDegenerateDimRemoverVisitor().RunOnModule(module));
  return changed;
}

}  // namespace gpu
}  // namespace xla
