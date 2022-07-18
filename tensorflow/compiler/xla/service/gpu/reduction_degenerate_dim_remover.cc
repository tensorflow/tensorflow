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
  Status HandleReduce(HloInstruction *hlo) override {
    auto instr = Cast<HloReduceInstruction>(hlo);
    absl::InlinedVector<HloInstruction *, 2> input_reshapes;
    absl::InlinedVector<Shape, 2> canonical_reduce_shapes;

    int idx = -1;
    std::vector<int64_t> updated_reduced_dimensions;
    for (HloInstruction *reduced_op : instr->inputs()) {
      idx++;
      const Shape &input_shape = reduced_op->shape();
      const Shape &reduce_shape = instr->shape().IsTuple()
                                      ? instr->shape().tuple_shapes(idx)
                                      : instr->shape();

      if (!ShapeUtil::HasDegenerateDimensions(reduced_op->shape())) {
        return OkStatus();
      }
      Shape canonical_input_shape =
          ShapeUtil::DropDegenerateDimensions(input_shape);

      Shape canonical_reduce_shape =
          ShapeUtil::DropDegenerateDimensions(reduce_shape);

      auto reduced_dimensions = instr->dimensions();
      int64_t shift = 0;

      for (int dim = 0; dim < input_shape.rank(); dim++) {
        if (input_shape.dimensions(dim) == 1) {
          shift++;
        } else {
          if (absl::c_linear_search(reduced_dimensions, dim) && idx == 0) {
            // Only populate on first iteration.
            updated_reduced_dimensions.push_back(dim - shift);
          }
        }
      }

      if (updated_reduced_dimensions.empty()) {
        std::unique_ptr<HloInstruction> reshape =
            HloInstruction::CreateBitcast(reduce_shape, reduced_op);
        return ReplaceWithNewInstruction(instr, std::move(reshape));
      }

      input_reshapes.push_back(instr->parent()->AddInstruction(
          HloInstruction::CreateBitcast(canonical_input_shape, reduced_op)));
      canonical_reduce_shapes.push_back(canonical_reduce_shape);
    }

    Shape canonical_reduce_shape =
        ShapeUtil::MakeMaybeTupleShape(canonical_reduce_shapes);
    const Shape &orig_reduce_shape = instr->shape();
    std::unique_ptr<HloInstruction> new_reduce = HloInstruction::CreateReduce(
        canonical_reduce_shape, input_reshapes, instr->init_values(),
        updated_reduced_dimensions, instr->to_apply());

    if (canonical_reduce_shape != instr->shape()) {
      HloInstruction *wrapped_reduce =
          instr->parent()->AddInstruction(std::move(new_reduce));
      absl::InlinedVector<HloInstruction *, 2> out;
      if (!canonical_reduce_shape.IsTuple()) {
        new_reduce =
            HloInstruction::CreateBitcast(orig_reduce_shape, wrapped_reduce);
      } else {
        for (int oidx = 0; oidx < instr->input_count(); oidx++) {
          HloInstruction *gte = instr->parent()->AddInstruction(
              HloInstruction::CreateGetTupleElement(wrapped_reduce, oidx));
          out.push_back(
              instr->parent()->AddInstruction(HloInstruction::CreateBitcast(
                  orig_reduce_shape.tuple_shapes(oidx), gte)));
        }
        new_reduce = HloInstruction::CreateTuple(out);
      }
    }

    return ReplaceWithNewInstruction(instr, std::move(new_reduce));
  }
};

StatusOr<bool> ReductionDegenerateDimRemover::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      ReductionDegenerateDimRemoverVisitor().RunOnModule(
                          module, execution_threads));
  return changed;
}

}  // namespace gpu
}  // namespace xla
