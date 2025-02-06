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

#include "xla/service/gpu/transforms/reduction_layout_normalizer.h"

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
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

class EnforceMinorToMajorReduceOpVisitor : public DfsHloRewriteVisitor {
  absl::Status HandleReduce(HloInstruction *hlo) override {
    auto reduce = Cast<HloReduceInstruction>(hlo);
    VLOG(5) << "Input: " << reduce->ToString();

    int operand_idx = -1;

    absl::InlinedVector<HloInstruction *, 2> canonical_reduce_inputs;
    absl::InlinedVector<Shape, 2> new_reduce_shapes;

    DimensionVector out_reduce_dimensions;
    const Shape &first_instruction_shape = reduce->inputs()[0]->shape();

    for (HloInstruction *operand : reduce->inputs()) {
      operand_idx++;

      if (operand_idx != 0 &&
          operand->shape().layout() != first_instruction_shape.layout()) {
        return FailedPrecondition(
            "Layout assignment should have assigned the same layout to all "
            "reduce inputs");
      }

      const Shape &operand_shape = operand->shape();
      const Layout &operand_layout = operand_shape.layout();

      const Shape &reduce_shape =
          reduce->shape().IsTuple() ? reduce->shape().tuple_shapes(operand_idx)
                                    : reduce->shape();

      DimensionVector new_reduce_dimensions;
      DimensionVector new_operand_shape_data;
      DimensionVector new_reduce_shape_data;

      // The layout order of the reduction output can be different to the
      // ordering of kept dimensions in the input operand, thus we need to
      // calculate the new layout.
      DimensionVector new_reduce_shape_layout(reduce_shape.rank());
      std::vector<int64_t> reduce_shape_logical_to_physical =
          LayoutUtil::MakeLogicalToPhysical(reduce_shape.layout());

      auto to_reduce_logical_dim = [&](int64_t op_logical_dim) {
        return op_logical_dim -
               absl::c_count_if(reduce->dimensions(), [&](int64_t dim) {
                 CHECK(dim != op_logical_dim);
                 return dim < op_logical_dim;
               });
      };

      for (int i = 0; i < operand_shape.rank(); i++) {
        // Process the dimensions in the major-to-minor order in order to
        // enforce the default layout.
        int64_t major_to_minor_dim_idx = operand_shape.rank() - i - 1;
        int64_t logical_dim =
            operand_layout.minor_to_major(major_to_minor_dim_idx);
        int64_t dim_size = operand_shape.dimensions(logical_dim);
        VLOG(5) << "Processing logical dimension " << logical_dim << " of size "
                << dim_size;
        new_operand_shape_data.push_back(dim_size);

        if (absl::c_linear_search(reduce->dimensions(), logical_dim)) {
          new_reduce_dimensions.push_back(i);
        } else {
          new_reduce_shape_data.push_back(dim_size);
          int64_t logical_reduce_dim = to_reduce_logical_dim(logical_dim);
          int64_t physical_reduce_dim =
              reduce_shape_logical_to_physical[logical_reduce_dim];
          VLOG(5) << "logical_reduce_dim = " << logical_reduce_dim << ", "
                  << "physical_reduce_dim = " << physical_reduce_dim;
          new_reduce_shape_layout[reduce_shape.rank() - physical_reduce_dim -
                                  1] = new_reduce_shape_data.size() - 1;
        }
      }

      Shape new_operand_shape = ShapeUtil::MakeShape(
          operand_shape.element_type(), new_operand_shape_data);
      Shape new_reduce_shape = ShapeUtil::MakeShapeWithDenseLayout(
          reduce_shape.element_type(), new_reduce_shape_data,
          new_reduce_shape_layout);

      if (new_operand_shape == operand_shape && reduce->inputs().size() == 1) {
        return absl::OkStatus();
      }

      HloInstruction *canonical_reduce_input =
          new_operand_shape != operand_shape
              ? reduce->parent()->AddInstruction(
                    HloInstruction::CreateBitcast(new_operand_shape, operand))
              : operand;
      canonical_reduce_input->set_metadata(operand->metadata());
      VLOG(5) << "Reduction input: " << canonical_reduce_input->ToString();

      new_reduce_shapes.push_back(new_reduce_shape);
      canonical_reduce_inputs.push_back(canonical_reduce_input);

      if (out_reduce_dimensions.empty()) {
        out_reduce_dimensions = new_reduce_dimensions;
      } else {
        TF_RET_CHECK(out_reduce_dimensions == new_reduce_dimensions);
      }
    }

    Shape new_reduce_shape = ShapeUtil::MakeMaybeTupleShape(new_reduce_shapes);

    std::unique_ptr<HloInstruction> new_reduce = HloInstruction::CreateReduce(
        new_reduce_shape, canonical_reduce_inputs, reduce->init_values(),
        out_reduce_dimensions, reduce->to_apply());
    VLOG(5) << "Generated new reduction: " << new_reduce->ToString();
    const Shape &orig_reduce_shape = reduce->shape();

    if (new_reduce_shape != orig_reduce_shape) {
      HloInstruction *wrapped_reduce =
          reduce->parent()->AddInstruction(std::move(new_reduce));

      if (!new_reduce_shape.IsTuple()) {
        new_reduce =
            HloInstruction::CreateBitcast(reduce->shape(), wrapped_reduce);
      } else {
        // Bitcast each element of the tuple.
        absl::InlinedVector<HloInstruction *, 2> out;
        for (int oidx = 0; oidx < reduce->input_count(); oidx++) {
          HloInstruction *gte = reduce->parent()->AddInstruction(
              HloInstruction::CreateGetTupleElement(wrapped_reduce, oidx));
          out.push_back(
              reduce->parent()->AddInstruction(HloInstruction::CreateBitcast(
                  orig_reduce_shape.tuple_shapes(oidx), gte)));
        }
        new_reduce = HloInstruction::CreateTuple(out);
      }
    }

    VLOG(5) << "Generated output: " << new_reduce->ToString();
    return ReplaceWithNewInstruction(reduce, std::move(new_reduce));
  }
};

absl::StatusOr<bool> ReductionLayoutNormalizer::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      EnforceMinorToMajorReduceOpVisitor().RunOnModule(
                          module, execution_threads));
  return changed;
}

}  // namespace gpu
}  // namespace xla
