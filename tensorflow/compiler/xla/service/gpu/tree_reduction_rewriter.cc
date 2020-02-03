/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/gpu/tree_reduction_rewriter.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {

class ReductionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReductionRewriterVisitor() {}

  Status HandleReduce(HloInstruction *hlo) override {
    if (!hlo->shape().IsArray()) {
      // TODO(b/130802338): handle variadic reduction.
      return Status::OK();
    }

    if (!IsReductionFromOrToContiguousDimensions(*hlo)) {
      return Status::OK();
    }
    return RewriteReduction(hlo);
  }

 private:
  Status RewriteReduction(HloInstruction *hlo) {
    ReductionDimensions reduction_dimensions =
        GetReductionKindAndContiguousComponents(*hlo);
    VLOG(3) << "Input: " << hlo->ToString();

    HloInstruction *input = hlo->mutable_operand(0);
    HloInstruction *initial_value = hlo->mutable_operand(1);
    Shape input_shape = input->shape();
    VLOG(3) << "Input shape: " << input_shape.ToString();

    std::array<int64, 3> reduction_tiling =
        GetReductionTiling(reduction_dimensions);

    int64 batched_atomic_free_bound = reduction_tiling[0];
    bool reduce_batch_dimension = hlo->dimensions().size() > 1;
    VLOG(3) << "reduce_batch_dimension = " << reduce_batch_dimension;
    VLOG(3) << "batched atomic free: " << batched_atomic_free_bound;

    std::vector<int64> reduced_dimensions = hlo->dimensions();
    absl::c_sort(reduced_dimensions);
    CHECK_LE(reduced_dimensions.size(), 2);
    int64 reduced_input_dimension =
        reduced_dimensions[reduced_dimensions.size() - 1];
    VLOG(3) << "reduced_input_dimension: " << reduced_input_dimension;

    // Case (1): batched dimension does not fit.
    if (reduce_batch_dimension &&
        input_shape.dimensions(0) > batched_atomic_free_bound) {
      VLOG(1) << "Splitting batched dimension reduce into a separate reduction";
      return RewriteBatchDimensionLargerThanTile(hlo, reduction_dimensions,
                                                 reduced_input_dimension,
                                                 input_shape, input);
    }

    int64 atomic_free_bound = reduction_dimensions.is_row_reduction
                                  ? reduction_tiling[2] * kWarpSize
                                  : reduction_tiling[1];
    VLOG(3) << "atomic_free_bound: " << atomic_free_bound;

    // Base case: everything fits.
    if (input_shape.dimensions(reduced_input_dimension) <= atomic_free_bound) {
      VLOG(3) << "Base case: dimensions fit";
      return Status::OK();
    }

    int64 reduced_dim_size = input_shape.dimensions(reduced_input_dimension);
    VLOG(3) << "reduced_dim_size = " << reduced_dim_size;
    int64 num_fit = CeilOfRatio(reduced_dim_size, atomic_free_bound);

    // Pad reduced dimension to the required number of elements.
    HloInstruction *padded = [&] {
      if (reduced_dim_size % atomic_free_bound != 0) {
        int64 padded_num_elements = num_fit * atomic_free_bound;
        PaddingConfig padding_config = MakeNoPaddingConfig(input_shape.rank());
        padding_config.mutable_dimensions(reduced_input_dimension)
            ->set_edge_padding_high(padded_num_elements - reduced_dim_size);
        std::vector<int64> padded_dimensions(input_shape.dimensions().begin(),
                                             input_shape.dimensions().end());
        padded_dimensions[reduced_input_dimension] = padded_num_elements;
        Shape padded_shape =
            ShapeUtil::MakeShape(input_shape.element_type(), padded_dimensions);
        VLOG(3) << "Generated padded shape: " << padded_shape.ToString();
        return hlo->parent()->AddInstruction(HloInstruction::CreatePad(
            padded_shape, input, initial_value, padding_config));
      }
      return input;
    }();

    VLOG(1) << "Generated padding: " << padded->ToString();
    std::vector<int64> reshaped_dimensions;
    for (int64 dim_idx = 0; dim_idx < padded->shape().dimensions_size();
         dim_idx++) {
      if (dim_idx == reduced_input_dimension) {
        reshaped_dimensions.push_back(num_fit);
        reshaped_dimensions.push_back(atomic_free_bound);
      } else {
        reshaped_dimensions.push_back(padded->shape().dimensions(dim_idx));
      }
    }

    Shape reshaped_shape =
        ShapeUtil::MakeShape(input_shape.element_type(), reshaped_dimensions);
    HloInstruction *reshaped_padded_input = hlo->parent()->AddInstruction(
        HloInstruction::CreateBitcast(reshaped_shape, padded));
    VLOG(1) << "Generated reshape: " << reshaped_padded_input->ToString();

    std::vector<int64> inner_reduce_dimensions = reshaped_dimensions;
    inner_reduce_dimensions.erase(inner_reduce_dimensions.begin() +
                                  reduced_input_dimension);
    if (reduce_batch_dimension) {
      inner_reduce_dimensions.erase(inner_reduce_dimensions.begin());
    }

    Shape inner_reduce_shape = ShapeUtil::MakeShape(input_shape.element_type(),
                                                    inner_reduce_dimensions);
    std::vector<int64> dims_to_reduce = {reduced_input_dimension};

    int64 reduced_inner_dimension = reduced_input_dimension;
    if (reduce_batch_dimension) {
      dims_to_reduce.push_back(0);
      reduced_inner_dimension -= 1;
    }

    HloInstruction *inner_reduce =
        hlo->parent()->AddInstruction(HloInstruction::CreateReduce(
            inner_reduce_shape, reshaped_padded_input, initial_value,
            dims_to_reduce, hlo->to_apply()));
    VLOG(1) << "Generated inner reduction: " << inner_reduce->ToString();

    std::vector<int64> outer_reduce_dimensions = inner_reduce_dimensions;
    VLOG(3) << "outer_reduce_dimensions = "
            << absl::StrJoin(outer_reduce_dimensions, ", ");
    VLOG(3) << "reduced_inner_dimension = " << reduced_inner_dimension;

    // Remove reduced dimension.
    outer_reduce_dimensions.erase(outer_reduce_dimensions.begin() +
                                  reduced_inner_dimension);
    Shape outer_reduce_shape = ShapeUtil::MakeShape(input_shape.element_type(),
                                                    outer_reduce_dimensions);
    std::unique_ptr<HloInstruction> outer_reduce = HloInstruction::CreateReduce(
        outer_reduce_shape, inner_reduce, initial_value,
        {reduced_inner_dimension}, hlo->to_apply());

    VLOG(1) << "Generated outer reduction: " << outer_reduce->ToString();
    return ReplaceWithNewInstruction(hlo, std::move(outer_reduce));
  }

  // Rewrites batch dimension reduction into a separate reduce operation.
  Status RewriteBatchDimensionLargerThanTile(
      HloInstruction *hlo, const ReductionDimensions &reduction_dimensions,
      int64 reduced_input_dimension, const Shape &input_shape,
      HloInstruction *input) {
    // TODO(cheshire): this codepath is essentially the exact reverse of what
    // algebraic_simplifier is doing, we need to make sure they don't keep
    // undoing each other.
    CHECK(reduction_dimensions.is_row_reduction);

    Shape inner_reduce_shape =
        ShapeUtil::DeleteDimension(reduced_input_dimension, input_shape);

    HloInstruction *inner_reduce =
        hlo->parent()->AddInstruction(HloInstruction::CreateReduce(
            inner_reduce_shape, input, hlo->mutable_operand(1),
            {reduced_input_dimension}, hlo->to_apply()));
    VLOG(1) << "Inner reduction: " << inner_reduce->ToString();
    std::unique_ptr<HloInstruction> out = HloInstruction::CreateReduce(
        hlo->shape(), inner_reduce, hlo->mutable_operand(1), {0},
        hlo->to_apply());
    VLOG(1) << "Generated: " << out->ToString();
    return ReplaceWithNewInstruction(hlo, std::move(out));
  }
};

StatusOr<bool> GpuTreeReductionRewriter::Run(HloModule *module) {
  VLOG(5) << "Rewriter input: " << module->ToString();
  TF_ASSIGN_OR_RETURN(bool changed,
                      ReductionRewriterVisitor().RunOnModule(module));
  VLOG(5) << "Rewriter output: " << module->ToString();
  return changed;
}

}  // end namespace gpu
}  // end namespace xla
