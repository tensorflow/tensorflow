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
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
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

// Returns the square root of the input rounded up to the nearest square.
static int64_t SqrtOfRoundUpToSquare(int64_t input) {
  return static_cast<int64_t>(std::ceil(std::sqrt(input)));
}

class ReductionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReductionRewriterVisitor(
      se::CudaComputeCapability cuda_compute_capability)
      : cuda_compute_capability_(cuda_compute_capability) {}

  Status HandleReduce(HloInstruction *hlo) override {
    if (IsMinMaxReduction(hlo)) {
      // TODO(cheshire): Also enable for integers.
      VLOG(1) << "Not performing tree expansion on min/max-reduction: "
              << hlo->ToString() << " since min/max operations are associative";
      return OkStatus();
    }

    if (!IsReductionFromOrToContiguousDimensions(*hlo)) {
      return OkStatus();
    }
    return RewriteReduction(hlo);
  }

 private:
  bool IsMinMaxReduction(HloInstruction *hlo) {
    HloComputation *called = hlo->called_computations()[0];
    if (std::optional<ReductionKind> reduction_kind =
            MatchReductionComputation(called)) {
      return reduction_kind == ReductionKind::MAX ||
             reduction_kind == ReductionKind::MIN;
    }
    return false;
  }

  Status RewriteReduction(HloInstruction *hlo) {
    ReductionDimensions reduction_dimensions =
        GetReductionKindAndContiguousComponents(*hlo);
    std::array<int64_t, 3> reduction_tiling =
        GetReductionTiling(reduction_dimensions, cuda_compute_capability_);
    VLOG(5) << "Input: " << hlo->ToString();
    auto *reduce = Cast<HloReduceInstruction>(hlo);
    absl::Span<int64_t const> input_shape_dims =
        reduce->inputs()[0]->shape().dimensions();
    VLOG(3) << "Input dimensions: " << absl::StrJoin(input_shape_dims, ", ");

    bool reduce_batch_dimension = hlo->dimensions().size() > 1;
    VLOG(3) << "reduce_batch_dimension = " << reduce_batch_dimension;

    std::vector<int64_t> reduced_dimensions = *hlo->mutable_dimensions();
    absl::c_sort(reduced_dimensions);
    CHECK_LE(reduced_dimensions.size(), 2);
    int64_t reduced_input_dimension =
        reduced_dimensions[reduced_dimensions.size() - 1];
    VLOG(3) << "reduced_input_dimension: " << reduced_input_dimension;

    // Case (1): batched dimension does not fit.
    if (reduce_batch_dimension &&
        input_shape_dims[0] > BatchedReductionRaceFreeBound()) {
      VLOG(2) << "Splitting batched dimension reduce into a separate reduction";
      VLOG(1) << "Input: " << hlo->ToString();
      return RewriteBatchDimensionLargerThanTile(reduce, reduction_dimensions,
                                                 reduced_input_dimension);
    }
    bool is_row_reduction = reduction_dimensions.is_row_reduction;

    // Base case: everything fits.
    if (ReductionIsRaceFree(reduction_dimensions, reduction_tiling)) {
      VLOG(3) << "Base case: dimensions fit";
      return OkStatus();
    }

    VLOG(1) << "Input: " << hlo->ToString();
    int64_t reduced_dim_size = input_shape_dims[reduced_input_dimension];
    VLOG(3) << "reduced_dim_size = " << reduced_dim_size;

    // We pad to a nearest square (ceil(sqrt(x)))^2.  Given that:
    //
    // (n + 1)^2 = n^2 + (2n+1)
    //
    // it can be seen that the distance to the nearest square is at most twice
    // the square root of the input number.
    int64_t num_fit = SqrtOfRoundUpToSquare(reduced_dim_size);

    // Pad reduced dimension to the required number of elements.
    bool no_padding_necessary = reduced_dim_size % num_fit == 0;
    using InstructionVector = absl::InlinedVector<HloInstruction *, 2>;
    auto padded = [&]() -> InstructionVector {
      if (no_padding_necessary) {
        return InstructionVector(reduce->inputs().begin(),
                                 reduce->inputs().end());
      }

      int64_t padded_num_elements = num_fit * num_fit;
      PaddingConfig padding_config =
          MakeNoPaddingConfig(input_shape_dims.size());
      padding_config.mutable_dimensions(reduced_input_dimension)
          ->set_edge_padding_high(padded_num_elements - reduced_dim_size);
      std::vector<int64_t> padded_dimensions(input_shape_dims.begin(),
                                             input_shape_dims.end());
      padded_dimensions[reduced_input_dimension] = padded_num_elements;

      absl::InlinedVector<HloInstruction *, 2> out;
      out.reserve(reduce->input_count());
      for (int i = 0; i < reduce->input_count(); i++) {
        HloInstruction *in = reduce->inputs()[i];
        Shape padded_shape =
            ShapeUtil::MakeShape(in->shape().element_type(), padded_dimensions);
        VLOG(3) << "Generated padded shape: " << padded_shape.ToString();
        out.push_back(hlo->parent()->AddInstruction(HloInstruction::CreatePad(
            padded_shape, in, reduce->init_values()[i], padding_config)));
      }
      return out;
    }();

    VLOG(2) << "Generated padding: " << padded[0]->ToString();
    absl::InlinedVector<int64_t, 3> reshaped_dimensions;
    for (int64_t dim_idx = 0; dim_idx < padded[0]->shape().dimensions_size();
         dim_idx++) {
      if (dim_idx == reduced_input_dimension) {
        if (no_padding_necessary) {
          reshaped_dimensions.push_back(reduced_dim_size / num_fit);
        } else {
          reshaped_dimensions.push_back(num_fit);
        }

        reshaped_dimensions.push_back(num_fit);
      } else {
        reshaped_dimensions.push_back(padded[0]->shape().dimensions(dim_idx));
      }
    }

    absl::InlinedVector<int64_t, 3> inner_reduce_dimensions =
        reshaped_dimensions;
    int64_t inner_reduced_dimension = is_row_reduction
                                          ? inner_reduce_dimensions.size() - 1
                                          : reduced_input_dimension;
    VLOG(2) << "inner_reduced_dimension = " << inner_reduced_dimension;
    inner_reduce_dimensions.erase(inner_reduce_dimensions.begin() +
                                  inner_reduced_dimension);
    if (reduce_batch_dimension) {
      inner_reduce_dimensions.erase(inner_reduce_dimensions.begin());
    }
    std::vector<int64_t> dims_to_reduce = {inner_reduced_dimension};
    if (reduce_batch_dimension) {
      dims_to_reduce.push_back(0);
      inner_reduced_dimension -= 1;
    }

    InstructionVector reshaped_padded_inputs;
    absl::InlinedVector<Shape, 2> inner_reduce_shapes;
    for (int i = 0; i < padded.size(); i++) {
      HloInstruction *p = padded[i];
      Shape reshaped_shape =
          ShapeUtil::MakeShape(p->shape().element_type(), reshaped_dimensions);
      HloInstruction *reshaped_padded_input = hlo->parent()->AddInstruction(
          HloInstruction::CreateBitcast(reshaped_shape, p));
      VLOG(2) << "Generated reshape: " << reshaped_padded_input->ToString();
      reshaped_padded_inputs.push_back(reshaped_padded_input);
      Shape inner_reduce_shape = ShapeUtil::MakeShape(p->shape().element_type(),
                                                      inner_reduce_dimensions);
      inner_reduce_shapes.push_back(inner_reduce_shape);
    }

    HloInstruction *inner_reduce =
        hlo->parent()->AddInstruction(HloInstruction::CreateReduce(
            ShapeUtil::MakeMaybeTupleShape(inner_reduce_shapes),
            reshaped_padded_inputs, reduce->init_values(), dims_to_reduce,
            hlo->to_apply()));
    VLOG(1) << "Generated inner reduction: " << inner_reduce->ToString();
    absl::InlinedVector<int64_t, 3> outer_reduce_dimensions =
        inner_reduce_dimensions;
    VLOG(3) << "outer_reduce_dimensions = "
            << absl::StrJoin(outer_reduce_dimensions, ", ");
    int64_t outer_reduced_dimension = is_row_reduction
                                          ? outer_reduce_dimensions.size() - 1
                                          : reduced_input_dimension;

    // Remove reduced dimension.
    outer_reduce_dimensions.erase(outer_reduce_dimensions.begin() +
                                  outer_reduced_dimension);
    std::unique_ptr<HloInstruction> outer_reduce = HloInstruction::CreateReduce(
        hlo->shape(), inner_reduce, reduce->init_values(),
        {outer_reduced_dimension}, hlo->to_apply());

    VLOG(1) << "Generated outer reduction: " << outer_reduce->ToString();
    return ReplaceWithNewInstruction(hlo, std::move(outer_reduce));
  }

  // Rewrites batch dimension reduction into a separate reduce operation.
  Status RewriteBatchDimensionLargerThanTile(
      HloReduceInstruction *hlo,
      const ReductionDimensions &reduction_dimensions,
      int64_t reduced_input_dimension) {
    // TODO(cheshire): this codepath is essentially the exact reverse of what
    // algebraic_simplifier is doing, we need to make sure they don't keep
    // undoing each other.
    CHECK(reduction_dimensions.is_row_reduction);

    absl::InlinedVector<Shape, 2> tuple_shapes;
    for (HloInstruction *input : hlo->inputs()) {
      tuple_shapes.push_back(
          ShapeUtil::DeleteDimension(reduced_input_dimension, input->shape()));
    }

    HloInstruction *inner_reduce =
        hlo->parent()->AddInstruction(HloInstruction::CreateReduce(
            ShapeUtil::MakeMaybeTupleShape(tuple_shapes), hlo->inputs(),
            hlo->init_values(), {reduced_input_dimension}, hlo->to_apply()));

    VLOG(1) << "Inner reduction: " << inner_reduce->ToString();
    std::unique_ptr<HloInstruction> out = HloInstruction::CreateReduce(
        hlo->shape(), inner_reduce, hlo->init_values(), {0}, hlo->to_apply());
    VLOG(1) << "Generated: " << out->ToString();
    return ReplaceWithNewInstruction(hlo, std::move(out));
  }

  se::CudaComputeCapability cuda_compute_capability_;
};

StatusOr<bool> GpuTreeReductionRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  VLOG(5) << "Rewriter input: " << module->ToString();
  TF_ASSIGN_OR_RETURN(bool changed,
                      ReductionRewriterVisitor(cuda_compute_capability_)
                          .RunOnModule(module, execution_threads));
  VLOG(5) << "Rewriter output: " << module->ToString();
  return changed;
}

}  // end namespace gpu
}  // end namespace xla
