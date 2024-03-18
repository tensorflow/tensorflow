/* Copyright 2020 The OpenXLA Authors.

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
#include "xla/service/gpu/tree_reduction_rewriter.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

class ReductionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReductionRewriterVisitor(se::GpuComputeCapability gpu_version)
      : gpu_version_(gpu_version) {}

  absl::Status HandleReduce(HloInstruction *hlo) override {
    if (IsMinMaxReduction(hlo)) {
      // TODO(cheshire): Also enable for integers.
      VLOG(1) << "Not performing tree expansion on min/max-reduction: "
              << hlo->ToString() << " since min/max operations are associative";
      return absl::OkStatus();
    }

    if (!IsReductionFromOrToContiguousDimensions(*hlo)) {
      return absl::OkStatus();
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

  absl::Status RewriteReduction(HloInstruction *hlo) {
    ReductionDimensions reduction_dimensions =
        GetReductionKindAndContiguousComponents(*hlo);
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
    if (ReductionIsRaceFree(hlo->GetModule()->config(), reduction_dimensions)) {
      VLOG(3) << "Base case: dimensions fit";
      return absl::OkStatus();
    }

    VLOG(1) << "Input: " << hlo->ToString();
    int64_t n = input_shape_dims[reduced_input_dimension];
    VLOG(3) << "n = " << n;

    // We will do this reduction in two stages.  The first will reduce from n
    // elements to k elements in the reduction dimension.  The second will
    // reduce further, from k to 1 element.
    //
    // We do this by splitting the input shape [a, n, b] into [a, k, n / k, b].
    //
    // We want to choose k to be roughly equal to sqrt(n) so that we process
    // "most of" the reduction in the first step. But it is also important that
    // we choose a value of k with the least amount of padding we need to add to
    // n to make it divisible by k. We search for the best value of n / k
    // between sqrt(n)/2 and sqrt(n). If there are several possible values for
    // n / k that result in the minimum amount of padding, we also want n / k to
    // be a power of 2, so that the GPU kernel doesn't spend all its time doing
    // slow integer divmods to compute indices into the shape [a,k,n/k,b].
    // Note that by searching in the range between sqrt(n)/2 and sqrt(n), we
    // will have a power of 2 in that range.
    uint64_t n_div_k = static_cast<uint64_t>(std::floor(std::sqrt(n)));
    int64_t race_free_bound = ReductionDimensionRaceFreeBound(
        hlo->GetModule()->config(), reduction_dimensions);
    if (n_div_k > race_free_bound) {
      // This means we need more than one split. It is best to limit the n/k
      // dimension to the maximum size that doesn't require further splitting.
      // Otherwise we might choose a rather small reduce dimension size for the
      // first step (in the worst case, sqrt(race_free_bound + 1)).
      n_div_k = race_free_bound;
    }
    uint64_t minimum_padding = (n_div_k - n % n_div_k) % n_div_k;
    uint64_t best_k = (n + minimum_padding) / n_div_k;
    for (uint64_t i = n_div_k - 1; i > n_div_k / 2; --i) {
      uint64_t padding = (i - n % i) % i;
      if (padding < minimum_padding ||
          (padding == minimum_padding && absl::has_single_bit(i))) {
        minimum_padding = padding;
        best_k = (n + padding) / i;
      }
    }
    uint64_t padded_n = n + minimum_padding;

    // Pad reduced dimension to the required number of elements.
    bool no_padding_necessary = n == padded_n;
    using InstructionVector = absl::InlinedVector<HloInstruction *, 2>;
    auto padded = [&]() -> InstructionVector {
      if (no_padding_necessary) {
        return InstructionVector(reduce->inputs().begin(),
                                 reduce->inputs().end());
      }

      PaddingConfig padding_config =
          MakeNoPaddingConfig(input_shape_dims.size());
      padding_config.mutable_dimensions(reduced_input_dimension)
          ->set_edge_padding_high(padded_n - n);
      std::vector<int64_t> padded_dimensions(input_shape_dims.begin(),
                                             input_shape_dims.end());
      padded_dimensions[reduced_input_dimension] = padded_n;

      absl::InlinedVector<HloInstruction *, 2> out;
      out.reserve(reduce->input_count());
      for (int i = 0; i < reduce->input_count(); i++) {
        HloInstruction *in = reduce->inputs()[i];
        Shape padded_shape =
            ShapeUtil::MakeShape(in->shape().element_type(), padded_dimensions);
        VLOG(3) << "Generated padded shape: " << padded_shape.ToString();
        out.push_back(hlo->parent()->AddInstruction(
            HloInstruction::CreatePad(padded_shape, in,
                                      reduce->init_values()[i], padding_config),
            &in->metadata()));
      }
      return out;
    }();

    VLOG(2) << "Generated padding: " << padded[0]->ToString();
    absl::InlinedVector<int64_t, 3> reshaped_dimensions;
    for (int64_t dim_idx = 0; dim_idx < padded[0]->shape().dimensions_size();
         dim_idx++) {
      if (dim_idx == reduced_input_dimension) {
        reshaped_dimensions.push_back(best_k);
        reshaped_dimensions.push_back(padded_n / best_k);
      } else {
        reshaped_dimensions.push_back(padded[0]->shape().dimensions(dim_idx));
      }
    }

    absl::InlinedVector<int64_t, 3> inner_reduce_dimensions =
        reshaped_dimensions;
    // We split reduced_input_dimension into two new dims.  We have the choice
    // of reducing along either of them.  We choose to reduce along the second,
    // more-minor dimension, because this should use the GPU caches better.
    int64_t inner_reduced_dimension = is_row_reduction
                                          ? inner_reduce_dimensions.size() - 1
                                          : reduced_input_dimension + 1;
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
          HloInstruction::CreateBitcast(reshaped_shape, p), &p->metadata());
      VLOG(2) << "Generated reshape: " << reshaped_padded_input->ToString();
      reshaped_padded_inputs.push_back(reshaped_padded_input);
      Shape inner_reduce_shape = ShapeUtil::MakeShape(p->shape().element_type(),
                                                      inner_reduce_dimensions);
      inner_reduce_shapes.push_back(inner_reduce_shape);
    }

    HloInstruction *inner_reduce = hlo->parent()->AddInstruction(
        HloInstruction::CreateReduce(
            ShapeUtil::MakeMaybeTupleShape(inner_reduce_shapes),
            reshaped_padded_inputs, reduce->init_values(), dims_to_reduce,
            hlo->to_apply()),
        &reduce->metadata());
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
  absl::Status RewriteBatchDimensionLargerThanTile(
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

  se::GpuComputeCapability gpu_version_;
};

absl::StatusOr<bool> GpuTreeReductionRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  VLOG(5) << "Rewriter input: " << module->ToString();
  TF_ASSIGN_OR_RETURN(bool changed,
                      ReductionRewriterVisitor(gpu_version_)
                          .RunOnModule(module, execution_threads));
  VLOG(5) << "Rewriter output: " << module->ToString();
  return changed;
}

}  // end namespace gpu
}  // end namespace xla
