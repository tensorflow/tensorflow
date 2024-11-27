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
#include "xla/service/gpu/transforms/tree_reduction_rewriter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

absl::InlinedVector<int64_t, 2> GetSortedReducedDims(
    HloReduceInstruction *reduce) {
  absl::InlinedVector<int64_t, 2> reduced_dims{reduce->dimensions().begin(),
                                               reduce->dimensions().end()};
  absl::c_sort(reduced_dims);
  return reduced_dims;
}

bool IsMinMaxReduction(HloReduceInstruction *reduce) {
  HloComputation *called = &reduce->to_apply()[0];
  if (auto reduction_kind = MatchReductionComputation(called)) {
    return reduction_kind == ReductionKind::MAX ||
           reduction_kind == ReductionKind::MIN;
  }
  return false;
}

}  // namespace

class ReductionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReductionRewriterVisitor(
      const se::DeviceDescription &device_description)
      : device_description_(device_description) {}

  absl::Status HandleReduce(HloInstruction *hlo) override {
    auto *reduce = Cast<HloReduceInstruction>(hlo);
    VLOG(3) << "Reduction instruction: " << reduce->ToString();

    const HloModuleConfig &config = reduce->GetModule()->config();
    if (!MatchReductionForSplit(reduce, config)) {
      return absl::OkStatus();
    }
    ReductionDimensions reduction_dims =
        GetReductionKindAndContiguousComponents(*hlo);
    if (ReductionIsRaceFree(reduction_dims, device_description_)) {
      VLOG(3) << "Base case: dimensions fit";
      return absl::OkStatus();
    }
    auto sorted_dims_to_reduce = GetSortedReducedDims(reduce);
    CHECK_LE(sorted_dims_to_reduce.size(), 2);

    // If the major reduced dimension does not fit, reduce the minor dimension
    // first, then the major.
    if (reduction_dims.is_row_reduction &&
        reduction_dims
                .dimensions[ReductionDimensions::kRowMajorReducedDimension] >
            BatchedReductionRaceFreeBound()) {
      VLOG(2) << "Splitting batched dimension reduce into a separate reduction";
      return RewriteBatchDimensionLargerThanTile(reduce, reduction_dims,
                                                 sorted_dims_to_reduce);
    }
    SplitParams split_params =
        ComputeSplitParams(reduce, reduction_dims, sorted_dims_to_reduce);
    return SplitReductionDimension(reduce, split_params, sorted_dims_to_reduce);
  }

 private:
  bool MatchReductionForSplit(HloReduceInstruction *reduce,
                              const HloModuleConfig &config) {
    // MLIR emitters only support race-free reductions.
    // TODO(jreiffers): Verify performance and implement atomics for reductions
    // if needed.
    if (!IsReductionFromOrToContiguousDimensions(*reduce,
                                                 device_description_)) {
      VLOG(3) << "Is not a reduction from or to contiguous dimensions";
      return false;
    }
    VLOG(3) << "Perform rewrite";
    return true;
  }

  // We observe larger n_div_k can improve tree reduction performance in most of
  // the cases by reducing memory store and the launch overhead of blocks. Swap
  // k and n_div_k if possible.
  bool ShouldSwapInnerAndOuterReducedMinorDimension(uint64_t k1, uint64_t k2,
                                                    uint64_t n,
                                                    int64_t race_free_bound,
                                                    bool is_row_reduction) {
    CHECK_GE(k1, k2);
    // Keep inner reduction as race free.
    if (k1 > race_free_bound) {
      return false;
    }
    // Swapping only affects row reduction vectorization.
    if (is_row_reduction) {
      // Rough conditions for row reduction vectorization, not mean that
      // vectorization will definitely occur.
      bool maybe_vectorized = k2 % 2 == 0 && n % 2 == 0;
      if (maybe_vectorized) {
        // Swap if n_div_k is small enough or k dim can be vectorized also.
        return k2 * 2 < k1 || k1 % 2 == 0;
      }
      // Current reduction emitter only checks reduction input dimensions but
      // not fusion input dimensions. Due to pad and inner reduction always fuse
      // into same computation, it may leads to each thread reads multiple non
      // aligned elements but can not vectorized so that get bad performance.
      // Don't swap If encountered this situation.
      return n % 2 == 0 || k1 % 2 != 0;
    }
    // There exists no specific situation where swapping has no performance gain
    // for column reduction.
    return true;
  }

  // Parameters how to split a dimension `dim` with `k` elements into `k1` x
  // `k2`.
  struct SplitParams {
    int64_t k1;
    int64_t k2;
    int64_t dim;
  };

  // Attempts to find the best way to split a dimension `dim` with `k` elements
  // into `k1` x `k2`.
  SplitParams ComputeSplitParams(
      HloReduceInstruction *reduce, const ReductionDimensions &reduction_dims,
      absl::Span<const int64_t> sorted_dims_to_reduce) {
    absl::Span<int64_t const> input_shape_dims =
        reduce->inputs()[0]->shape().dimensions();

    int64_t reduced_dim = sorted_dims_to_reduce.back();
    int64_t reduced_dim_size = input_shape_dims[reduced_dim];
    VLOG(3) << "reduced dim size = " << reduced_dim_size;

    // We will do this reduction in two stages.  The first will reduce from k
    // elements to k1 elements in the reduction dimension.  The second will
    // reduce further, from k2 to 1 element.
    //
    // We do this by splitting the input shape [a, k, b] into [a, k1, k2, b].
    //
    // We want to choose k1 to be roughly equal to sqrt(k) so that we process
    // "most of" the reduction in the first step. But it is also important that
    // we choose a value of k1 with the least amount of padding we need to add
    // to n to make it divisible by k1. We search for the best value of k2
    // between sqrt(k)/2 and sqrt(k). If there are several possible values for
    // k2 that result in the minimum amount of padding, we also want k2 to
    // be a power of 2, so that the GPU kernel doesn't spend all its time doing
    // slow integer divmods to compute indices into the shape [a,k1,k2,b].
    // Note that by searching in the range between sqrt(k)/2 and sqrt(k), we
    // will have a power of 2 in that range.
    uint64_t k2 =
        static_cast<uint64_t>(std::floor(std::sqrt(reduced_dim_size)));
    int64_t race_free_bound =
        ReductionDimensionRaceFreeBound(reduction_dims, device_description_);
    if (k2 > race_free_bound) {
      // This means we need more than one split. It is best to limit the n/k
      // dimension to the maximum size that doesn't require further splitting.
      // Otherwise we might choose a rather small reduce dimension size for the
      // first step (in the worst case, sqrt(race_free_bound + 1)).
      k2 = race_free_bound;
    }
    uint64_t minimum_padding = (k2 - reduced_dim_size % k2) % k2;
    uint64_t best_k1 = (reduced_dim_size + minimum_padding) / k2;
    for (uint64_t i = k2 - 1; i > k2 / 2; --i) {
      uint64_t padding = (i - reduced_dim_size % i) % i;
      if (padding < minimum_padding ||
          (padding == minimum_padding && absl::has_single_bit(i))) {
        minimum_padding = padding;
        best_k1 = (reduced_dim_size + padding) / i;
      }
    }
    uint64_t padded_k = reduced_dim_size + minimum_padding;

    // We get the best {k_1, k_2} pair by the size of padding and whether
    // index computation is fast. But we ignored the overhead of memory
    // read/write and blocks launch, which are also important for kernel
    // performance. It is obvious that the swapped {k1, k2} pairs has same
    // padding size and consumption of index computation as the original. So we
    // only need to compare the memory read/write and blocks launch to choose
    // the better one of them.
    uint64_t best_k2 = padded_k / best_k1;
    if (ShouldSwapInnerAndOuterReducedMinorDimension(
            best_k1, best_k2, reduced_dim_size, race_free_bound,
            reduction_dims.is_row_reduction)) {
      std::swap(best_k1, best_k2);
    }
    return SplitParams{static_cast<int64_t>(best_k1),
                       static_cast<int64_t>(best_k2), reduced_dim};
  }

  // Replaces the original reduce with pad->reshape>inner_reduce->outer_reduce.
  // * 1. pads split dimension of the inputs to k1 * k2 if necessary.
  // * 2. reshapes split dimension of the padded inputs into [k1, k2].
  // * 3. inner reduction reduces the dims specified in the original reduction.
  //      Instead of reducing the split dimension, reduces K2.
  // * 4. outer_reduction reduces K1 only.
  absl::Status SplitReductionDimension(
      HloReduceInstruction *reduce, const SplitParams &split_params,
      absl::Span<const int64_t> sorted_dims_to_reduce) {
    absl::Span<int64_t const> reduce_input_dims =
        reduce->inputs()[0]->shape().dimensions();
    int64_t split_dim_size = reduce_input_dims[split_params.dim];
    VLOG(2) << "dimension to split = " << split_params.dim << " with "
            << split_dim_size << " elements into " << split_params.k1 << " by "
            << split_params.k2;

    // Pad 'k' to 'k1 * k2' if necessary.
    HloInstruction::InstructionVector padded_inputs(reduce->inputs().begin(),
                                                    reduce->inputs().end());
    auto padded_size = split_params.k1 * split_params.k2;
    absl::InlinedVector<int64_t, 3> padded_dimensions(reduce_input_dims.begin(),
                                                      reduce_input_dims.end());
    if (split_dim_size != padded_size) {
      padded_dimensions[split_params.dim] = padded_size;
      PaddingConfig padding_config =
          MakeNoPaddingConfig(reduce_input_dims.size());
      padding_config.mutable_dimensions(split_params.dim)
          ->set_edge_padding_high(padded_size - split_dim_size);

      for (int input_idx = 0; input_idx < padded_inputs.size(); ++input_idx) {
        auto &reduction_input = padded_inputs[input_idx];
        Shape padded_shape = ShapeUtil::MakeShape(
            reduction_input->shape().element_type(), padded_dimensions);
        VLOG(2) << "Generated padded shape: " << padded_shape.ToString();
        reduction_input = reduce->parent()->AddInstruction(
            HloInstruction::CreatePad(padded_shape, reduction_input,
                                      reduce->init_values()[input_idx],
                                      padding_config),
            &reduction_input->metadata());
      }
    }

    // Compute output type of reshape that expands the split dimension into
    // [k1, k2].
    absl::InlinedVector<int64_t, 3> reshaped_dimensions;
    int64_t input_rank = reduce_input_dims.size();
    for (int64_t dim_idx = 0; dim_idx < input_rank; dim_idx++) {
      if (dim_idx == split_params.dim) {
        reshaped_dimensions.push_back(split_params.k1);
        reshaped_dimensions.push_back(split_params.k2);
      } else {
        reshaped_dimensions.push_back(padded_dimensions[dim_idx]);
      }
    }

    // Compute dimensions to reduce for inner reduction.
    absl::InlinedVector<int64_t, 2> inner_reduce_dims(
        sorted_dims_to_reduce.begin(), sorted_dims_to_reduce.end());
    auto split_dim_it = std::find(inner_reduce_dims.begin(),
                                  inner_reduce_dims.end(), split_params.dim);
    *split_dim_it += 1;

    // Compute dimension to reduce for outer reduction.
    absl::InlinedVector<int64_t, 1> outer_reduce_dims{
        split_params.dim -
        std::distance(inner_reduce_dims.begin(), split_dim_it)};

    // Compute output shape of the inner reduction.
    absl::InlinedVector<int64_t, 3> inner_reduce_shape =
        RemoveElements(inner_reduce_dims, reshaped_dimensions);

    // Reshape the split dimensions of the padded inputs into [k1, k2].
    HloInstruction::InstructionVector reshaped_padded_inputs;
    absl::InlinedVector<Shape, 2> inner_reduce_shapes;
    for (HloInstruction *padded_input : padded_inputs) {
      Shape reshaped_shape = ShapeUtil::MakeShape(
          padded_input->shape().element_type(), reshaped_dimensions);
      HloInstruction *reshaped_padded_input = reduce->parent()->AddInstruction(
          HloInstruction::CreateBitcast(reshaped_shape, padded_input),
          &padded_input->metadata());
      VLOG(2) << "Generated reshape: " << reshaped_padded_input->ToString();
      reshaped_padded_inputs.push_back(reshaped_padded_input);
      inner_reduce_shapes.push_back(ShapeUtil::MakeShape(
          padded_input->shape().element_type(), inner_reduce_shape));
    }

    // Inner reduce that reduces [k1, k2] to [k1].
    HloInstruction *inner_reduce = reduce->parent()->AddInstruction(
        HloInstruction::CreateReduce(
            ShapeUtil::MakeMaybeTupleShape(inner_reduce_shapes),
            reshaped_padded_inputs, reduce->init_values(), inner_reduce_dims,
            reduce->to_apply()),
        &reduce->metadata());
    VLOG(1) << "Generated inner reduction: " << inner_reduce->ToString();

    // Outer reduce that reduces [k2].
    std::unique_ptr<HloInstruction> outer_reduce = HloInstruction::CreateReduce(
        reduce->shape(), inner_reduce, reduce->init_values(), outer_reduce_dims,
        reduce->to_apply());

    VLOG(1) << "Generated outer reduction: " << outer_reduce->ToString();
    return ReplaceWithNewInstruction(reduce, std::move(outer_reduce));
  }

  // Rewrites batch dimension reduction into a separate reduce operation.
  absl::Status RewriteBatchDimensionLargerThanTile(
      HloReduceInstruction *hlo,
      const ReductionDimensions &reduction_dimensions,
      absl::Span<const int64_t> sorted_dims_to_reduce) {
    // TODO(cheshire): this codepath is essentially the exact reverse of what
    // algebraic_simplifier is doing, we need to make sure they don't keep
    // undoing each other.
    CHECK(reduction_dimensions.is_row_reduction);

    absl::InlinedVector<Shape, 2> tuple_shapes;
    int64_t minor_reduction_dim = sorted_dims_to_reduce.back();
    for (HloInstruction *input : hlo->inputs()) {
      tuple_shapes.push_back(
          ShapeUtil::DeleteDimension(minor_reduction_dim, input->shape()));
    }

    HloInstruction *inner_reduce =
        hlo->parent()->AddInstruction(HloInstruction::CreateReduce(
            ShapeUtil::MakeMaybeTupleShape(tuple_shapes), hlo->inputs(),
            hlo->init_values(), {minor_reduction_dim}, hlo->to_apply()));

    VLOG(1) << "Inner reduction: " << inner_reduce->ToString();
    std::unique_ptr<HloInstruction> out = HloInstruction::CreateReduce(
        hlo->shape(), inner_reduce, hlo->init_values(), {0}, hlo->to_apply());
    VLOG(1) << "Generated: " << out->ToString();
    return ReplaceWithNewInstruction(hlo, std::move(out));
  }

  const se::DeviceDescription &device_description_;
};

absl::StatusOr<bool> TreeReductionRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  VLOG(5) << "Rewriter input: " << module->ToString();
  TF_ASSIGN_OR_RETURN(bool changed,
                      ReductionRewriterVisitor(device_description_)
                          .RunOnModule(module, execution_threads));
  VLOG(5) << "Rewriter output: " << module->ToString();
  return changed;
}

}  // end namespace gpu
}  // end namespace xla
