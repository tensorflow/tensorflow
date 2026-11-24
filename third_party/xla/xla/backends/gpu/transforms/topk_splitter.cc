/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/topk_splitter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

constexpr size_t kRequiredAlignment = 1024;
constexpr size_t kMaximumBatchSize = 1024;

class TopkSplitterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit TopkSplitterVisitor(size_t split_threshold)
      : split_threshold_(split_threshold) {}

  absl::Status HandleCustomCall(HloInstruction* inst) override {
    HloCustomCallInstruction* topk = DynCast<HloCustomCallInstruction>(inst);
    if (topk == nullptr || topk->custom_call_target() != "TopK") {
      return absl::OkStatus();
    }
    HloComputation* comp = inst->parent();
    Shape data_shape = topk->operand(0)->shape();
    bool has_batch = data_shape.dimensions().size() == 2;
    // TODO(doak): Support multiple batches.
    if (has_batch && data_shape.dimensions(0) != 1) {
      return absl::OkStatus();
    }
    size_t n = data_shape.dimensions(has_batch ? 1 : 0);
    int64_t k = topk->shape().tuple_shapes(0).dimensions(has_batch ? 1 : 0);
    // If K approaches N, splitting the input will not be beneficial anymore.
    if (k > sqrt(n)) {
      return absl::OkStatus();
    }
    // TODO(doak): Relax this alignment requirement.
    if (n % kRequiredAlignment != 0) {
      return absl::OkStatus();
    }
    if (n <= split_threshold_) {
      return absl::OkStatus();
    }
    TF_RET_CHECK(topk->called_computations().size() == 1)
        << "Expected TopK custom call to have 1 called computation, got "
        << topk->called_computations().size();
    ABSL_ASSIGN_OR_RETURN(
        HloComputation * sort_comparator,
        GetOrCreateFourParamComparator(topk->to_apply(), comp->parent()));
    int new_batch =
        std::min(absl::bit_floor(n / split_threshold_), kMaximumBatchSize);
    int new_n = n / new_batch;
    // Split the input into B batches and compute TopK over the batched arrays.
    Shape split_input_shape =
        ShapeUtil::MakeShape(data_shape.element_type(), {new_batch, new_n});
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * reshaped,
        MakeReshapeHlo(split_input_shape, topk->mutable_operand(0)));
    Shape batch_topk_shape = ShapeUtil::MakeTupleShape(
        {ShapeUtil::MakeShape(data_shape.element_type(), {new_batch, k}),
         ShapeUtil::MakeShape(S32, {new_batch, k})});
    HloInstruction* batch_topk =
        comp->AddInstruction(HloInstruction::CreateCustomCall(
            batch_topk_shape, {reshaped}, sort_comparator, "TopK",
            topk->raw_backend_config_string()));
    // Fix indices, adding j*split_N to the j-th batch of indices.
    ABSL_ASSIGN_OR_RETURN(HloInstruction * indices,
                     MakeGetTupleElementHlo(batch_topk, 1));
    ABSL_ASSIGN_OR_RETURN(HloInstruction * values,
                     MakeGetTupleElementHlo(batch_topk, 0));
    Shape iota_shape = ShapeUtil::MakeShape(S32, {new_batch});
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * fix,
        MakeBinaryHlo(
            HloOpcode::kMultiply, MakeIotaHlo(comp, iota_shape, 0),
            MakeBroadcastHlo(MakeR0ConstantHlo<int32_t>(comp, new_n),
                             /*broadcast_dimensions=*/{}, iota_shape)));
    ABSL_ASSIGN_OR_RETURN(
        indices, MakeBinaryHlo(HloOpcode::kAdd, indices,
                               MakeBroadcastHlo(fix, {0}, indices->shape())));
    // With the indices restored, compute a final top-k. Since this topk uses
    // arbitrary indices, we need to use sort+slice.
    Shape linear_index_shape = ShapeUtil::MakeShape(S32, {k * new_batch});
    Shape linear_shape = ShapeUtil::ChangeElementType(
        linear_index_shape, data_shape.element_type());
    Shape linear_sort_shape =
        ShapeUtil::MakeTupleShape({linear_shape, linear_index_shape});
    // Assuming the outputs of the TopK above are stably sorted, using a stable
    // sort here is enough to guarantee global stable sorting:
    //  - Within a blocks elements are stably sorted by TopK.
    //  - Since blocks are organized linearly from smallest to largest, the
    //    index used on the stable sort below will also respect block ordering.
    HloInstruction* aggregated_sort =
        comp->AddInstruction(HloInstruction::CreateSort(
            linear_sort_shape, 0,
            {*MakeReshapeHlo(linear_shape, values),
             *MakeReshapeHlo(linear_index_shape, indices)},
            sort_comparator, /*is_stable=*/true));
    auto slice_tuple = [&](HloInstruction* sort, const size_t index) {
      return *MakeReshapeHlo(
          topk->shape().tuple_shapes(index),
          *MakeSliceHlo(*MakeGetTupleElementHlo(sort, index), {0}, {k}, {1}));
    };
    return ReplaceInstruction(topk,
                              comp->AddInstruction(HloInstruction::CreateTuple({
                                  slice_tuple(aggregated_sort, 0),
                                  slice_tuple(aggregated_sort, 1),
                              })));
  }

 private:
  // Expands a 2-parameter value-only comparator into a 4-parameter comparator
  // by appending two unused S32 scalar index parameters so it matches the arity
  // required by two-operand (value, index) sort and TopK instructions.
  absl::StatusOr<HloComputation*> GetOrCreateFourParamComparator(
      HloComputation* comparator, HloModule* module) {
    if (comparator->num_parameters() == 4) {
      return comparator;
    }
    TF_RET_CHECK(comparator->num_parameters() == 2)
        << "Expected TopK comparator to have 2 or 4 parameters, got "
        << comparator->num_parameters();
    HloComputation*& expanded = expanded_comparators_[comparator];
    if (expanded != nullptr) {
      return expanded;
    }
    Shape scalar_s32 = ShapeUtil::MakeScalarShape(S32);
    std::unique_ptr<HloInstruction> lhs_idx =
        HloInstruction::CreateParameter(2, scalar_s32, "p.1.lhs");
    std::unique_ptr<HloInstruction> rhs_idx =
        HloInstruction::CreateParameter(3, scalar_s32, "p.1.rhs");
    std::vector<const HloInstruction*> extra_parameters = {lhs_idx.get(),
                                                           rhs_idx.get()};
    expanded = module->AddEmbeddedComputation(comparator->CloneWithReplacements(
        /*replacements=*/nullptr, extra_parameters));
    return expanded;
  }

  size_t split_threshold_;
  absl::flat_hash_map<const HloComputation*, HloComputation*>
      expanded_comparators_;
};

}  // namespace

absl::StatusOr<bool> TopKSplitter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return TopkSplitterVisitor(split_threshold_)
      .RunOnModule(module, execution_threads);
}

}  // namespace gpu
}  // namespace xla
