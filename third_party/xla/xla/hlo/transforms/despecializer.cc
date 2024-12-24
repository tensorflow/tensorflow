/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/transforms/despecializer.h"

#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/transforms/defuser.h"
#include "xla/hlo/transforms/simplifiers/float_normalization.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/hlo/transforms/simplifiers/sub_byte_normalization.h"
#include "xla/xla_data.pb.h"

namespace xla {

Despecializer::Despecializer() : pipeline_("despecializer") {
  // TODO(b/70588125): Also deal with window reversal in a fast way.
  pipeline_.AddPass<HloDescheduler>();
  pipeline_.AddPass<ControlDepRemover>();
  pipeline_.AddPass<Defuser>();
  pipeline_.AddPass<BFloat16MixedPrecisionRemoval>();
  pipeline_.AddPass<SubByteNormalization>(
      SubByteNormalization::REMOVE_ELEMENT_SIZE);
}

void Despecializer::AddAssumeGatherIndicesInBoundRewriteToCopy() {
  pipeline_.AddPass<AssumeGatherIndicesInBoundRewriteToCopy>();
}

void Despecializer::AddReduceWindowToReduceBroadcastDeconstruct() {
  pipeline_.AddPass<DeconstructReduceWindowToReduceBroadcast>();
}

absl::StatusOr<bool> Despecializer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return pipeline_.Run(module, execution_threads);
}

// AssumeGatherIndicesInBoundRewriteToCopy is needed to handle the
// "AssumeGatherIndicesInBound" custom-call in a gather fusion.
// "AssumeGatherIndicesInBound" custom-call is a
// no-op that allows the compiler to optimize a gather fusion lowering. From a
// reference platform perspective, i.e., for testing, this custom-call should be
// a copy since no optimizations are performed and runtime is not the criterion
// while obtaining reference results.
absl::StatusOr<bool> AssumeGatherIndicesInBoundRewriteToCopy::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> candidates;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->IsCustomCall("AssumeGatherIndicesInBound")) {
        candidates.push_back(instruction);
      }
    }
  }
  for (HloInstruction* gather_indices : candidates) {
    auto computation = gather_indices->parent();
    auto copy = computation->AddInstruction(
        HloInstruction::CreateUnary(gather_indices->shape(), HloOpcode::kCopy,
                                    gather_indices->mutable_operand(0)));
    TF_CHECK_OK(computation->ReplaceInstruction(gather_indices, copy));
  }
  return !candidates.empty();
}

absl::StatusOr<bool> DeconstructReduceWindowToReduceBroadcast::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  // Candidate ReduceWindows are those that reduce only one dimension of the
  // input tensor to a singleton and subsequently broadcast it out to the
  // dimension span. The below structure holds such candidate reduce-windows
  // and the dimension that is reduce_broadcasted.
  std::vector<std::pair<HloInstruction*, int64_t>> candidate_rw;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kReduceWindow) {
        continue;
      }
      auto* reduce_window = CastOrNull<HloReduceWindowInstruction>(instruction);
      if (reduce_window == nullptr) {
        continue;
      }
      if (reduce_window->operand(0)->shape() != reduce_window->shape()) {
        continue;
      }
      const Window& window = reduce_window->window();
      int64_t num_stride_dilations = absl::c_count_if(
          window.dimensions(), [](const WindowDimension& win_dim) {
            return (
                win_dim.stride() != 1 || win_dim.window_reversal() == true ||
                win_dim.window_dilation() != 1 || win_dim.base_dilation() != 1);
          });
      if (num_stride_dilations != 0) {
        continue;
      }
      // 1) Obtain the Dimensions being reduced.
      int64_t num_dimensions_reduced = absl::c_count_if(
          window.dimensions(),
          [](const WindowDimension& win_dim) { return (win_dim.size() != 1); });
      // 2) Ignore reduce-windows performing multi-dim reductions.
      if (num_dimensions_reduced != 1) {
        continue;
      }
      auto reduce_dim = absl::c_find_if(
          window.dimensions(),
          [](const WindowDimension& win_dim) { return (win_dim.size() != 1); });
      if (reduce_dim == window.dimensions().end()) {
        continue;
      }
      // 3) Find the logical dimension index of the single reduced dimension.
      int64_t reduce_dim_index =
          std::distance(window.dimensions().begin(), reduce_dim);

      // 4) Check if this dimension undergoes a full dimension reduce and then
      // a broadcast back to the full span.
      auto input_dim_size =
          reduce_window->operand(0)->shape().dimensions(reduce_dim_index);
      if (reduce_dim->size() != 2 * input_dim_size - 1) {
        continue;
      }
      if (reduce_dim->padding_low() != input_dim_size - 1) {
        continue;
      }
      if (reduce_dim->padding_high() != input_dim_size - 1) {
        continue;
      }
      // 5) If (4), then add the reduce-window candidate.
      VLOG(2) << "Adding Candidate ReduceWindow:" << reduce_window->ToString();
      candidate_rw.push_back(std::make_pair(reduce_window, reduce_dim_index));
    }
  }
  // Loop through the candidate reduce-windows and deconstruct them into their
  // reduce and broadcast equivalents.
  for (const auto& rw : candidate_rw) {
    auto reduce_window = rw.first;
    auto reduce_dim_index = rw.second;
    if (reduce_window == nullptr || reduce_dim_index < 0 ||
        reduce_dim_index >= reduce_window->operand(0)->shape().rank()) {
      continue;
    }
    std::vector<int64_t> reduce_instr_dimensions;
    std::vector<int64_t> broadcast_dimensions;
    const Window& window = reduce_window->window();
    // Below loop identifies the logical dimensions that were not reduced.
    // These logical dimensions are used to create the reduce HLO's output
    // shape and the broadcast HLO's dimensions parameter.
    for (int64_t index = 0; index < window.dimensions().size(); ++index) {
      const auto& window_dimension = window.dimensions(index);
      if (window_dimension.size() == 1) {
        reduce_instr_dimensions.push_back(
            reduce_window->operand(0)->shape().dimensions(index));
        broadcast_dimensions.push_back(index);
      }
    }
    Shape reduce_shape = ShapeUtil::MakeShape(
        reduce_window->shape().element_type(), reduce_instr_dimensions);
    auto reduce_instr =
        reduce_window->AddInstruction(HloInstruction::CreateReduce(
            reduce_shape, reduce_window->mutable_operand(0),
            reduce_window->mutable_operand(1), {reduce_dim_index},
            reduce_window->called_computations()[0]));
    auto broadcast_instr =
        reduce_window->AddInstruction(HloInstruction::CreateBroadcast(
            reduce_window->shape(), reduce_instr, broadcast_dimensions));
    VLOG(2) << "reduce_window:" << reduce_window->ToString();
    VLOG(2) << "reduce:" << reduce_instr->ToString();
    VLOG(2) << "broadcast:" << broadcast_instr->ToString();
    TF_CHECK_OK(reduce_window->parent()->ReplaceInstruction(reduce_window,
                                                            broadcast_instr));
    changed = true;
  }
  return changed;
}
}  // namespace xla
