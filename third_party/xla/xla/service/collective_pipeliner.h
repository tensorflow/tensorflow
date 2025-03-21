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

#ifndef XLA_SERVICE_COLLECTIVE_PIPELINER_H_
#define XLA_SERVICE_COLLECTIVE_PIPELINER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// This transformation peels off loop iterations of models with stacked layers
// that perform data parallelism using reduce-scatter/all-reduce/all-gather.
// Collective instructions are pushed to the next iteration in which they can
// overlap with the entirely of the next layer rather than with a more limited
// amount of computation in the current iteration. An example of transformation
// is this:
//
// while (i < LAYERS) {
//   p0 = param(0)
//   p1 = param(1)
//   x = computation(p0)
//   xg = all-reduce(x)
//   y = computation(p1)
//   yg = all-reduce(y)
// }
//
// to
//
// x_prev = computation(p0)
// y_prev = computation(p1)
// i = i + 1
// while (i < LAYERS, x_prev, y_prev) {
//   p0 = param(0)
//   p1 = param(1)
//   xg = all-reduce(x_prev)
//   yg = all-reduce(y_prev)
//   x = computation(p0)
//   y = computation(p1)
//   x_prev = x
//   y_prev = y
// }
class CollectivePipeliner : public HloModulePass {
 public:
  enum PipeliningDirection {
    kBackward,
    kForward,
    kForwardSink,
  };

  // Postprocessing cloned collective instructions, such as peeled instructions
  // before and after the loop, and rotated instructions. The new while op is
  // only passed for the peeled trailing ops when the new while op was already
  // created.
  using HloPostprocessor = std::optional<std::function<absl::Status(
      HloInstruction* instr, HloInstruction* new_while_instr)>>;

  struct Config {
    int64_t level_to_operate_on = 0;
    // Maximum number of HLOs to pipeline per loop. (Meant to help controlling
    // memory pressure manually).
    int64_t max_pipelining_per_loop = 0;
    bool last_run = true;
    // The pipeliner should try to pipeline instructions that have a tree of
    // uses of allowed instructions. This could increase memory pressure as
    // multiple instructions might have to be saved to be pushed to the next
    // iteration.
    bool pipeline_use_tree = false;
    bool process_different_sized_ops = false;
    PipeliningDirection pipelining_direction = PipeliningDirection::kForward;
    HloPredicate should_process;
    // Filter acceptable formatting ops for for forward pipelining to discard
    // cases that pipeline formatting operations that we don't want to support.
    HloPredicate acceptable_formatting;
    // If the pipelined op has same input/output size the we reuse  the same
    // buffer we are storing the value in in the output loop for forward
    // pipelining. This function allows to not do it for certain ops.
    HloPredicate reuse_pipelined_op_buffer;
    // Determine whether a loop variant parameter should be allowed in
    // pipelining chains. This is currently only used to support kBackward
    // pipelinining.
    HloPredicate should_allow_loop_variant_parameter_in_chain =
        HloPredicateFalse;
    // Whether we allow control dependencies on the Collective operation being
    // pipelined. The control dependencies will be dropped when the operation is
    // pipelined. This is currently only used to support kBackward pipelining.
    bool should_allow_control_dependencies = false;
    // TODO(b/399476667): Consolidate these postprocessing functions.
    HloPostprocessor postprocess_backward_peeled_op = std::nullopt;
    HloPostprocessor postprocess_backward_rotated_op = std::nullopt;
    HloPostprocessor postprocess_backward_peeled_trailing_op = std::nullopt;
    // Determines whether a loop invariant instruction can be considered
    // in the pipelining chain.
    bool should_add_loop_invariant_op_in_chain = false;
    // Postprocessing hook which runs for every successfully pipelined op.
    HloPostprocessor postprocess_pipelined_ops = std::nullopt;
    int64_t collective_size_threshold_to_stop_sinking = INT64_MAX;
  };
  static const char* const kInsertedByPreviousStep;
  static const char* const kSunkByPreviousStep;
  explicit CollectivePipeliner(const Config& config) : config_(config) {}
  CollectivePipeliner(CollectivePipeliner&& other) = default;
  CollectivePipeliner& operator=(CollectivePipeliner&& other) = default;
  absl::string_view GetPipelineDirectionString(PipeliningDirection direction) {
    switch (direction) {
      case PipeliningDirection::kForward: {
        return "forward";
      }
      case PipeliningDirection::kBackward: {
        return "backward";
      }
      case PipeliningDirection::kForwardSink: {
        return "forwardsink";
      }
    }
  }

  static constexpr absl::string_view kName = "collective-pipeliner";
  absl::string_view name() const override {
    if (config_.pipelining_direction == kForward) {
      return "collective-pipeliner-forward";
    } else if (config_.pipelining_direction == kBackward) {
      return "collective-pipeliner-backward";
    } else {
      return "collective-pipeliner-forwardsink";
    }
  }

  // Pipelines the collectives that do not have any other pipelineable
  // collectives in their user subtree.
  absl::StatusOr<bool> RunPipeliner(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const Config config_;
};

}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_PIPELINER_H_
