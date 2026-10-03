/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_PROPAGATOR_H_
#define XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_PROPAGATOR_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/shape_util.h"

namespace xla::numerics::comparison {

// Propagates tensor summaries across specific HLO instructions within an
// original HLO module.
//
// This class is designed to be used as the second pass in a three-pass
// comparison architecture (Recover, Propagate, Compare). It takes an initial
// set of "raw" tensor summaries recovered from runtime logs and augments this
// set by propagating summaries across simple, value-preserving HLO instructions
// like reshape, transpose, copy, and broadcast.
//
// The propagation is performed iteratively until a fixed point is reached,
// meaning no new summaries can be generated. This helps create a more dense
// set of summaries for comparison, addressing issues where logging is sparse.
//
// The propagator ensures that it does not create a summary for a tensor that
// already has one, preventing duplicates.
class OriginalTensorSummaryPropagator {
 public:
  // `original_module`: The HLO module to propagate summaries within. The
  // propagator does not take ownership.
  // `on_propagated_tensor_summary`: A callback that will be invoked for
  // each newly created summary.
  // `is_original_tensor_already_recovered`: A callback that will be invoked
  // to check if a tensor has already been recovered from
  // OriginalTensorSummaryCalculator.
  OriginalTensorSummaryPropagator(
      const HloModule* original_module,
      OriginalTensorSummaryCallback&& on_propagated_tensor_summary,
      IsOriginalTensorAlreadyRecoveredCallback&&
          is_original_tensor_already_recovered);

  // Not copyable.
  OriginalTensorSummaryPropagator(const OriginalTensorSummaryPropagator&) =
      delete;
  OriginalTensorSummaryPropagator& operator=(
      const OriginalTensorSummaryPropagator&) = delete;

  // Initialize the propagator. This should be called before any other
  // methods. Note that this method may call `on_propagated_tensor_summary_`
  // callback. This is also why the logic in this method is not folded into  the
  // constructor.
  absl::Status Initialize();

  // Process from the given root tensor key across the original HLO module along
  // edges of value preserving HLO instructions. It's assumed that these tensors
  // are provided in the order following the actual computation, respecting
  // all data dependencies.
  absl::Status Process(
      const AbsoluteScopedTensorKey& original_tensor_key,
      std::shared_ptr<const tensor_transformation::TensorTransformation>
          pending_transformation,
      const OriginalTensorSummary& root_tensor_summary);

  // Finish the propagation. This should be called after all Process() calls
  // are done.
  absl::Status Finish();

  struct ProcessingMetrics {
    int64_t skipped_unrecoverable_tensor_summaries = 0;
    int64_t recovered_from_runtime_count = 0;
    int64_t total_propagated_tensor_count = 0;
  };

  const ProcessingMetrics& GetProcessingMetrics() const {
    return processing_metrics_;
  }

 private:
  struct CallArgKey {
    int64_t arg_number;
    ShapeIndex shape_index;

    bool operator==(const CallArgKey& other) const {
      return arg_number == other.arg_number && shape_index == other.shape_index;
    }
    bool operator!=(const CallArgKey& other) const { return !(*this == other); }
    template <typename H>
    friend H AbslHashValue(H h, const CallArgKey& key) {
      return H::combine(std::move(h), key.arg_number, key.shape_index);
    }
  };

  struct CallFrameState {
    absl::flat_hash_map<CallArgKey, OriginalTensorSummary> arg_summaries;
  };

  struct CallFrame {
    ScopeInstruction call_instruction;
    // The state of the current call frame
    CallFrameState current_state;
    // The states of the child call frames indexed by the call-like instruction.
    // The purpose of this is to track the tensor summaries of computation
    // arguments when propagating summaries across the current call frame.
    // When a child call actually happens (due to a Process() call with a
    // absolute scoped tensor identifying an instruction in the child call),
    // the child state would be moved from this map to top of the call stack.
    absl::flat_hash_map<ScopeInstruction, CallFrameState> child_states;

    // The set of tensors that have been propagated in this computation. This is
    // used to avoid propagating the same tensor summary multiple times.
    absl::flat_hash_set<TensorKey> propagated_tensors;

    HloComputation* computation = nullptr;
    bool should_propagate_parameters = false;

    static CallFrame ForComputation(HloComputation* comp) {
      CallFrame frame;
      frame.computation = comp;
      return frame;
    }
    static CallFrame ForCallInstruction(ScopeInstruction call_instr) {
      CallFrame frame;
      frame.call_instruction = std::move(call_instr);
      return frame;
    }
  };

  absl::Status OnEnterCall(ScopeInstruction call_instruction,
                           absl::string_view instruction_in_called_computation);

  absl::Status OnExitCall(ScopeInstruction call_instruction);

  absl::Status OnNextIteration(ScopeInstruction next_iteration_instruction);

  // Propagate tensor summaries for instructions like constant, iota, etc and
  // parameters, recursively.
  absl::Status PropagateConstantsAndParameters();

  // Propagate tensor summaries forward across value-preserving HLO
  // instructions in the current computation. Forward means this propagation is
  // performed on users of the specified `original_tensor_key`. This propagation
  // logic is performed recursively, unless the tensor being propagated is
  // already recovered by `is_original_tensor_already_recovered_` or it's
  // already propagated by a previous call to PropagateForward() or
  // PropagateBackward(). This function also invokes
  // `on_propagated_tensor_summary_` for each newly created summary in the order
  // of propagation.
  // When propagation reaches a call instruction, the tensor summaries of
  // computation arguments will be propagated to the callee. In this case, the
  // `child_states` of the current call frame is updated.
  // Note that the starting tensor is already processed by the time this
  // function is called.
  absl::Status PropagateForward(
      const HloInstruction* starting_instruction,
      ShapeIndexView starting_shape_index,
      std::shared_ptr<const tensor_transformation::TensorTransformation>
          pending_transformation,
      const OriginalTensorSummary& original_tensor_summary);

  // Propagate tensor summaries backward across value-preserving HLO
  // instructions in the current computation. Backward means this propagation is
  // performed on the instruction itself and its operands. This propagation
  // logic is performed recursively, unless the tensor being propagated is
  // already recovered by `is_original_tensor_already_recovered_` or it's
  // already propagated by a previous call to PropagateForward() or
  // PropagateBackward(). This function also invokes
  // `on_propagated_tensor_summary_` for each newly created summary in the
  // reverse order of propagation. That is, the last propagated summary would
  // be passed to the callback first.
  // Note that the starting tensor is already processed by the time this
  // function is called.
  absl::Status PropagateBackward(
      const HloInstruction* starting_instruction,
      ShapeIndexView starting_shape_index,
      std::shared_ptr<const tensor_transformation::TensorTransformation>
          pending_transformation,
      const OriginalTensorSummary& original_tensor_summary);

  // Checks if propagation should proceed to the given instruction and shape
  // index.
  bool ShouldPropagateTo(const HloInstruction* instruction,
                         ShapeIndexView shape_index);

  absl::Status DoPropagateBackward(
      const HloInstruction* instruction, ShapeIndexView shape_index,
      std::shared_ptr<const tensor_transformation::TensorTransformation>
          transformation,
      const OriginalTensorSummary& original_tensor_summary);

  absl::Status DoPropagateForward(
      const HloInstruction* instruction, ShapeIndexView shape_index,
      std::shared_ptr<const tensor_transformation::TensorTransformation>
          transformation,
      const OriginalTensorSummary& original_tensor_summary);

  std::vector<ScopeInstruction> GetCurrentScopeInstructions() const;

  AbsoluteScopedTensorKey GetCurrentAbsoluteScopedTensorKey(
      absl::string_view instruction_name, ShapeIndexView shape_index) const;

  absl::Status InvokePropagatedCallback(
      const AbsoluteScopedTensorKey& original_tensor_key,
      std::shared_ptr<const tensor_transformation::TensorTransformation>
          pending_transformation,
      const OriginalTensorSummary& root_tensor_summary);

  const HloModule* original_module_;
  OriginalTensorSummaryCallback on_propagated_tensor_summary_;
  IsOriginalTensorAlreadyRecoveredCallback
      is_original_tensor_already_recovered_;
  std::vector<CallFrame> call_stack_;
  absl::flat_hash_map<absl::string_view, HloInstruction*> name_to_instruction_;
  ProcessingMetrics processing_metrics_;
  struct RecoveredTensorSummary {
    AbsoluteScopedTensorKey original_tensor_key;
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        pending_transformation;
    OriginalTensorSummary original_tensor_summary;
  };
  std::vector<RecoveredTensorSummary> wildcard_summaries_;
  absl::flat_hash_set<std::string> processed_wildcards_;
};

}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_PROPAGATOR_H_
