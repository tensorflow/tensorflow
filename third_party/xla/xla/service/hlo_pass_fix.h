/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_PASS_FIX_H_
#define XLA_SERVICE_HLO_PASS_FIX_H_

#include <algorithm>
#include <type_traits>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/types.h"

namespace xla {

// Do an HLO pass to a fix point.
template <typename Pass, int kIterationLimit = 25>
class HloPassFix : public Pass {
 public:
  static_assert(std::is_base_of<HloPassInterface, Pass>::value,
                "Pass must be a subclass of HloPassInterface");
  using RunState = HloPassInterface::RunState;
  template <typename... Args>
  explicit HloPassFix(Args&&... args) : Pass(args...) {}

  Status RunOnChangedComputations(HloModule* module, RunState* outer_run_state,
                                  const absl::flat_hash_set<absl::string_view>&
                                      execution_threads) override {
    RunState run_state;
    run_state.changed_last_iteration = outer_run_state->changed_last_iteration;
    TF_RETURN_IF_ERROR(RunToFixPoint(module, &run_state, execution_threads));
    outer_run_state->changed_this_iteration.insert(run_state.changed.begin(),
                                                   run_state.changed.end());
    return OkStatus();
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&
                               execution_threads) override {
    RunState run_state(module);
    TF_RETURN_IF_ERROR(RunToFixPoint(module, &run_state, execution_threads));
    return !run_state.changed.empty();
  }

  using HloPassInterface::RunOnModuleGroup;
  absl::StatusOr<bool> RunOnModuleGroup(
      HloModuleGroup* module_group,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override {
    bool changed = false;
    bool changed_this_iteration = true;
    int64_t iteration_count = 0;
    VLOG(3) << "Running HloPassFix.";
    while (changed_this_iteration) {
      TF_ASSIGN_OR_RETURN(
          changed_this_iteration,
          Pass::RunOnModuleGroup(module_group, execution_threads));
      changed |= changed_this_iteration;
      VLOG(3) << "changed_this_iteration: " << changed_this_iteration;
      ++iteration_count;
      if (iteration_count == kIterationLimit) {
        VLOG(1) << "Unexpectedly high number of iterations in HLO passes, "
                   "exiting fixed point loop.";
        // Return false in case this is fixed point is nested.
        return false;
      }
    }
    return changed;
  }

 private:
  Status RunToFixPoint(
      HloModule* module, RunState* run_state,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {
    VLOG(3) << "Running HloPassFix on " << Pass::name();
    while (!run_state->changed_last_iteration.empty()) {
      TF_RETURN_IF_ERROR(
          RunOnChangedComputationsOnce(module, run_state, execution_threads));
      VLOG(3) << Pass::name() << " iteration " << run_state->iteration
              << " changed_this_iteration: "
              << !run_state->changed_last_iteration.empty();
      run_state->IncrementIteration();
      if (run_state->iteration == kIterationLimit) {
        VLOG(1) << "Unexpectedly high number of iterations in HLO passes '"
                << Pass::name() << "' for module '" << module->name()
                << "'. Exiting fixed point loop.";
        // Clear changed and abort in case this is fixed point is nested.
        run_state->changed.clear();
        break;
      }
    }
    return OkStatus();
  }

  Status RunOnChangedComputationsOnce(
      HloModule* module, RunState* run_state,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {
    // If Pass overrides RunOnChangedComputations, just forward to it.
    if (!std::is_same<decltype(&HloPassInterface::RunOnChangedComputations),
                      decltype(&Pass::RunOnChangedComputations)>::value) {
      return Pass::RunOnChangedComputations(module, run_state,
                                            execution_threads);
    }
    // If Pass does not override the default
    // HloPassInterface::RunOnChangedComputations that calls into
    // HloPassFix<Pass>::Run, avoid infinite recursion.
    TF_ASSIGN_OR_RETURN(bool changed, Pass::Run(module, execution_threads));
    if (changed) {
      auto computations = module->computations(execution_threads);
      run_state->changed_this_iteration.insert(computations.begin(),
                                               computations.end());
    }
    return OkStatus();
  }
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_PASS_FIX_H_
