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

#ifndef XLA_HLO_PASS_HLO_PASS_FIX_H_
#define XLA_HLO_PASS_HLO_PASS_FIX_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

// Do an HLO pass to a fix point.
template <typename Pass>
class HloPassFix : public Pass {
 public:
  static_assert(std::is_base_of<HloPassInterface, Pass>::value,
                "Pass must be a subclass of HloPassInterface");
  using RunState = HloPassInterface::RunState;
  template <typename... Args>
  explicit HloPassFix(Args&&... args) : Pass(args...) {}
  template <typename... Args>
  static std::unique_ptr<HloPassFix> Create(int iteration_limit,
                                            Args&&... args) {
    auto pass = std::make_unique<HloPassFix>(args...);
    pass->iteration_limit_ = iteration_limit;
    return pass;
  }

  absl::Status RunOnChangedComputations(
      HloModule* module, RunState* outer_run_state,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override {
    RunState run_state;
    run_state.changed_last_iteration = outer_run_state->changed_last_iteration;
    TF_RETURN_IF_ERROR(RunToFixPoint(module, &run_state, execution_threads));
    outer_run_state->changed_this_iteration.insert(run_state.changed.begin(),
                                                   run_state.changed.end());
    return absl::OkStatus();
  }

 protected:
  absl::StatusOr<bool> RunImpl(HloModule* module,
                               const absl::flat_hash_set<absl::string_view>&
                                   execution_threads) override {
    RunState run_state(module);
    TF_RETURN_IF_ERROR(RunToFixPoint(module, &run_state, execution_threads));
    return !run_state.changed.empty();
  }

 private:
  absl::Status RunToFixPoint(
      HloModule* module, RunState* run_state,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {
    VLOG(3) << "Running HloPassFix on " << Pass::name();

    absl::flat_hash_set<size_t> hashes;
    while (!run_state->changed_last_iteration.empty()) {
      if (module->config().debug_options().xla_hlo_pass_fix_detect_cycles()) {
        size_t hash = absl::HashOf(*module);
        VLOG(3) << "Module hash for " << Pass::name() << " at iteration "
                << run_state->iteration << ": " << hash;
        if (hashes.contains(hash)) {
          LOG(WARNING) << "Cycle detected when running " << Pass::name()
                       << " on iteration " << run_state->iteration
                       << "; hash: " << hash;
        } else {
          hashes.insert(hash);
        }
      }
      TF_RETURN_IF_ERROR(
          RunOnChangedComputationsOnce(module, run_state, execution_threads));
      VLOG(3) << Pass::name() << " iteration " << run_state->iteration
              << " changed_this_iteration: "
              << !run_state->changed_this_iteration.empty();
      run_state->IncrementIteration();
      if (run_state->iteration == iteration_limit_) {
        const DebugOptions& debug_options = module->config().debug_options();
        if (debug_options
                .xla_unsupported_crash_on_hlo_pass_fix_max_iterations()) {
          LOG(FATAL) << "Unexpectedly high number of iterations "
                     << iteration_limit_ << " in HLO pass '" << Pass::name()
                     << "' for module '" << module->name() << "'";
        }
        VLOG(1) << "Unexpectedly high number of iterations in HLO passes '"
                << Pass::name() << "' for module '" << module->name()
                << "'. Exiting fixed point loop.";
        // When crash on silent HLO changes is enabled, we can't lie about not
        // changing the module, as that will lead to an immediate crash.
        if (!debug_options
                 .xla_unsupported_crash_on_hlo_pass_silent_hlo_change()) {
          // Clear changed and abort in case this is fixed point is nested.
          run_state->changed.clear();
        }
        break;
      }
    }
    VLOG(3) << "Finished running HloPassFix on " << Pass::name();
    return absl::OkStatus();
  }

  absl::Status RunOnChangedComputationsOnce(
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
    // HloPassFix<Pass>::RunImpl, avoid infinite recursion.
    TF_ASSIGN_OR_RETURN(bool changed, Pass::RunImpl(module, execution_threads));
    if (changed) {
      auto computations = module->computations(execution_threads);
      run_state->changed_this_iteration.insert(computations.begin(),
                                               computations.end());
    }
    return absl::OkStatus();
  }

  static constexpr int kDefaultIterationLimit = 25;
  int iteration_limit_ = kDefaultIterationLimit;
};

}  // namespace xla

#endif  // XLA_HLO_PASS_HLO_PASS_FIX_H_
