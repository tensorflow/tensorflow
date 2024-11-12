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

#ifndef XLA_HLO_PASS_HLO_PASS_INTERFACE_H_
#define XLA_HLO_PASS_HLO_PASS_INTERFACE_H_

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {

// Base class for HLO passes. These are used with the HloPassPipeline to
// organize a sequence of passes. An HLO pass should not extend this class
// directly; it should extend HloModulePass or HloModuleGroupPass.
class HloPassInterface {
 public:
  // Struct that holds states of pass runs across multiple iterations.
  struct RunState {
    // The current iteration number.
    int iteration = 0;
    // Set of all changed computations from all pass runs using this state.
    absl::flat_hash_set<HloComputation*> changed;
    // Set of changed computation from previous iteration.
    absl::flat_hash_set<HloComputation*> changed_last_iteration;
    // Set of changed computation from current iteration.
    absl::flat_hash_set<HloComputation*> changed_this_iteration;

    RunState() = default;
    explicit RunState(HloModule* module)
        : changed_last_iteration(module->computations().begin(),
                                 module->computations().end()) {}

    // Transition to the next iteration.
    //
    // Depending on the pass implmentation, one iteration includes all the work
    // done between two IncrementIteration calls, there can be arbitrary number
    // of passes that ran arbitrary times with this state.
    void IncrementIteration() {
      using std::swap;
      changed.insert(changed_this_iteration.begin(),
                     changed_this_iteration.end());
      swap(changed_last_iteration, changed_this_iteration);
      changed_this_iteration.clear();
      ++iteration;
    }
  };
  virtual ~HloPassInterface() = default;
  virtual absl::string_view name() const = 0;

  // Run the pass on the given HLO module with specified execution_threads.
  // Empty execution_threads list means all execution_threads are included.
  // Returns whether it modified the module.
  //
  // Note: C++ hides non-explicitly declared overloaded functions.
  // You can make all overloaded variants available in the child class  by
  // adding `using HloPassInterface::Run;` to the child class declaration.
  absl::StatusOr<bool> Run(HloModule* module) {
    return Run(module, /*execution_threads=*/{});
  }
  virtual absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) = 0;

  // Run the pass on computation on changed computations from last iteration in
  // given HLO module for specified execution_threads, with caller provided
  // RunState which holds the state information across multiple iterations.
  //
  // NOTE: This is a temporary default implementation that conservatively treats
  // all computations as changed. Eventually all passes should override this
  // method instead of Run() and Run() will call into this method instead.
  virtual absl::Status RunOnChangedComputations(
      HloModule* module, RunState* run_state,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {
    TF_ASSIGN_OR_RETURN(bool changed, Run(module, execution_threads));
    if (changed) {
      auto computations = module->computations(execution_threads);
      run_state->changed_this_iteration.insert(computations.begin(),
                                               computations.end());
    }
    return absl::OkStatus();
  }

  // Run the pass on the given HLO module group for specified
  // `execution_threads`. Empty `execution_threads` list means all execution
  // threads are included. Returns whether it modified the module group.
  // Ideally, the module group variant would be named "Run" as well, but C++
  // does not handle overloaded virtual methods well.
  //
  // See the caveat about C++ hiding overloaded functions in the Run function
  // above.
  absl::StatusOr<bool> RunOnModuleGroup(HloModuleGroup* module_group) {
    return RunOnModuleGroup(module_group, /*execution_threads=*/{});
  }
  virtual absl::StatusOr<bool> RunOnModuleGroup(
      HloModuleGroup* module_group,
      const absl::flat_hash_set<absl::string_view>& execution_threads) = 0;

  virtual bool IsPassPipeline() const { return false; }

  // If an HloPassMetadata has previously been created, it adds a (key, value)
  // pair metric if none was already set or updates the existing value.
  // If an HloPassMetadata doesn't exist, it simply returns.
  static void SetKVMetric(HloModule* module, const std::string& key,
                          int64_t value) {
    auto status = module->metadata()->set_key_value_metric(key, value);
    if (!status.ok()) {
      // Only logging since this should not crash the application.
      // It usually means the pass was invoked on its own.
      LOG(WARNING) << "Failed to set stat: " << status;
    }
  }
};

// Base class for passes which are module-scoped.
class HloModulePass : public HloPassInterface {
 public:
  // Runs the pass on a module group by iterating through each module in the
  // group.
  absl::StatusOr<bool> RunOnModuleGroup(
      HloModuleGroup* module_group,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      override {
    bool changed = false;
    for (HloModule* module : module_group->modules()) {
      TF_ASSIGN_OR_RETURN(bool module_changed, Run(module, execution_threads));
      changed |= module_changed;
    }
    return changed;
  };

  // Update the layout of a Shape to one that is supported by a given backend.
  // One can call this function after modifying the Shape in case that modifying
  // the Shape requires changes to the layout for the given Backend.
  //
  // TODO(b/129084868): Make this Backend dependent instead of requiring
  // deriving from the pass and overriding this function.
  virtual void UpdateLayout(Shape* shape) {}
};

// Base class for passes which are module-group scoped. These passes cannot run
// on an HLO module.
class HloModuleGroupPass : public HloPassInterface {
 public:
  absl::StatusOr<bool> Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&
                               execution_threads) override {
    return Internal("Module group pass cannot be run on a module");
  }
};

}  // namespace xla

#endif  // XLA_HLO_PASS_HLO_PASS_INTERFACE_H_
