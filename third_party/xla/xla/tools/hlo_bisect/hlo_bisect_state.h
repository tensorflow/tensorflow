/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_TOOLS_HLO_BISECT_HLO_BISECT_STATE_H_
#define XLA_TOOLS_HLO_BISECT_HLO_BISECT_STATE_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/statusor.h"

namespace xla {
namespace bisect {

// Processes an HloModule, such as compiling the module or executing the module,
// to check whether a bug exists. When the module is executed, should provide
// the resulting literals for the reference implementation.
class BugCheckerInterface {
 public:
  virtual ~BugCheckerInterface() {}

  // Returns true if `module` has a bug we're interested in.
  virtual absl::StatusOr<bool> Run(const HloModule& module) = 0;

  // Returns mapping of instruction names to their results after the run
  // (empty if this information is unavailable).
  virtual absl::flat_hash_map<std::string, Literal> GetResults() = 0;
};

// Trims down an HloModule that manifests a bug to a smaller module that
// still exhibits a problem. Only the entry computation is reduced.
class HloBisectState {
 public:
  explicit HloBisectState(std::unique_ptr<HloModule> module,
                          BugCheckerInterface* bug_checker)
      : module_(std::move(module)), bug_checker_(bug_checker) {}

  // Returns true if the current module has a bug and should be processed.
  absl::StatusOr<bool> ShouldProcess();

  // Trims the entry computation until no more reductions are possible. Returns
  // a boolean to indicate whether the computation has been reduced.
  absl::StatusOr<bool> TrimEntryComputation();

  // Returns the resulting module.
  std::unique_ptr<xla::HloModule>&& GetResult();

 private:
  // Runs a modified module and updates the foldable instructions data, if
  // available. Returns true if `module` has a bug.
  absl::StatusOr<bool> RunModule(const HloModule& module);

  // Trims the entry computation by reducing the total number of outputs.
  // Returns a boolean to indicate whether the computation has been reduced.
  absl::StatusOr<bool> TrimByOutputs();

  // Trims the entry computation by reducing the total number of instructions.
  // Returns a boolean to indicate whether the computation has been reduced.
  absl::StatusOr<bool> TrimByInstructions();

  // Trims the given computation by replacing instructions with constant values.
  // Returns a boolean to indicate whether the computation has been reduced.
  absl::StatusOr<bool> TrimByUsingConstants();

  // Asserts that the module still has the bug. If negative, runs the bug
  // checker repeatedly to verify that it's deterministic.
  absl::Status ExpectModuleIsBuggy();

  std::unique_ptr<xla::HloModule> module_;
  BugCheckerInterface* bug_checker_;
  absl::flat_hash_set<std::string> foldable_instructions_;
  absl::flat_hash_map<std::string, Literal> foldable_instructions_values_;
};

}  // namespace bisect
}  // namespace xla

#endif  // XLA_TOOLS_HLO_BISECT_HLO_BISECT_STATE_H_
