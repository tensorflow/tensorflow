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

#ifndef XLA_TOOLS_HLO_BISECT_HLO_BISECT_UTILS_H_
#define XLA_TOOLS_HLO_BISECT_HLO_BISECT_UTILS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tools/hlo_bisect/hlo_bisect_state.h"

namespace xla {
namespace bisect {

// Checks whether the execution of an HloModule on the test platform and the
// reference platform produce different results.
class MiscompareChecker : public BugCheckerInterface {
 public:
  MiscompareChecker(HloModule* module, std::vector<Literal>&& input_data,
                    absl::string_view test_platform,
                    absl::string_view reference_platform,
                    ErrorSpec error_spec);
  absl::StatusOr<bool> Run(const HloModule& module) override;
  absl::flat_hash_map<std::string, Literal> GetResults() override;

  virtual absl::StatusOr<std::unique_ptr<HloModule>> PrepareReferenceModule(
      const HloModule& hlo_module, HloRunnerInterface* hlo_runner) const;

 private:
  std::vector<Literal> input_data_;
  std::unique_ptr<HloRunnerInterface> reference_runner_;
  std::unique_ptr<HloRunnerInterface> test_runner_;
  absl::flat_hash_map<std::string, Literal> results_;
  ErrorSpec error_spec_;
};

// Runs a user provided script and considers an HLO module to be buggy if the
// script exits with a non-zero exit code.
class ScriptChecker : public BugCheckerInterface {
 public:
  explicit ScriptChecker(std::string path_to_script)
      : path_to_script_(std::move(path_to_script)) {}
  absl::StatusOr<bool> Run(const HloModule& module) override;
  absl::flat_hash_map<std::string, Literal> GetResults() override;

 private:
  std::string path_to_script_;
};

// Runner class for the bisect tool.
class BisectRunner {
 public:
  BisectRunner(std::unique_ptr<HloModule> module,
               std::unique_ptr<BugCheckerInterface> bug_checker)
      : module_(std::move(module)), bug_checker_(std::move(bug_checker)) {}

  absl::StatusOr<std::unique_ptr<HloModule>> RunEntry();
  absl::StatusOr<std::unique_ptr<HloModule>> RunAll();

 protected:
  std::unique_ptr<HloModule> module_;
  std::unique_ptr<BugCheckerInterface> bug_checker_;
};

// Main runner for the bisect tool.
void RunBisect(std::unique_ptr<BisectRunner> runner, bool all_computations,
               absl::string_view dump_path, absl::string_view output_format);

// Utility function for getting the verified module and optional inputs.
using ModuleWithInputs =
    std::pair<std::unique_ptr<HloModule>, std::vector<Literal>>;
absl::StatusOr<ModuleWithInputs> GetVerifiedModuleAndInputData(
    absl::string_view input_filename);

}  // namespace bisect
}  // namespace xla

#endif  // XLA_TOOLS_HLO_BISECT_HLO_BISECT_UTILS_H_
