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

#ifndef XLA_HLO_TOOLS_HLO_OPT_OPT_LIB_H_
#define XLA_HLO_TOOLS_HLO_OPT_OPT_LIB_H_

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"

namespace xla {

// Platform-independent provider of `hlo-opt` functionality.
class OptProvider {
 public:
  OptProvider() : pass_registry_() { RegisterAllHardwareIndependentPasses(); }

  virtual ~OptProvider() = default;

  // Generates textual output for a given stage on a given platform, returns
  // empty optional if the stage is not supported.
  virtual absl::StatusOr<std::optional<std::string>> GenerateStage(
      std::unique_ptr<HloModule> module, absl::string_view stage);

  // Returns a set of stages supported by the opt provider.
  virtual std::set<std::string> SupportedStages();

  // Registers a given provider for a given platform.
  static void RegisterForPlatform(
      std::string platform, std::unique_ptr<OptProvider> translate_provider);

  // Gets a provider for a given platform.
  static absl::StatusOr<OptProvider*> GetProviderForPlatform(
      std::string platform);

  // Runs input passes on a input module and returns the optimized module
  // string.
  absl::StatusOr<std::optional<std::string>> BuildAndRunTransformPipeline(
      std::unique_ptr<HloModule> input_module,
      const std::string& input_pass_names);

  // Registers all passes and pipelines provided by this provider.
  virtual void RegisterProviderPasses(HloModule& module);

  // Returns a string of all registered pass names.
  virtual std::string GetRegisteredPassNames();

 protected:
  // Map of pass names to pass registration functions. The pass registration
  // function takes a HloPassPipeline and adds the corresponding pass to it.
  absl::flat_hash_map<std::string, std::function<void(HloPassPipeline&)>>
      pass_registry_;

  // Adds an entry of pass name vs pass registration function to registry.
  template <typename T, typename... Args>
  void RegisterPass(Args... args) {
    pass_registry_.insert(std::make_pair(
        std::string(T(std::forward<Args>(args)...).name()),
        [args...](HloPassPipeline& p) {
          p.AddPass<T>(std::forward<decltype(std::as_const(args))>(
              std::as_const(args))...);
        }));
  }

  // Registers all hardware independent passes.
  void RegisterAllHardwareIndependentPasses();

  // Returns a string of all registered pass names. Helper function for
  // GetRegisteredPassNames, avoids duplicating code for each provider.
  std::string GetRegisteredPassNamesHelper(
      const absl::flat_hash_map<
          std::string, std::function<void(HloPassPipeline&)>>& pass_registry_);
};

}  // namespace xla

#endif  // XLA_HLO_TOOLS_HLO_OPT_OPT_LIB_H_
