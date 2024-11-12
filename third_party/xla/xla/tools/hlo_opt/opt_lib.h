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

#ifndef XLA_TOOLS_HLO_OPT_OPT_LIB_H_
#define XLA_TOOLS_HLO_OPT_OPT_LIB_H_

#include <memory>
#include <optional>
#include <set>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// Platform-independent provider of `hlo-opt` functionality.
class OptProvider {
 public:
  // Generates textual output for a given stage on a given platform, returns
  // empty optional if the stage is not supported.
  virtual absl::StatusOr<std::optional<std::string>> GenerateStage(
      std::unique_ptr<HloModule> module, absl::string_view stage);

  virtual ~OptProvider() = default;

  // Returns a set of stages supported by the opt provider.
  virtual std::set<std::string> SupportedStages();

  // Registers a given provider for a given platform.
  static void RegisterForPlatform(
      std::string platform, std::unique_ptr<OptProvider> translate_provider);

  // Gets a provider for a given platform.
  static absl::StatusOr<OptProvider *> ProviderForPlatform(
      std::string platform);

 protected:
  // Generates optimized HLO.
  virtual absl::StatusOr<std::unique_ptr<HloModule>> GetOptimizedHlo(
      std::unique_ptr<HloModule> input_module);
};

}  // namespace xla

#endif  // XLA_TOOLS_HLO_OPT_OPT_LIB_H_
