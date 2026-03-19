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

#ifndef XLA_TOOLS_HLO_OPT_COMPILED_OPT_LIB_H_
#define XLA_TOOLS_HLO_OPT_COMPILED_OPT_LIB_H_

#include <memory>
#include <optional>
#include <set>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/hlo_opt/opt_lib.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/platform.h"

namespace xla {

// Platform-specific provider of `hlo-opt` functionality.
class CompiledOptProvider : public OptProvider {
 public:
  CompiledOptProvider() : OptProvider() {
    RegisterSharedHardwareSpecificPasses();
  }

  // Generates textual output for a given stage on a given platform, returns
  // empty optional if the stage is not supported.
  absl::StatusOr<std::optional<std::string>> GenerateStage(
      std::unique_ptr<HloModule> module, absl::string_view stage) override;

  // Returns a set of stages supported by the opt provider.
  std::set<std::string> SupportedStages() override;

 protected:
  // Returns platform name associated with the provider.
  virtual std::string GetPlatformName() = 0;

  // Returns a stream executor for the provider (could be nullptr).
  virtual absl::StatusOr<se::StreamExecutor *> GetExecutor();

  // Generates executable from a given input module.
  absl::StatusOr<std::unique_ptr<Executable>> GetExecutable(
      std::unique_ptr<HloModule> input_module);

  // Generates optimized HLO.
  absl::StatusOr<std::unique_ptr<HloModule>> GetOptimizedHlo(
      std::unique_ptr<HloModule> input_module);

  // Gets a compiler associated with the provider.
  virtual absl::StatusOr<std::unique_ptr<Compiler>> GetCompiler();

  // Registers hardware-specific passes which are shared by
  // multiple backends (CPU, GPU, xPU).
  void RegisterSharedHardwareSpecificPasses();
};

}  // namespace xla

#endif  // XLA_TOOLS_HLO_OPT_COMPILED_OPT_LIB_H_
