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

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/tools/hlo_opt/opt_lib.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

class CpuOptProvider : public OptProvider {
 public:
  absl::StatusOr<std::optional<std::string>> GenerateStage(
      std::unique_ptr<HloModule> module, absl::string_view s) override {
    if (s == "llvm-before-optimizations") {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                          GetExecutable(std::move(module)));
      return static_cast<cpu::CpuExecutable*>(executable.get())
          ->ir_module_string();
    }
    return OptProvider::GenerateStage(std::move(module), s);
  }

  std::set<std::string> SupportedStages() override {
    std::set<std::string> supported = OptProvider::SupportedStages();
    supported.insert({"llvm-before-optimizations"});
    return supported;
  }

  std::string GetPlatformName() override { return "cpu"; }
};

}  // namespace
}  // namespace xla

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(cpu_opt_provider, {
  xla::OptProvider::RegisterForPlatform(
      "cpu", std::make_unique<xla::CpuOptProvider>());
});
