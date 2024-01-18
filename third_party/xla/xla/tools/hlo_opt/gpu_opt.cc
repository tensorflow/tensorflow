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

#include "absl/container/flat_hash_map.h"
#include "xla/debug_options_flags.h"
#include "xla/service/compiler.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/executable.pb.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/platform_util.h"
#include "xla/statusor.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/tools/hlo_opt/opt_lib.h"
#include "xla/types.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

class GpuOptProvider : public OptProvider {
 public:
  StatusOr<std::optional<std::string>> GenerateStage(
      std::unique_ptr<HloModule> module, absl::string_view s) override {
    if (s == "llvm") {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                          GetExecutable(std::move(module)));
      return static_cast<gpu::GpuExecutable*>(executable.get())
          ->ir_module_string();
    } else if (s == "ptx") {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                          GetExecutable(std::move(module)));
      return static_cast<gpu::GpuExecutable*>(executable.get())->text();
    } else if (s == "buffer-assignment") {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                          GetExecutable(std::move(module)));
      return static_cast<gpu::GpuExecutable*>(executable.get())
          ->buffer_assignment()
          ->ToVerboseString(9999);
    } else {
      // Delegate to base class.
      TF_ASSIGN_OR_RETURN(std::optional<std::string> out,
                          OptProvider::GenerateStage(std::move(module), s));
      return out;
    }
  }

  std::string GetPlatformName() override { return "gpu"; }

  std::set<std::string> SupportedStages() override {
    std::set<std::string> supported = OptProvider::SupportedStages();
    supported.insert({"ptx", "llvm", "buffer-assignment"});
    return supported;
  }
};

}  // namespace
}  // namespace xla

REGISTER_MODULE_INITIALIZER(gpu_opt_provider, {
  xla::OptProvider::RegisterForPlatform(
      "gpu", std::make_unique<xla::GpuOptProvider>());
});
