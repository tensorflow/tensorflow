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

#include "xla/tools/hlo_opt/opt_lib.h"

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/tools/hlo_opt/transforms_example_passes.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {
using ProviderMap =
    absl::flat_hash_map<std::string, std::unique_ptr<OptProvider>>;
static absl::Mutex provider_mu(absl::kConstInit);

static ProviderMap& GetProviderMap() {
  static auto& provider_map = *new ProviderMap();
  return provider_map;
}
}  // namespace

/*static*/ void OptProvider::RegisterForPlatform(
    std::string platform, std::unique_ptr<OptProvider> translate_provider) {
  absl::MutexLock l(&provider_mu);
  CHECK(!GetProviderMap().contains(platform));
  absl::StatusOr<std::string> canonical_name =
      xla::PlatformUtil::CanonicalPlatformName(platform);
  CHECK_OK(canonical_name);
  GetProviderMap()[*canonical_name] = std::move(translate_provider);
}

/*static*/ absl::StatusOr<OptProvider*> OptProvider::GetProviderForPlatform(
    std::string platform) {
  absl::MutexLock l(&provider_mu);

  TF_ASSIGN_OR_RETURN(std::string canonical_name,
                      xla::PlatformUtil::CanonicalPlatformName(platform));
  auto it = GetProviderMap().find(canonical_name);
  if (it == GetProviderMap().end()) {
    return absl::UnimplementedError(absl::StrCat(
        "Provider not found for platform ", platform, "; canonical expansion: ",
        canonical_name, "; supported platforms are: ",
        absl::StrJoin(GetProviderMap(), ", ",
                      [&](std::string* s, const auto& p) {
                        absl::StrAppend(s, p.first);
                      })));
  }

  return it->second.get();
}

// Placeholder for `key function` of the class to avoid an error due to
// missing vtable entry.
absl::StatusOr<std::optional<std::string>> OptProvider::GenerateStage(
    std::unique_ptr<HloModule> module, absl::string_view stage) {
  return module->ToString();
}

absl::StatusOr<std::optional<std::string>>
OptProvider::BuildAndRunTransformPipeline(std::unique_ptr<HloModule> module,
                                          const std::string& input_pass_names) {
  HloPassPipeline transforms_pipeline{"transforms_pipeline"};
  for (const auto& pass_name :
       std::vector<std::string>(absl::StrSplit(input_pass_names, ','))) {
    auto it = pass_registry_.find(pass_name);
    if (it != pass_registry_.end()) {
      it->second(transforms_pipeline);
    } else {
      LOG(ERROR) << "Pass " << pass_name << " not found.";
    }
  }
  CHECK_OK(transforms_pipeline.Run(module.get(), {}));
  return module->ToString();
}

std::set<std::string> OptProvider::SupportedStages() { return {"hlo"}; }

// Hardware Independent passes are already registered in the constructor.
// Hardware Specific passes can be populated by respective hardware provider
// subclasses using this method.
void OptProvider::RegisterProviderPasses(HloModule& module) {}

// Register Hardware-independent HLO passes here if you want the hlo-opt tool
// to be able to apply them.
void OptProvider::RegisterAllHardwareIndependentPasses() {
  RegisterPass<FooToBarModulePass>();
  RegisterPass<BarToHelloModulePass>();
}

}  // namespace xla

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(transforms_opt_provider, {
  xla::OptProvider::RegisterForPlatform("transforms",
                                        std::make_unique<xla::OptProvider>());
});
