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

#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/platform_util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

using ProviderMap =
    absl::flat_hash_map<std::string, std::unique_ptr<OptProvider>>;
static absl::Mutex provider_mu(absl::kConstInit);

static ProviderMap& GetProviderMap() {
  static auto& provider_map = *new ProviderMap();
  return provider_map;
}

/*static*/ void OptProvider::RegisterForPlatform(
    std::string platform, std::unique_ptr<OptProvider> translate_provider) {
  absl::MutexLock l(&provider_mu);
  CHECK(!GetProviderMap().contains(platform));
  absl::StatusOr<std::string> canonical_name =
      xla::PlatformUtil::CanonicalPlatformName(platform);
  CHECK_OK(canonical_name);
  GetProviderMap()[*canonical_name] = std::move(translate_provider);
}

/*static*/ absl::StatusOr<OptProvider*> OptProvider::ProviderForPlatform(
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

absl::StatusOr<std::optional<std::string>> OptProvider::GenerateStage(
    std::unique_ptr<HloModule> module, absl::string_view stage) {
  if (stage == "hlo") {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> optimized_module,
                        GetOptimizedHlo(std::move(module)));
    return optimized_module->ToString();
  } else if (stage == "html") {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> optimized_module,
                        GetOptimizedHlo(std::move(module)));
    TF_ASSIGN_OR_RETURN(std::string cmps,
                        RenderAllComputationsToHtml(*optimized_module));
    return cmps;
  }

  return std::nullopt;
}

std::set<std::string> OptProvider::SupportedStages() { return {"hlo", "html"}; }

absl::StatusOr<std::unique_ptr<HloModule>> OptProvider::GetOptimizedHlo(
    std::unique_ptr<HloModule> input_module) {
  return absl::UnimplementedError(
      "Not supported in platform independent opt tool");
}

}  // namespace xla
