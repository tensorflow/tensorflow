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

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/platform_util.h"
#include "xla/statusor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
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

absl::StatusOr<se::StreamExecutor*> OptProvider::GetExecutor() {
  DebugOptions debug_opts = GetDebugOptionsFromFlags();
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform(GetPlatformName()));
  if (debug_opts.xla_gpu_target_config_filename().empty()) {
    TF_ASSIGN_OR_RETURN(std::vector<se::StreamExecutor*> stream_executors,
                        PlatformUtil::GetStreamExecutors(
                            platform, /*allowed_devices=*/std::nullopt));
    return stream_executors[0];
  }
  return nullptr;
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
  } else if (stage == "hlo-backend") {
    TF_ASSIGN_OR_RETURN(auto executable, GetExecutable(std::move(module)));
    return executable->module().ToString();
  }

  return std::nullopt;
}

absl::StatusOr<Compiler*> OptProvider::GetCompiler() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform(GetPlatformName()));

  TF_ASSIGN_OR_RETURN(Compiler * compiler, Compiler::GetForPlatform(platform));
  return compiler;
}

absl::StatusOr<std::unique_ptr<HloModule>> OptProvider::GetOptimizedHlo(
    std::unique_ptr<HloModule> input_module) {
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor, GetExecutor());

  DebugOptions debug_opts = GetDebugOptionsFromFlags();
  Compiler::CompileOptions opts;
  TF_ASSIGN_OR_RETURN(Compiler * compiler, GetCompiler());
  DebugOptions d = input_module->config().debug_options();
  d.set_xla_embed_ir_in_executable(true);
  input_module->mutable_config().set_debug_options(d);

  if (input_module->has_schedule()) {
    return input_module;
  }

  // But run-hlo-passes does not actually run the scheduling.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> optimized_module,
      compiler->RunHloPasses(std::move(input_module), executor, opts));

  return optimized_module;
}

absl::StatusOr<std::unique_ptr<Executable>> OptProvider::GetExecutable(
    std::unique_ptr<HloModule> input_module) {
  Compiler::CompileOptions opts;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> optimized_module,
                      GetOptimizedHlo(std::move(input_module)));
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor, GetExecutor());
  TF_ASSIGN_OR_RETURN(Compiler * compiler, GetCompiler());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      compiler->RunBackend(std::move(optimized_module), executor, opts));
  return executable;
}

std::set<std::string> OptProvider::SupportedStages() {
  return {"hlo", "html", "hlo-backend"};
}

}  // namespace xla
