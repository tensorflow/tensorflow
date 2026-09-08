/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/autotuner/codegen_orchestrator.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/autotune_cache.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

absl::Status MakeCombinedConfigError(absl::Span<const absl::Status> errors) {
  std::string combined_error = "All backends failed to find supported configs:";
  for (const auto& err : errors) {
    absl::StrAppend(&combined_error, "\n - ", err.ToString());
  }
  return absl::InternalError(combined_error);
}

absl::StatusOr<std::vector<autotuner::Config>> LoadCandidateConfigs(
    absl::string_view candidate_configs_file,
    absl::Span<const std::unique_ptr<CodegenBackend>> codegen_backends) {
  if (candidate_configs_file.empty()) {
    return std::vector<autotuner::Config>{};
  }
  std::string content;
  absl::Status read_status = tsl::ReadFileToString(
      tsl::Env::Default(), std::string(candidate_configs_file), &content);
  if (!read_status.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to read candidate configs file '",
                     candidate_configs_file, "': ", read_status.message()));
  }
  autotuner::CandidateConfigs candidate_configs_proto;
  bool parsed = tsl::protobuf::TextFormat::ParseFromString(
      content, &candidate_configs_proto);
  if (!parsed) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse candidate configs file '", candidate_configs_file,
        "' as textproto or binary proto."));
  }
  std::vector<autotuner::Config> candidate_configs;
  candidate_configs.reserve(candidate_configs_proto.configs_size());
  for (const auto& config : candidate_configs_proto.configs()) {
    bool backend_found = false;
    for (const auto& backend : codegen_backends) {
      if (backend->backend() == config.backend()) {
        backend_found = true;
        break;
      }
    }
    if (!backend_found) {
      return absl::InvalidArgumentError(
          absl::StrCat("Backend ", Backend_Name(config.backend()),
                       " in candidate configs is not registered with the "
                       "orchestrator."));
    }
    candidate_configs.push_back(config);
  }
  return candidate_configs;
}

}  // namespace

absl::StatusOr<std::unique_ptr<CodegenOrchestrator>>
CodegenOrchestrator::Create(
    std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
    Options options) {
  if (codegen_backends.empty()) {
    return absl::InvalidArgumentError(
        "CodegenOrchestrator initialization failed. No codegen backends "
        "provided.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::vector<autotuner::Config> candidate_configs,
      LoadCandidateConfigs(options.candidate_configs_file, codegen_backends));
  return absl::WrapUnique(
      new CodegenOrchestrator(std::move(codegen_backends), std::move(options),
                              std::move(candidate_configs)));
}

absl::StatusOr<std::vector<CodegenOrchestrator::Config>>
CodegenOrchestrator::GetSupportedConfigs(const HloInstruction& instr) const {
  if (!candidate_configs_.empty()) {
    std::vector<Config> configs;
    configs.reserve(candidate_configs_.size());
    for (const auto& candidate : candidate_configs_) {
      for (const auto& codegen_backend : codegen_backends_) {
        if (codegen_backend->backend() == candidate.backend()) {
          configs.push_back(Config{
              codegen_backend.get(),
              std::make_unique<BackendConfig>(candidate.backend_config())});
          break;
        }
      }
    }
    return configs;
  }
  std::vector<Config> configs;
  std::vector<absl::Status> errors;
  for (auto& codegen_backend : codegen_backends_) {
    absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
        per_backend_configs = codegen_backend->GetSupportedConfigs(instr);
    if (!per_backend_configs.ok()) {
      errors.push_back(per_backend_configs.status());
      VLOG(3) << "Failed to get supported configs for backend "
              << codegen_backend->name() << ": "
              << per_backend_configs.status();
      continue;
    }
    VLOG(3) << "Found " << per_backend_configs->size()
            << " supported configs for backend " << codegen_backend->name();
    for (auto& config : *per_backend_configs) {
      configs.push_back(Config{codegen_backend.get(), std::move(config)});
    }
  }
  if (configs.empty() && !errors.empty()) {
    return MakeCombinedConfigError(errors);
  }
  return configs;
}

absl::StatusOr<std::vector<CodegenOrchestrator::EstimatedConfig>>
CodegenOrchestrator::GetSupportedConfigsWithEstimates(
    const HloInstruction& instr) const {
  if (!candidate_configs_.empty()) {
    ABSL_ASSIGN_OR_RETURN(std::vector<Config> configs, GetSupportedConfigs(instr));
    std::vector<EstimatedConfig> estimated_configs;
    estimated_configs.reserve(configs.size());
    for (auto& config : configs) {
      estimated_configs.push_back(
          EstimatedConfig{std::move(config), std::nullopt});
    }
    return estimated_configs;
  }
  std::vector<EstimatedConfig> configs;
  std::vector<absl::Status> errors;
  for (auto& codegen_backend : codegen_backends_) {
    absl::StatusOr<std::vector<CodegenBackend::EstimatedConfig>>
        per_backend_configs =
            codegen_backend->GetSupportedConfigsWithEstimates(instr);
    if (!per_backend_configs.ok()) {
      errors.push_back(per_backend_configs.status());
      VLOG(3) << "Failed to get supported configs with estimates for backend "
              << codegen_backend->name() << ": "
              << per_backend_configs.status();
      continue;
    }
    VLOG(3) << "Found " << per_backend_configs->size()
            << " supported configs with estimates for backend "
            << codegen_backend->name();
    for (auto& config : *per_backend_configs) {
      configs.push_back(EstimatedConfig{
          Config{codegen_backend.get(), std::move(config.config)},
          config.estimated_runtime});
    }
  }
  if (configs.empty() && !errors.empty()) {
    return MakeCombinedConfigError(errors);
  }
  return configs;
}

absl::StatusOr<CodegenOrchestrator::Config>
CodegenOrchestrator::GetDefaultConfig(const HloInstruction& instr) const {
  std::vector<absl::Status> errors;
  for (auto& backend : codegen_backends_) {
    auto config = backend->GetDefaultConfig(instr);
    if (config.ok()) {
      return Config{backend.get(), std::move(*config)};
    }
    errors.push_back(config.status());
  }
  std::string combined_error =
      absl::StrCat("No backend with default config found for instruction: ",
                   instr.ToString());
  for (const auto& err : errors) {
    absl::StrAppend(&combined_error, "\n - ", err.ToString());
  }
  return absl::NotFoundError(combined_error);
}

absl::StatusOr<std::unique_ptr<Executable>> CodegenOrchestrator::Compile(
    const HloInstruction& instr, const Config& config) const {
  VLOG(4) << "Compiling config " << config.ToString() << " for HLO "
          << instr.ToString();
  absl::StatusOr<std::unique_ptr<Executable>> executable =
      config.codegen_backend->Compile(instr, *config.backend_config);
  if (absl::Status status = IsValidExecutable(executable, instr, config);
      !status.ok()) {
    return status;
  }
  return executable;
}

tsl::Future<std::vector<CodegenOrchestrator::MaybeExecutableCandidate>>
CodegenOrchestrator::CompileAll(const HloInstruction& instr,
                                std::vector<Config> configs,
                                tsl::thread::ThreadPool* thread_pool) const {
  tsl::Executor* executor = thread_pool != nullptr
                                ? thread_pool->AsExecutor()
                                : &tsl::InlineExecutor::Instance();

  std::vector<tsl::Future<MaybeExecutableCandidate>> futures;
  futures.reserve(configs.size());
  for (int i = 0; i < configs.size(); ++i) {
    futures.push_back(tsl::MakeFutureOn(
        *executor, [&, config = std::move(configs[i])]() mutable {
          absl::StatusOr<std::unique_ptr<Executable>> executable =
              Compile(instr, config);
          return MaybeExecutableCandidate{std::move(config),
                                          std::move(executable)};
        }));
  }
  return tsl::JoinFutures(absl::MakeSpan(futures));
}

absl::Status CodegenOrchestrator::ApplyConfig(HloInstruction& instr,
                                              const Config& config) const {
  return config.codegen_backend->ApplyConfig(instr, *config.backend_config);
}

absl::Status CodegenOrchestrator::IsValidExecutable(
    const absl::StatusOr<std::unique_ptr<Executable>>& executable,
    const HloInstruction& instr, const Config& config) const {
  if (!executable.ok()) {
    return tsl::errors::CreateWithUpdatedMessage(
        executable.status(),
        absl::StrCat("Compilation failed: ", executable.status().message()));
  }

  if (!*executable) {
    return absl::OkStatus();
  }

  if (options_.allow_reg_spills_fn &&
      options_.allow_reg_spills_fn(instr, config.codegen_backend->backend())) {
    return absl::OkStatus();
  }

  // Fail if any registers spilled.
  ModuleStats module_stats = (*executable)->module_stats();
  for (const auto& [kernel_name, kernel_stats] : module_stats) {
    if (kernel_stats.store_bytes_spilled > 0 ||
        kernel_stats.load_bytes_spilled > 0) {
      return absl::ResourceExhaustedError(
          "Discarding compilation due to register spilling.");
    }
  }
  return absl::OkStatus();
}

std::string CodegenOrchestrator::Config::ToString() const {
  return absl::StrFormat("%s : %s", codegen_backend->name(),
                         backend_config->ShortDebugString());
}

}  // namespace xla
