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
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

absl::StatusOr<std::unique_ptr<CodegenOrchestrator>>
CodegenOrchestrator::Create(
    std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
    Options options, tsl::thread::ThreadPool* thread_pool) {
  if (codegen_backends.empty()) {
    return absl::InvalidArgumentError(
        "CodegenOrchestrator initialization failed. No codegen backends "
        "provided.");
  }
  return absl::WrapUnique(new CodegenOrchestrator(
      std::move(codegen_backends), std::move(options), thread_pool));
}

absl::StatusOr<std::vector<CodegenOrchestrator::Config>>
CodegenOrchestrator::GetSupportedConfigs(const HloInstruction& instr) const {
  std::vector<Config> configs;
  for (auto& codegen_backend : codegen_backends_) {
    absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
        per_backend_configs = codegen_backend->GetSupportedConfigs(instr);
    if (!per_backend_configs.ok()) {
      VLOG(3) << "Failed to get supported configs for backend "
              << codegen_backend->name() << ": "
              << per_backend_configs.status();
      continue;
    }
    VLOG(3) << "Found " << per_backend_configs->size()
            << " supported configs for backend " << codegen_backend->name();
    for (auto& config : *per_backend_configs) {
      configs.push_back({codegen_backend.get(), std::move(config)});
    }
  }
  return configs;
}

absl::StatusOr<CodegenOrchestrator::Config>
CodegenOrchestrator::GetDefaultConfig(const HloInstruction& instr) const {
  for (auto& backend : codegen_backends_) {
    auto config = backend->GetDefaultConfig(instr);
    if (config.ok()) {
      return Config{backend.get(), std::move(*config)};
    }
  }
  return absl::NotFoundError(
      absl::StrCat("No backend with default config found for instruction: ",
                   instr.ToString()));
}

absl::StatusOr<std::unique_ptr<Executable>> CodegenOrchestrator::Compile(
    const HloInstruction& instr, const Config& config) const {
  if (options_.exclude_cublas_config &&
      (config.codegen_backend->backend() ==
           autotuner::Backend::CUBLASLT_FISSION ||
       config.codegen_backend->backend() ==
           autotuner::Backend::HIPBLASLT_FISSION)) {
    return absl::CancelledError("exclude_cublas_config is set.");
  }
  VLOG(4) << "Compiling config " << config.ToString() << " for HLO "
          << instr.ToString();
  absl::StatusOr<std::unique_ptr<Executable>> executable =
      config.codegen_backend->Compile(instr, *config.backend_config);
  if (absl::Status status = IsValidExecutable(executable, instr);
      !status.ok()) {
    return status;
  }
  return executable;
}

tsl::Future<std::vector<CodegenOrchestrator::CompilationResult>>
CodegenOrchestrator::CompileAll(const HloInstruction& instr,
                                std::vector<Config> configs) const {
  tsl::Executor* executor = thread_pool_ != nullptr
                                ? thread_pool_->AsExecutor()
                                : &tsl::InlineExecutor::Instance();

  std::vector<tsl::Future<CompilationResult>> futures;
  futures.reserve(configs.size());
  for (int i = 0; i < configs.size(); ++i) {
    futures.push_back(tsl::MakeFutureOn(
        *executor, [&, config = std::move(configs[i])]() mutable {
          absl::StatusOr<std::unique_ptr<Executable>> executable =
              Compile(instr, config);
          return CompilationResult{std::move(config), std::move(executable)};
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
    const HloInstruction& instr) const {
  if (!executable.ok()) {
    return tsl::errors::CreateWithUpdatedMessage(
        executable.status(),
        absl::StrCat("Compilation failed: ", executable.status().message()));
  }

  bool allow_spills = false;
  if (options_.allow_reg_spills_fn) {
    allow_spills = options_.allow_reg_spills_fn(instr);
  }

  if (!allow_spills && *executable) {
    const auto spills_registers = [](const auto& pair) {
      const KernelStats& kernel_stats = pair.second;
      return kernel_stats.store_bytes_spilled > 0 ||
             kernel_stats.load_bytes_spilled > 0;
    };
    ModuleStats module_stats = (*executable)->module_stats();
    if (absl::c_any_of(module_stats, spills_registers)) {
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
