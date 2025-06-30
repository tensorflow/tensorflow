/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/autotuner/autotuner.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/executable.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

namespace {

tsl::Fprint128 GetFingerprint(const HloInstruction* instr) {
  auto options = HloPrintOptions::Fingerprint();
  options.set_print_backend_config(true);
  options.set_sort_backend_config(true);
  options.set_print_operand_shape(true);

  return tsl::Fingerprint128(instr->ToString(options));
}
}  // namespace

absl::StatusOr<std::unique_ptr<Autotuner>> Autotuner::Create(
    std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
    std::unique_ptr<Profiler> profiler, AutotuneConfig autotune_config,
    tsl::thread::ThreadPool* thread_pool) {
  if (codegen_backends.empty()) {
    return absl::InvalidArgumentError("No codegen backends provided");
  }
  return absl::WrapUnique(
      new Autotuner(std::move(codegen_backends), std::move(profiler),
                    std::move(autotune_config), thread_pool));
}

absl::Status Autotuner::Autotune(HloInstruction* instr) {
  VLOG(1) << "Autotuning HLO: " << instr->ToString();
  TF_ASSIGN_OR_RETURN(auto best_config, GetBestConfig(instr));
  CodegenBackend* best_codegen_backend = best_config.codegen_backend;
  return best_codegen_backend->ApplyConfig(*instr, *best_config.backend_config);
}

absl::StatusOr<Autotuner::Config> Autotuner::GetBestConfig(
    HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(std::vector<Config> supported_configs,
                      GetSupportedConfigs(instr));
  if (supported_configs.empty()) {
    return absl::InternalError("No supported configs found!");
  }
  VLOG(1) << "Found " << supported_configs.size() << " supported configs.";

  std::vector<absl::StatusOr<std::unique_ptr<Executable>>> executables =
      CompileAll(instr, supported_configs);

  std::vector<Config> valid_configs;
  std::vector<std::unique_ptr<Executable>> valid_executables;
  for (int i = 0; i < supported_configs.size(); ++i) {
    if (executables[i].ok()) {
      valid_configs.push_back(std::move(supported_configs[i]));
      valid_executables.push_back(std::move(executables[i].value()));
    }
  }
  VLOG(1) << "Successfully compiled " << valid_configs.size()
          << " configs out of " << supported_configs.size() << " configs.";

  TF_ASSIGN_OR_RETURN(
      std::vector<ProfileResult> results,
      profiler_->ProfileWithSharedBuffers(std::move(valid_executables)));
  absl::Duration min_duration = absl::InfiniteDuration();
  Config best_config{nullptr, nullptr};
  for (int i = 0; i < results.size(); ++i) {
    if (results[i].duration < min_duration) {
      min_duration = results[i].duration;
      best_config = std::move(valid_configs[i]);
    }
  }
  if (best_config.codegen_backend == nullptr) {
    return absl::InternalError("No valid config found!");
  }
  CHECK(best_config.backend_config != nullptr);
  return best_config;
}

Autotuner::InstructionsByFingerprint Autotuner::GetAutotuningCandidates(
    const HloModule* module, const InstructionFilterFn& should_autotune) {
  InstructionsByFingerprint instrunctions_by_fingerprint;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (should_autotune(*instr)) {
        instrunctions_by_fingerprint[GetFingerprint(instr)].push_back(instr);
      }
    }
  }
  return instrunctions_by_fingerprint;
}

absl::Status Autotuner::Autotune(HloModule* module,
                                 const InstructionFilterFn& should_autotune) {
  InstructionsByFingerprint instrunctions_by_fingerprint =
      GetAutotuningCandidates(module, should_autotune);
  if (instrunctions_by_fingerprint.empty()) {
    VLOG(1) << "No instructions to autotune.";
    return absl::OkStatus();
  }

  VLOG(1) << "Autotuning " << instrunctions_by_fingerprint.size()
          << " unique instructions.";
  for (auto& [_, instructions] : instrunctions_by_fingerprint) {
    CHECK(!instructions.empty());
    VLOG(1) << "Autotuning instruction:" << instructions[0]->ToString();
    TF_ASSIGN_OR_RETURN(Config best_config, GetBestConfig(instructions[0]));
    CodegenBackend* best_codegen_backend = best_config.codegen_backend;
    for (auto* instr : instructions) {
      TF_RETURN_IF_ERROR(best_codegen_backend->ApplyConfig(
          *instr, *best_config.backend_config));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Autotuner::Config>> Autotuner::GetSupportedConfigs(
    HloInstruction* instr) {
  std::vector<Config> configs;
  for (auto& codegen_backend : codegen_backends_) {
    std::vector<std::unique_ptr<BackendConfig>> per_backend_configs;
    TF_ASSIGN_OR_RETURN(per_backend_configs,
                        codegen_backend->GetSupportedConfigs(*instr));
    for (auto& config : per_backend_configs) {
      configs.push_back({codegen_backend.get(), std::move(config)});
    }
  }
  return configs;
}

std::vector<absl::StatusOr<std::unique_ptr<Executable>>> Autotuner::CompileAll(
    HloInstruction* instr, std::vector<Config>& configs) {
  if (thread_pool_ == nullptr) {
    std::vector<absl::StatusOr<std::unique_ptr<Executable>>> executables;
    executables.reserve(configs.size());
    for (auto& config : configs) {
      executables.emplace_back(
          config.codegen_backend->Compile(*instr, *config.backend_config));
    }
    return executables;
  }

  std::vector<absl::StatusOr<std::unique_ptr<Executable>>> executables(
      configs.size());
  tsl::BlockingCounter counter(configs.size());
  for (int i = 0; i < configs.size(); ++i) {
    auto compile_fn = [&, i]() {
      executables[i] = configs[i].codegen_backend->Compile(
          *instr, *configs[i].backend_config);
      counter.DecrementCount();
    };
    thread_pool_->Schedule(compile_fn);
  }
  counter.Wait();
  return executables;
}

}  // namespace xla
