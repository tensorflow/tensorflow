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
#include <optional>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/executable.h"
#include "xla/service/shaped_buffer.h"
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
    std::unique_ptr<AutotunerCacheInterface> cache,
    tsl::thread::ThreadPool* thread_pool) {
  if (codegen_backends.empty()) {
    return absl::InvalidArgumentError("No codegen backends provided");
  }
  return absl::WrapUnique(
      new Autotuner(std::move(codegen_backends), std::move(profiler),
                    std::move(autotune_config), std::move(cache), thread_pool));
}

absl::Status Autotuner::Autotune(HloInstruction* instr) {
  VLOG(1) << "Autotuning HLO: " << instr->ToString();
  TF_ASSIGN_OR_RETURN(auto best_config, GetBestConfig(instr));
  CodegenBackend* best_codegen_backend = best_config.codegen_backend;
  return best_codegen_backend->ApplyConfig(*instr, *best_config.backend_config);
}

absl::StatusOr<Autotuner::Config> Autotuner::GetBestConfig(
    HloInstruction* instr) {
  if (cache_) {
    auto cached_entry = cache_->Lookup(instr);
    if (cached_entry.has_value()) {
      VLOG(1) << "Found cached entry for HLO: " << instr->ToString();
      for (auto& codegen_backend : codegen_backends_) {
        if (codegen_backend->name() == cached_entry->codegen_backend()) {
          auto backend_config = std::make_unique<google::protobuf::Any>(
              cached_entry->backend_config());
          return Config{codegen_backend.get(), std::move(backend_config)};
        }
      }
      return absl::InternalError("Cached backend not found!");
    }
  }

  TF_ASSIGN_OR_RETURN(std::vector<Config> supported_configs,
                      GetSupportedConfigs(instr));
  if (supported_configs.empty()) {
    return absl::InternalError("No supported configs found!");
  }
  VLOG(1) << "Found " << supported_configs.size() << " supported configs.";

  std::vector<absl::StatusOr<std::unique_ptr<Executable>>> executables =
      CompileAll(instr, supported_configs);

  std::vector<ExecutableCandidate> executable_candidates;
  for (int i = 0; i < executables.size(); ++i) {
    if (!executables[i].ok()) {
      VLOG(2) << "Failed to compile config " << i << ": "
              << executables[i].status();
      continue;
    }
    executable_candidates.push_back(
        {std::move(supported_configs[i]), std::move(executables[i].value())});
  }
  VLOG(1) << "Successfully compiled " << executable_candidates.size()
          << " configs out of " << supported_configs.size() << " configs.";

  return ProfileAndPickBest(instr, executable_candidates);
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

absl::StatusOr<Autotuner::Config> Autotuner::ProfileAndPickBest(
    HloInstruction* instr, std::vector<ExecutableCandidate>& candidates) {
  if (candidates.empty()) {
    return absl::InternalError("No executables to profile!");
  }
  VLOG(1) << "Profiling " << candidates.size() << " executable candidates.";
  struct ConfigAndScratchBytes {
    Config* config;
    int scratch_bytes;
  };
  std::vector<ConfigAndScratchBytes> top_configs_and_scratch_bytes;
  Config* min_duration_config = nullptr;
  absl::Duration min_duration = absl::InfiniteDuration();

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<InputBuffers> input_buffers,
      profiler_->CreateInputBuffers(candidates[0].executable.get()));

  std::optional<ScopedShapedBuffer> reference_output;
  if (autotune_config_.check_buffers) {
    TF_ASSIGN_OR_RETURN(reference_output,
                        GetReferenceOutput(candidates, *input_buffers));
  }

  for (int i = 0; i < candidates.size(); ++i) {
    absl::StatusOr<ProfileResult> profile_result =
        profiler_->Profile(candidates[i].executable.get(), *input_buffers);
    if (!profile_result.ok()) {
      VLOG(2) << "Failed to profile config " << i << ": "
              << profile_result.status();
      continue;
    }
    VLOG(3) << "Config " << i << " ("
            << candidates[i].config.backend_config->ShortDebugString()
            << ") duration: " << profile_result.value().duration;

    if (autotune_config_.check_buffers) {
      CHECK(reference_output.has_value());
      CHECK(profile_result.value().output_buffer.has_value());
      absl::Status status = CheckBuffers(
          *input_buffers, profile_result.value().output_buffer.value(),
          reference_output.value());
      if (!status.ok()) {
        continue;
      }
    }

    absl::Duration duration = profile_result.value().duration;
    if (autotune_config_.optimize_scratch_bytes &&
        duration <
            min_duration + absl::Microseconds(
                               autotune_config_.scratch_bytes_window_size_us)) {
      top_configs_and_scratch_bytes.push_back(
          {&candidates[i].config, profile_result.value().scratch_bytes});
    }
    if (duration < min_duration) {
      min_duration = duration;
      min_duration_config = &candidates[i].config;
    }
  }
  if (min_duration_config == nullptr) {
    return absl::InternalError("No valid config found!");
  }

  Config* best_config = min_duration_config;
  if (autotune_config_.optimize_scratch_bytes) {
    Config* best_scratch_bytes_config = nullptr;
    int min_scratch_bytes = -1;
    for (auto& config_and_scratch : top_configs_and_scratch_bytes) {
      if (best_scratch_bytes_config == nullptr ||
          config_and_scratch.scratch_bytes < min_scratch_bytes) {
        best_scratch_bytes_config = config_and_scratch.config;
        min_scratch_bytes = config_and_scratch.scratch_bytes;
      }
    }
    if (best_scratch_bytes_config != nullptr) {
      best_config = best_scratch_bytes_config;
    }
  }

  AutotunerCacheEntry cache_entry;
  cache_entry.set_codegen_backend(min_duration_config->codegen_backend->name());
  *cache_entry.mutable_backend_config() = *best_config->backend_config;
  if (cache_) {
    TF_RETURN_IF_ERROR(cache_->Insert(instr, cache_entry));
  }
  return std::move(*best_config);
}

absl::StatusOr<ScopedShapedBuffer> Autotuner::GetReferenceOutput(
    std::vector<ExecutableCandidate>& candidates, InputBuffers& input_buffers) {
  for (auto& candidate : candidates) {
    if (candidate.config.codegen_backend->CanProduceWrongResults()) {
      continue;
    }
    absl::StatusOr<ProfileResult> profile_result =
        profiler_->Profile(candidate.executable.get(), input_buffers);
    if (!profile_result.ok()) {
      continue;
    }
    if (profile_result.value().output_buffer.has_value()) {
      return std::move(profile_result.value().output_buffer.value());
    }
  }
  return absl::InternalError("No reference output found!");
}

absl::Status Autotuner::CheckBuffers(InputBuffers& input_buffers,
                                     ScopedShapedBuffer& output_buffer,
                                     ScopedShapedBuffer& reference_output) {
  absl::Status status = profiler_->CheckInputBuffers(input_buffers);
  if (!status.ok()) {
    VLOG(2) << "Input buffers check failed: " << status;
    CHECK(!autotune_config_.crash_on_check_failure);
    return status;
  }
  status = profiler_->CheckOutputBuffer(output_buffer, reference_output,
                                        autotune_config_.relative_tolerance);
  if (!status.ok()) {
    VLOG(2) << "Output buffers check failed: " << status;
    return status;
  }
  return absl::OkStatus();
}

}  // namespace xla
