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

#include "xla/backends/autotuner/autotuner.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/config_runner.h"
#include "xla/backends/autotuner/config_selector.h"
#include "xla/backends/autotuner/hlo_extractor.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {

absl::StatusOr<std::unique_ptr<Autotuner>> Autotuner::Create(
    absl_nonnull std::unique_ptr<CodegenOrchestrator> orchestrator,
    std::vector<absl_nonnull std::unique_ptr<Profiler>> profilers,
    Options options, tsl::thread::ThreadPool* thread_pool) {
  std::vector<absl_nonnull std::unique_ptr<ConfigRunner>> config_runners;
  TF_RET_CHECK(!profilers.empty())
      << "At least one profiler is required to create an Autotuner.";
  TF_RET_CHECK(orchestrator != nullptr)
      << "CodegenOrchestrator is required to create an Autotuner.";
  config_runners.reserve(profilers.size());
  for (auto& profiler : profilers) {
    ABSL_ASSIGN_OR_RETURN(config_runners.emplace_back(),
                     ConfigRunner::Create(std::move(profiler),
                                          options.correctness_check_options));
  }

  CodegenOrchestrator* orchestrator_ptr = orchestrator.get();
  return absl::WrapUnique(new Autotuner(
      std::move(orchestrator), *orchestrator_ptr, std::move(config_runners),
      std::move(options), thread_pool));
}

absl::StatusOr<std::unique_ptr<Autotuner>> Autotuner::Create(
    CodegenOrchestrator& orchestrator,
    std::vector<absl_nonnull std::unique_ptr<Profiler>> profilers,
    Options options, tsl::thread::ThreadPool* thread_pool) {
  std::vector<absl_nonnull std::unique_ptr<ConfigRunner>> config_runners;
  TF_RET_CHECK(!profilers.empty())
      << "At least one profiler is required to create an Autotuner.";
  config_runners.reserve(profilers.size());
  for (auto& profiler : profilers) {
    ABSL_ASSIGN_OR_RETURN(config_runners.emplace_back(),
                     ConfigRunner::Create(std::move(profiler),
                                          options.correctness_check_options));
  }

  return absl::WrapUnique(new Autotuner(nullptr, orchestrator,
                                        std::move(config_runners),
                                        std::move(options), thread_pool));
}

Autotuner::Autotuner(
    std::unique_ptr<CodegenOrchestrator> owned_orchestrator,
    CodegenOrchestrator& orchestrator,
    std::vector<absl_nonnull std::unique_ptr<ConfigRunner>> runners,
    Options options, tsl::thread::ThreadPool* thread_pool)
    : options_(std::move(options)),
      owned_orchestrator_(std::move(owned_orchestrator)),
      orchestrator_(&orchestrator),
      runners_(std::move(runners)),
      thread_pool_(thread_pool) {}

absl::StatusOr<std::vector<Autotuner::TuningResult>> Autotuner::TuneConfigs(
    const HloModule& module, const InstructionFilterFn& should_autotune) const {
  std::vector<EquivalentInstructions> instruction_groups =
      ExtractEquivalentInstructions(module, should_autotune);
  if (instruction_groups.empty()) {
    VLOG(1) << "No instructions to autotune.";
    return std::vector<TuningResult>{};
  }

  VLOG(1) << "Autotuning " << instruction_groups.size()
          << " unique HLO instruction groups.";

  std::vector<tsl::Future<Config>> future_configs;
  std::vector<const HloInstruction*> leaders;
  leaders.reserve(instruction_groups.size());

  const int num_runners = runners_.size();
  for (int i = 0; i < instruction_groups.size(); ++i) {
    const EquivalentInstructions& group = instruction_groups[i];
    TF_RET_CHECK(!group.empty()) << "Instruction group cannot be empty.";
    const HloInstruction* leader = group.front();
    leaders.push_back(leader);
    int runner_index = i % num_runners;
    future_configs.push_back(GetTunedConfig(leader, runner_index));
  }

  // Await and verify all configuration selections.
  std::vector<TuningResult> tuning_results;
  absl::Status combined_status = absl::OkStatus();
  for (int i = 0; i < future_configs.size(); ++i) {
    absl::StatusOr<Config> config_or = std::move(future_configs[i]).Await();
    if (config_or.ok()) {
      tuning_results.push_back(TuningResult{leaders[i], std::move(*config_or)});
      continue;
    }

    LOG(ERROR) << "Autotuning failed for instruction group " << i << ": "
               << config_or.status();
    combined_status.Update(config_or.status());
  }
  ABSL_RETURN_IF_ERROR(combined_status);
  return tuning_results;
}

tsl::Future<Autotuner::Config> Autotuner::GetTunedConfig(
    const HloInstruction* absl_nonnull instr) const {
  // TODO(b/521833070): Use next available runner rather than always using the
  // first one.
  return GetTunedConfig(instr, 0);
}

tsl::Future<Autotuner::Config> Autotuner::GetTunedConfig(
    const HloInstruction* absl_nonnull instr, int runner_index) const {
  ABSL_ASSIGN_OR_RETURN(std::vector<CodegenOrchestrator::Config> supported_configs,
                   orchestrator_->GetSupportedConfigs(*instr));
  if (supported_configs.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "No supported configs found for HLO: ", instr->ToString()));
  }

  if (supported_configs.size() == 1) {
    VLOG(1) << "Found only one supported config: "
            << supported_configs[0].ToString();
    return std::move(supported_configs[0]);
  }

  tsl::Future<std::vector<CodegenOrchestrator::MaybeExecutableCandidate>>
      maybe_candidates = orchestrator_->CompileAll(
          *instr, std::move(supported_configs), thread_pool_);

  return std::move(maybe_candidates)
      .Map([instr, runner_index,
            this](std::vector<CodegenOrchestrator::MaybeExecutableCandidate>
                      maybe_candidates) mutable -> absl::StatusOr<Config> {
        std::vector<ConfigRunner::ExecutableCandidate> candidates;
        std::vector<ConfigRunner::ConfigProfile> compilation_failures;
        for (auto& maybe_candidate : maybe_candidates) {
          if (maybe_candidate.executable.ok()) {
            candidates.push_back(
                {std::move(maybe_candidate.config),
                 std::move(maybe_candidate.executable.value())});
          } else {
            compilation_failures.push_back(
                {std::move(maybe_candidate.config),
                 ConfigRunner::Failure{
                     ConfigRunner::FailureKind::kCompilationFailed,
                     maybe_candidate.executable.status().ToString()}});
          }
        }

        if (candidates.empty()) {
          LogConfigProfiles(*instr, {}, compilation_failures);
          return absl::InternalError("No candidates could be compiled.");
        }

        if (candidates.size() == 1) {
          LogConfigProfiles(*instr, {}, compilation_failures);
          VLOG(1) << "Using the only compilable config: "
                  << candidates[0].config.ToString();
          return std::move(candidates[0].config);
        }

        ABSL_ASSIGN_OR_RETURN(
            std::vector<ConfigRunner::ConfigProfile> profiles,
            runners_[runner_index]->ProfileAll(std::move(candidates), instr));

        LogConfigProfiles(*instr, profiles, compilation_failures);

        TF_RET_CHECK(!profiles.empty())
            << "No configs could be profiled." << instr->ToString();

        ABSL_ASSIGN_OR_RETURN(
            ConfigRunner::ConfigProfile best_profile,
            PickBestConfig(profiles, options_.scratch_bytes_window_size_us));

        return std::move(best_profile.config);
      });
}

void Autotuner::LogConfigProfiles(
    const HloInstruction& instr,
    absl::Span<const ConfigRunner::ConfigProfile> profiles,
    absl::Span<const ConfigRunner::ConfigProfile> compilation_failures) const {
  for (const auto& profile : profiles) {
    VLOG(2) << profile.ToString(/*verbose=*/VLOG_IS_ON(3));
  }
  for (const auto& result : compilation_failures) {
    VLOG(2) << result.ToString(/*verbose=*/VLOG_IS_ON(3));
  }

  if (options_.dump_logs_to.empty()) {
    return;
  }

  AutotuningLog log;
  log.mutable_instr()->PackFrom(instr.ToProto());
  for (const auto& profile : profiles) {
    *log.add_results() = profile.ToProto();
  }
  for (const auto& failed_config : compilation_failures) {
    *log.add_results() = failed_config.ToProto();
  }
  absl::MutexLock lock(logs_mutex_);
  *logs_.add_logs() = std::move(log);
}

absl::Status Autotuner::DumpTuningLogs() {
  if (options_.dump_logs_to.empty()) {
    return absl::OkStatus();
  }

  AutotuningLogs logs_to_dump;
  {
    absl::MutexLock lock(logs_mutex_);
    if (logs_.logs().empty()) {
      return absl::OkStatus();
    }
    logs_to_dump.Swap(&logs_);
  }

  std::string textproto;
  if (!tsl::protobuf::TextFormat::PrintToString(logs_to_dump, &textproto)) {
    return absl::InternalError(
        "Failed to convert AutotuningLogs to textproto.");
  }

  ABSL_RETURN_IF_ERROR(tsl::AppendStringToFile(tsl::Env::Default(),
                                          options_.dump_logs_to, textproto));
  VLOG(1) << "Autotune logs appended to file: " << options_.dump_logs_to;
  return absl::OkStatus();
}

}  // namespace xla
