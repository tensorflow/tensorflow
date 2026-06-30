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
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/config_runner.h"
#include "xla/backends/autotuner/config_selector.h"
#include "xla/backends/autotuner/hlo_extractor.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/future.h"
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
    ASSIGN_OR_RETURN(auto runner,
                     ConfigRunner::Create(std::move(profiler),
                                          options.correctness_check_options));
    config_runners.push_back(std::move(runner));
  }

  return absl::WrapUnique(new Autotuner(std::move(orchestrator),
                                        std::move(config_runners),
                                        std::move(options), thread_pool));
}

Autotuner::Autotuner(
    absl_nonnull std::unique_ptr<CodegenOrchestrator> orchestrator,
    std::vector<absl_nonnull std::unique_ptr<ConfigRunner>> runners,
    Options options, tsl::thread::ThreadPool* thread_pool)
    : options_(std::move(options)),
      orchestrator_(std::move(orchestrator)),
      runners_(std::move(runners)),
      thread_pool_(thread_pool) {}

absl::StatusOr<std::vector<Autotuner::TuningResult>> Autotuner::TuneConfigs(
    const HloModule& module, const InstructionFilterFn& should_autotune,
    bool tolerate_no_supported_configs) const {
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

  tsl::Executor* executor = thread_pool_ != nullptr
                                ? thread_pool_->AsExecutor()
                                : &tsl::InlineExecutor::Instance();

  const int num_runners = runners_.size();
  for (int i = 0; i < instruction_groups.size(); ++i) {
    const EquivalentInstructions& group = instruction_groups[i];
    TF_RET_CHECK(!group.empty()) << "Instruction group cannot be empty.";
    const HloInstruction* leader = group.front();
    leaders.push_back(leader);
    int runner_index = i % num_runners;
    future_configs.push_back(
        tsl::MakeFutureOn(*executor, [this, leader, runner_index]() {
          return GetTunedConfig(leader, runner_index).Await();
        }));
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

    if (tolerate_no_supported_configs && absl::IsNotFound(config_or.status())) {
      VLOG(1) << "Tolerating autotuning failure for instruction group " << i
              << ": " << config_or.status();
      continue;
    }

    LOG(ERROR) << "Autotuning failed for instruction group " << i << ": "
               << config_or.status();
    combined_status.Update(config_or.status());
  }
  RETURN_IF_ERROR(combined_status);
  return tuning_results;
}

tsl::Future<Autotuner::Config> Autotuner::GetTunedConfig(
    const HloInstruction* absl_nonnull instr, int runner_index) const {
  ASSIGN_OR_RETURN(std::vector<CodegenOrchestrator::Config> supported_configs,
                   orchestrator_->GetSupportedConfigs(*instr));
  if (supported_configs.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "No supported configs found for HLO: ", instr->ToString()));
  }

  tsl::Future<std::vector<CodegenOrchestrator::MaybeExecutableCandidate>>
      maybe_candidates =
          orchestrator_->CompileAll(*instr, std::move(supported_configs));

  return std::move(maybe_candidates)
      .Map([instr, runner_index,
            this](std::vector<CodegenOrchestrator::MaybeExecutableCandidate>
                      maybe_candidates) mutable -> absl::StatusOr<Config> {
        std::vector<ConfigRunner::ExecutableCandidate> candidates;
        for (auto& maybe_candidate : maybe_candidates) {
          if (maybe_candidate.executable.ok()) {
            candidates.push_back(
                {std::move(maybe_candidate.config),
                 std::move(maybe_candidate.executable.value())});
          }
        }

        if (candidates.empty()) {
          return absl::InternalError("No candidates could be compiled.");
        }

        ASSIGN_OR_RETURN(
            std::vector<ConfigRunner::ConfigProfile> profiles,
            runners_[runner_index]->ProfileAll(std::move(candidates), instr));

        TF_RET_CHECK(!profiles.empty())
            << "No configs could be profiled." << instr->ToString();

        ASSIGN_OR_RETURN(
            ConfigRunner::ConfigProfile best_profile,
            PickBestConfig(profiles, options_.scratch_bytes_window_size_us));

        return std::move(best_profile.config);
      });
}

}  // namespace xla
