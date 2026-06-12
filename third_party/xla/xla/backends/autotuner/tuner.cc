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

#include "xla/backends/autotuner/tuner.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "google/protobuf/text_format.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/shaped_buffer.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/proto/proto_utils.h"
#include "xla/util.h"

namespace xla {

absl::StatusOr<std::unique_ptr<Tuner>> Tuner::Create(
    std::unique_ptr<Profiler> profiler,
    CodegenOrchestrator* absl_nonnull orchestrator, Options options) {
  if (orchestrator == nullptr) {
    return absl::InvalidArgumentError(
        "Tuner initialization failed. CodegenOrchestrator is null.");
  }
  if (profiler == nullptr) {
    return absl::InvalidArgumentError(
        "Tuner initialization failed. Profiler is null.");
  }
  return absl::WrapUnique(
      new Tuner(std::move(profiler), orchestrator, std::move(options)));
}

tsl::Future<Tuner::Config> Tuner::GetTunedConfig(const HloInstruction* instr) {
  ASSIGN_OR_RETURN(std::vector<CodegenOrchestrator::Config> supported_configs,
                   orchestrator_->GetSupportedConfigs(*instr));
  std::cerr << "DEBUG AUTOTUNER: supported_configs size = "
            << supported_configs.size() << " for instruction: " << instr->name()
            << std::endl;
  if (supported_configs.empty()) {
    return absl::InternalError(
        absl::StrCat("Autotuning failed for HLO: ", instr->ToString(),
                     ". No supported configs found for this instruction."));
  }
  if (supported_configs.size() == 1) {
    VLOG(1) << "Found only one supported config: "
            << supported_configs[0].ToString();
    std::vector<ConfigResult> results;
    results.push_back({/*config=*/std::move(supported_configs[0]),
                       /*failure=*/std::nullopt,
                       /*duration=*/absl::ZeroDuration(),
                       /*scratch_bytes=*/0});
    LogConfigResults(*instr, results);
    return std::move(results.back().config);
  }
  VLOG(1) << "Found total of " << supported_configs.size()
          << " supported configs.";

  auto compile_results =
      orchestrator_->CompileAll(*instr, std::move(supported_configs));
  return std::move(compile_results)
      .Map(
          [instr, this](std::vector<CodegenOrchestrator::CompilationResult>&&
                            compile_results) mutable -> absl::StatusOr<Config> {
            std::vector<ExecutableCandidate> executable_candidates;
            std::vector<ConfigResult> results;
            for (auto& [config, executable] : compile_results) {
              if (!executable.ok()) {
                std::cerr << "DEBUG AUTOTUNER compile failure: "
                          << executable.status().ToString()
                          << " for config: " << config.ToString() << std::endl;
                results.push_back({/*config=*/std::move(config),
                                   /*failure=*/
                                   Failure{FailureKind::kCompilationFailed,
                                           executable.status().ToString()},
                                   /*duration=*/absl::ZeroDuration(),
                                   /*scratch_bytes=*/0});
              } else {
                executable_candidates.push_back(
                    {std::move(config), std::move(*executable)});
              }
            }
            if (executable_candidates.empty()) {
              LogConfigResults(*instr, results);
              return absl::InternalError(
                  absl::StrCat("Autotuning failed for HLO: ", instr->ToString(),
                               ". No configs could be compiled."));
            }
            VLOG(1) << "Successfully compiled " << executable_candidates.size()
                    << " configs out of " << compile_results.size()
                    << " configs with " << results.size()
                    << " compilation failures.";

            if (executable_candidates.size() == 1) {
              VLOG(1) << "Skipping profiling and using the only config: "
                      << executable_candidates[0].config.ToString();
              results.push_back(
                  {/*config=*/std::move(executable_candidates[0].config),
                   /*failure=*/std::nullopt, /*duration=*/absl::ZeroDuration(),
                   /*scratch_bytes=*/0});
              LogConfigResults(*instr, results);
              return std::move(results.back().config);
            }

            auto profile_results_or =
                ProfileAll(std::move(executable_candidates), instr);
            if (!profile_results_or.ok()) {
              return absl::InternalError(
                  absl::StrCat("Autotuning failed for HLO: ", instr->ToString(),
                               ". Failed to profile configs: ",
                               profile_results_or.status().message()));
            }
            std::vector<ConfigResult> profile_results =
                std::move(profile_results_or).value();
            results.insert(results.end(),
                           std::make_move_iterator(profile_results.begin()),
                           std::make_move_iterator(profile_results.end()));
            LogConfigResults(*instr, results);
            absl::StatusOr<ConfigResult> best_result = PickBestConfig(results);
            if (!best_result.ok()) {
              return absl::InternalError(
                  absl::StrCat("Autotuning failed for HLO: ", instr->ToString(),
                               ". Failed to pick best config: ",
                               best_result.status().message()));
            }
            VLOG(1) << "Picked best config: " << best_result->ToString();
            return std::move(best_result->config);
          });
}

absl::StatusOr<std::vector<Tuner::ConfigResult>> Tuner::ProfileAll(
    std::vector<ExecutableCandidate> candidates, const HloInstruction* instr) {
  std::vector<ConfigResult> config_results(candidates.size());

  absl::MutexLock lock(profiler_m_);

  ASSIGN_OR_RETURN(
      std::unique_ptr<InputBuffers> input_buffers,
      profiler_->CreateInputBuffers(candidates[0].executable.get(), instr));

  const auto is_trusted = [](const ExecutableCandidate& candidate) {
    return !candidate.config.codegen_backend->CanProduceWrongResults();
  };

  std::vector<OutputCluster> clusters;
  std::vector<int> profile_order(candidates.size());
  std::iota(profile_order.begin(), profile_order.end(), 0);

  auto first_untrusted = profile_order.begin();
  if (options_.check_buffers) {
    first_untrusted =
        std::stable_partition(profile_order.begin(), profile_order.end(),
                              [&](int i) { return is_trusted(candidates[i]); });

    VLOG(2) << "Validating outputs via clustering across " << candidates.size()
            << " config(s).";
  }

  for (auto it = profile_order.begin(); it != first_untrusted; ++it) {
    const int i = *it;
    config_results[i] =
        ProfileCandidate(std::move(candidates[i]), *input_buffers, clusters,
                         /*is_trusted_config=*/true,
                         /*allow_new_cluster=*/true);
  }

  const bool has_trusted_reference = !clusters.empty();
  for (auto it = first_untrusted; it != profile_order.end(); ++it) {
    const int i = *it;
    config_results[i] =
        ProfileCandidate(std::move(candidates[i]), *input_buffers, clusters,
                         /*is_trusted_config=*/false,
                         /*allow_new_cluster=*/!has_trusted_reference);
  }

  if (options_.check_buffers) {
    DemoteNonWinningClusterConfigs(config_results, clusters);
    if (options_.crash_on_check_failure) {
      for (const ConfigResult& r : config_results) {
        CHECK(!r.failure.has_value() ||
              (r.failure->kind != FailureKind::kRedzoneCheckFailed &&
               r.failure->kind != FailureKind::kWrongResults))
            << "crash_on_check_failure: " << r.failure->ToString();
      }
    }
  }
  candidates.clear();
  return config_results;
}

Tuner::ConfigResult Tuner::ProfileCandidate(
    ExecutableCandidate candidate, InputBuffers& input_buffers,
    std::vector<OutputCluster>& clusters, bool is_trusted_config,
    bool allow_new_cluster) {
  absl::StatusOr<ProfileResult> profile_result =
      profiler_->Profile(candidate.executable.get(), input_buffers);

  if (!profile_result.ok()) {
    return ConfigResult{
        /*config=*/std::move(candidate.config),
        /*failure=*/
        Failure{FailureKind::kExecutionFailed,
                profile_result.status().ToString()},
    };
  }

  if (!options_.check_buffers) {
    return ConfigResult{/*config=*/std::move(candidate.config),
                        /*failure=*/std::nullopt,
                        /*duration=*/profile_result->duration,
                        /*scratch_bytes=*/profile_result->scratch_bytes};
  }

  absl::Status redzone = profiler_->CheckInputBuffers(input_buffers);
  if (!redzone.ok()) {
    return ConfigResult{
        /*config=*/std::move(candidate.config),
        /*failure=*/
        Failure{FailureKind::kRedzoneCheckFailed, redzone.ToString()},
    };
  }

  CHECK(profile_result->output_buffer.has_value());
  int assigned_cluster =
      AssignToOutputCluster(clusters, profile_result->output_buffer.value(),
                            is_trusted_config, allow_new_cluster);

  return ConfigResult{/*config=*/std::move(candidate.config),
                      /*failure=*/std::nullopt,
                      /*duration=*/profile_result->duration,
                      /*scratch_bytes=*/profile_result->scratch_bytes,
                      /*cluster_index=*/assigned_cluster};
}

absl::StatusOr<Tuner::ConfigResult> Tuner::PickBestConfig(
    std::vector<ConfigResult>& results) {
  absl::Duration min_duration = absl::InfiniteDuration();
  ConfigResult* best_result = nullptr;
  std::vector<std::string> failures;
  for (ConfigResult& result : results) {
    if (result.failure.has_value()) {
      failures.push_back(result.failure->ToString());
    } else if (result.duration < min_duration) {
      min_duration = result.duration;
      best_result = &result;
    }
  }

  if (best_result == nullptr) {
    std::string message = "All configs failed during profiling.";
    if (!failures.empty()) {
      absl::StrAppend(&message, "\nFailures (", failures.size(), "):\n",
                      absl::StrJoin(failures, "\n"));
    }
    return absl::NotFoundError(message);
  }

  const ConfigResult* fastest_result = best_result;
  int64_t min_scratch_bytes = std::numeric_limits<int64_t>::max();
  absl::Duration duration_limit =
      min_duration + absl::Microseconds(options_.scratch_bytes_window_size_us);
  absl::Duration min_duration_with_optimzed_scratch_bytes =
      absl::InfiniteDuration();
  for (ConfigResult& result : results) {
    if (!result.failure.has_value() && result.duration <= duration_limit) {
      bool current_result_is_better =
          result.scratch_bytes < min_scratch_bytes ||
          (result.scratch_bytes == min_scratch_bytes &&
           result.duration < min_duration_with_optimzed_scratch_bytes);
      if (current_result_is_better) {
        min_scratch_bytes = result.scratch_bytes;
        min_duration_with_optimzed_scratch_bytes = result.duration;
        best_result = &result;
      }
    }
  }
  if (best_result != fastest_result) {
    VLOG(2) << "Autotuner picked a slower config to save scratch memory. "
            << "Fastest config: " << fastest_result->ToString() << ". "
            << "Selected config: " << best_result->ToString() << ". "
            << "Tolerance: " << options_.scratch_bytes_window_size_us << "us.";
  }

  return std::move(*best_result);
}

int Tuner::AssignToOutputCluster(std::vector<OutputCluster>& clusters,
                                 ScopedShapedBuffer& output,
                                 bool is_trusted_config,
                                 bool allow_new_cluster) {
  for (int c = 0; c < clusters.size(); ++c) {
    if (profiler_
            ->CheckOutputBuffer(output, clusters[c].representative,
                                options_.relative_tolerance)
            .ok()) {
      clusters[c].count++;
      clusters[c].has_trusted_member |= is_trusted_config;
      return c;
    }
  }
  if (!allow_new_cluster) {
    return -1;
  }
  clusters.push_back(OutputCluster{std::move(output), /*count=*/1,
                                   /*has_trusted_member=*/is_trusted_config});
  return clusters.size() - 1;  // Return the index of the new cluster.
}

void Tuner::DemoteNonWinningClusterConfigs(
    std::vector<ConfigResult>& results,
    const std::vector<OutputCluster>& clusters) {
  if (clusters.empty()) {
    return;
  }
  const bool any_trusted = absl::c_any_of(
      clusters, [](const OutputCluster& c) { return c.has_trusted_member; });
  int winner = -1;
  for (int c = 0; c < clusters.size(); ++c) {
    if (any_trusted && !clusters[c].has_trusted_member) {
      continue;
    }
    if (winner < 0 || clusters[c].count > clusters[winner].count) {
      winner = c;
    }
  }
  VLOG(2) << "Output clustering formed " << clusters.size()
          << " cluster(s); selected cluster " << winner << " with "
          << clusters[winner].count
          << " member(s), trusted=" << clusters[winner].has_trusted_member;
  for (ConfigResult& result : results) {
    if (!result.failure.has_value() && result.cluster_index != winner) {
      result.failure = Failure{
          FailureKind::kWrongResults,
          absl::StrCat("Output disagrees with winning cluster (member of "
                       "cluster ",
                       result.cluster_index, " of ", clusters.size(),
                       "; winning cluster has ", clusters[winner].count,
                       " member(s), trusted=",
                       clusters[winner].has_trusted_member, ").")};
      VLOG(3) << "Demoted config " << result.config.ToString() << " (cluster "
              << result.cluster_index << ").";
    }
  }
}

void Tuner::LogConfigResults(const HloInstruction& instr,
                             absl::Span<const ConfigResult> results) {
  for (const ConfigResult& result : results) {
    VLOG(2) << result.ToString(/*verbose=*/VLOG_IS_ON(3));
  }
  if (options_.dump_logs_to.empty()) {
    return;
  }
  AutotuningLog log;
  log.mutable_instr()->PackFrom(instr.ToProto());
  for (const ConfigResult& result : results) {
    *log.add_results() = result.ToProto();
  }
  *logs_.add_logs() = std::move(log);
}

absl::Status Tuner::DumpLogsToFile() {
  if (options_.dump_logs_to.empty()) {
    return absl::OkStatus();
  }

  std::string textproto;
  tsl::protobuf::TextFormat::PrintToString(logs_, &textproto);

  RETURN_IF_ERROR(tsl::AppendStringToFile(tsl::Env::Default(),
                                          options_.dump_logs_to, textproto));
  VLOG(1) << "Autotune logs appended to file: " << options_.dump_logs_to;
  logs_.Clear();
  return absl::OkStatus();
}

std::string Tuner::Failure::ToString() const {
  absl::string_view kind_str;
  switch (kind) {
    case FailureKind::kCompilationFailed:
      kind_str = "COMPILATION FAILED";
      break;
    case FailureKind::kExecutionFailed:
      kind_str = "EXECUTION FAILED";
      break;
    case FailureKind::kRedzoneCheckFailed:
      kind_str = "REDZONE CHECK FAILED";
      break;
    case FailureKind::kWrongResults:
      kind_str = "WRONG RESULTS";
      break;
  }
  return absl::StrFormat("%s: %s", kind_str, message);
}

std::string Tuner::ConfigResult::ToString(bool verbose) const {
  std::string config_str =
      absl::StrFormat("%s : %s", config.codegen_backend->name(),
                      verbose ? config.backend_config->ShortDebugString() : "");
  if (failure.has_value()) {
    absl::StrAppend(&config_str, " ", failure->ToString());
  }
  return absl::StrFormat("{%s duration: %s, scratch_bytes: %d}", config_str,
                         absl::FormatDuration(duration), scratch_bytes);
}

AutotuneResult::FailureResult Tuner::Failure::ToProto() const {
  AutotuneResult::FailureResult failure_proto;
  switch (kind) {
    case FailureKind::kCompilationFailed:
      failure_proto.set_kind(AutotuneResult::DISQUALIFIED);
      break;
    case FailureKind::kExecutionFailed:
      failure_proto.set_kind(AutotuneResult::DISQUALIFIED);
      break;
    case FailureKind::kRedzoneCheckFailed:
      failure_proto.set_kind(AutotuneResult::REDZONE_MODIFIED);
      break;
    case FailureKind::kWrongResults:
      failure_proto.set_kind(AutotuneResult::WRONG_RESULT);
      break;
  }
  failure_proto.set_msg(message);
  return failure_proto;
}

AutotuneResult Tuner::ConfigResult::ToProto() const {
  AutotuneResult result;
  if (config.backend_config->has_gemm()) {
    *result.mutable_gemm() = config.backend_config->gemm();
  } else if (config.backend_config->has_triton()) {
    *result.mutable_triton() = config.backend_config->triton();
  } else if (config.backend_config->has_algorithm()) {
    *result.mutable_algorithm() = config.backend_config->algorithm();
  } else {
    result.mutable_other()->set_name(config.codegen_backend->name());
    result.mutable_other()->mutable_config()->PackFrom(*config.backend_config);
  }
  if (failure.has_value()) {
    *result.mutable_failure() = failure->ToProto();
  }
  *result.mutable_run_time() = tsl::proto_utils::ToDurationProto(duration);
  result.set_scratch_bytes(scratch_bytes);
  return result;
}

}  // namespace xla
