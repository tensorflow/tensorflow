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

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/executable.h"
#include "xla/service/shaped_buffer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/proto/proto_utils.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/protobuf.h"

namespace xla {

namespace {

tsl::Fprint128 GetFingerprint(const HloInstruction* instr) {
  auto options = HloPrintOptions::Fingerprint();
  options.set_print_backend_config(true);
  options.set_sort_backend_config(true);
  options.set_print_operand_shape(true);

  return tsl::Fingerprint128(instr->ToString(options));
}

// Returns ShortDebugString of contents of Any proto, without type URL.
std::string UnpackedAnyShortDebugString(const google::protobuf::Any& any) {
  std::string s = any.ShortDebugString();
  // Any is serialized as "go/debugonly [type/url] {<serialized_proto>}".
  std::string type_url = absl::StrCat(" [", any.type_url(), "] ");
  absl::StrReplaceAll({{type_url, ""}}, &s);
  return s;
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
    TF_ASSIGN_OR_RETURN(Config best_config,
                        GetCachedOrTuneBestConfig(instructions[0]));
    CodegenBackend* best_codegen_backend = best_config.codegen_backend;
    for (auto* instr : instructions) {
      TF_RETURN_IF_ERROR(best_codegen_backend->ApplyConfig(
          *instr, *best_config.backend_config));
    }
  }
  return DumpLogsToFile();
}

absl::Status Autotuner::Autotune(HloInstruction* instr) {
  VLOG(1) << "Autotuning HLO: " << instr->ToString();
  TF_ASSIGN_OR_RETURN(Config best_config, GetCachedOrTuneBestConfig(instr));
  CodegenBackend* best_codegen_backend = best_config.codegen_backend;
  TF_RETURN_IF_ERROR(
      best_codegen_backend->ApplyConfig(*instr, *best_config.backend_config));
  return DumpLogsToFile();
}

absl::StatusOr<Autotuner::Config> Autotuner::GetCachedOrTuneBestConfig(
    HloInstruction* instr) {
  std::optional<Config> cached_config = LookUp(instr);
  Config best_config;
  if (cached_config.has_value()) {
    best_config = std::move(*cached_config);
  } else {
    if (autotune_config_.expect_all_instructions_in_cache) {
      return absl::NotFoundError("No cached config found for HLO instr: " +
                                 instr->ToString());
    }
    TF_ASSIGN_OR_RETURN(best_config, TuneBestConfig(instr));
    Insert(instr, best_config);
  }
  return best_config;
}

absl::StatusOr<Autotuner::Config> Autotuner::TuneBestConfig(
    HloInstruction* instr) {
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
    if (executables[i].ok()) {
      executable_candidates.push_back(
          {std::move(supported_configs[i]), std::move(executables[i].value())});
    } else {
      VLOG(4) << "Compilation failed for config "
              << supported_configs[i].codegen_backend->name() << " : "
              << UnpackedAnyShortDebugString(
                     *supported_configs[i].backend_config)
              << " with status: " << executables[i].status();
    }
  }

  if (executable_candidates.empty()) {
    return absl::InternalError("No executable candidates to profile!");
  }
  VLOG(1) << "Successfully compiled " << executable_candidates.size()
          << " configs out of " << supported_configs.size() << " configs.";

  TF_ASSIGN_OR_RETURN(std::vector<ConfigResult> results,
                      ProfileAll(executable_candidates));
  LogConfigResults(*instr, results);
  TF_ASSIGN_OR_RETURN(auto best_result, PickBestConfig(results));
  VLOG(1) << "Picked best config: " << best_result.ToString();
  return std::move(best_result.config);
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

std::optional<Autotuner::Config> Autotuner::LookUp(
    const HloInstruction* instr) {
  if (cache_) {
    auto cached_config = cache_->Lookup(instr);
    if (cached_config.has_value()) {
      VLOG(1) << "Found cached config for HLO: " << instr->ToString();
      for (auto& codegen_backend : codegen_backends_) {
        if (codegen_backend->name() == cached_config->codegen_backend_name) {
          auto backend_config = std::make_unique<google::protobuf::Any>(
              cached_config->backend_config);
          return Config{codegen_backend.get(), std::move(backend_config)};
        }
      }
      LOG(WARNING) << "Cached config for HLO: " << instr->ToString()
                   << " has unsupported backend "
                   << cached_config->codegen_backend_name;
    }
  }
  return std::nullopt;
}

void Autotuner::Insert(const HloInstruction* instr, Autotuner::Config& config) {
  if (cache_) {
    AutotunerCacheInterface::Config cached_config;
    cached_config.codegen_backend_name = config.codegen_backend->name();
    cached_config.backend_config = *config.backend_config;
    CHECK_OK(cache_->Insert(instr, cached_config));
  }
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

absl::StatusOr<std::vector<Autotuner::ConfigResult>> Autotuner::ProfileAll(
    std::vector<ExecutableCandidate>& candidates) {
  std::vector<ConfigResult> results_vec;
  results_vec.reserve(candidates.size());

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

    std::optional<Failure> failure = std::nullopt;
    absl::Duration duration = absl::ZeroDuration();
    int scratch_bytes = 0;
    if (!profile_result.ok()) {
      failure = Failure{FailureKind::kExecutionFailed,
                        profile_result.status().ToString()};
    } else {
      duration = profile_result->duration;
      scratch_bytes = profile_result->scratch_bytes;
      if (autotune_config_.check_buffers) {
        CHECK(reference_output.has_value());
        CHECK(profile_result->output_buffer.has_value());
        failure =
            CheckBuffers(*input_buffers, profile_result->output_buffer.value(),
                         reference_output.value());
        if (failure.has_value()) {
          CHECK(!autotune_config_.crash_on_check_failure);
        }
      }
    }
    results_vec.push_back(
        {std::move(candidates[i].config), failure, duration, scratch_bytes});
  }
  return results_vec;
}

absl::StatusOr<Autotuner::ConfigResult> Autotuner::PickBestConfig(
    std::vector<ConfigResult>& results) {
  absl::Duration min_duration = absl::InfiniteDuration();
  ConfigResult* best_result = nullptr;
  for (ConfigResult& result : results) {
    if (!result.failure.has_value() && result.duration < min_duration) {
      min_duration = result.duration;
      best_result = &result;
    }
  }

  if (autotune_config_.optimize_scratch_bytes) {
    int64_t min_scratch_bytes = std::numeric_limits<int64_t>::max();
    absl::Duration duration_limit =
        min_duration +
        absl::Microseconds(autotune_config_.scratch_bytes_window_size_us);
    for (ConfigResult& result : results) {
      if (!result.failure.has_value() && result.duration <= duration_limit &&
          result.scratch_bytes < min_scratch_bytes) {
        best_result = &result;
        min_scratch_bytes = result.scratch_bytes;
      }
    }
  }

  if (best_result == nullptr) {
    return absl::InternalError("No valid config found!");
  }

  return std::move(*best_result);
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
      VLOG(2) << "Failed to profile executable: " << profile_result.status();
      continue;
    }
    if (profile_result.value().output_buffer.has_value()) {
      return std::move(profile_result.value().output_buffer.value());
    }
  }
  return absl::InternalError("No reference output found!");
}

std::optional<Autotuner::Failure> Autotuner::CheckBuffers(
    InputBuffers& input_buffers, ScopedShapedBuffer& output_buffer,
    ScopedShapedBuffer& reference_output) {
  absl::Status status = profiler_->CheckInputBuffers(input_buffers);
  if (!status.ok()) {
    return Failure{FailureKind::kRedzoneCheckFailed, status.ToString()};
  }
  status = profiler_->CheckOutputBuffer(output_buffer, reference_output,
                                        autotune_config_.relative_tolerance);
  if (!status.ok()) {
    return Failure{FailureKind::kWrongResults, status.ToString()};
  }
  return std::nullopt;
}

void Autotuner::LogConfigResults(const HloInstruction& instr,
                                 const std::vector<ConfigResult>& results) {
  for (const auto& result : results) {
    VLOG(2) << result.ToString(/*verbose=*/VLOG_IS_ON(3));
  }
  if (!autotune_config_.dump_logs_to.empty()) {
    AutotuningLog log;
    log.mutable_instr()->PackFrom(instr.ToProto());
    for (const auto& result : results) {
      *log.add_results() = result.ToProto();
    }
    *logs_.add_logs() = log;
  }
}

absl::Status Autotuner::DumpLogsToFile() {
  if (autotune_config_.dump_logs_to.empty()) {
    return absl::OkStatus();
  }

  std::string textproto;
  tsl::protobuf::TextFormat::PrintToString(logs_, &textproto);

  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(
      tsl::Env::Default(), autotune_config_.dump_logs_to, textproto));
  VLOG(1) << "Autotune logs serialized to file: "
          << autotune_config_.dump_logs_to;
  return absl::OkStatus();
}

std::string Autotuner::Failure::ToString() const {
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

std::string Autotuner::ConfigResult::ToString(bool verbose) const {
  std::string config_str = absl::StrFormat(
      "%s : %s", config.codegen_backend->name(),
      verbose ? UnpackedAnyShortDebugString(*config.backend_config) : "");
  if (failure.has_value()) {
    absl::StrAppend(&config_str, " ", failure->ToString());
  }
  return absl::StrFormat("{%s duration: %s, scratch_bytes: %d}", config_str,
                         absl::FormatDuration(duration), scratch_bytes);
}

AutotuneResult::FailureResult Autotuner::Failure::ToProto() const {
  AutotuneResult::FailureResult failure_proto;
  switch (kind) {
    case FailureKind::kCompilationFailed:
      failure_proto.set_kind(AutotuneResult::UNKNOWN);
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

AutotuneResult Autotuner::ConfigResult::ToProto() const {
  AutotuneResult result;
  if (config.backend_config->Is<AutotuneResult::GemmKey>()) {
    config.backend_config->UnpackTo(result.mutable_gemm());
  } else if (config.backend_config->Is<AutotuneResult::TritonGemmKey>()) {
    config.backend_config->UnpackTo(result.mutable_triton());
  } else if (config.backend_config
                 ->Is<stream_executor::dnn::AlgorithmProto>()) {
    config.backend_config->UnpackTo(result.mutable_algorithm());
  }
  if (failure.has_value()) {
    *result.mutable_failure() = failure->ToProto();
  }
  *result.mutable_run_time() = tsl::proto_utils::ToDurationProto(duration);
  result.set_scratch_bytes(scratch_bytes);
  return result;
}

}  // namespace xla
