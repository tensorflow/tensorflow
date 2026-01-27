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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "google/protobuf/text_format.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/autotuner_status_key.h"
#include "xla/service/shaped_buffer.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/proto/proto_utils.h"
#include "tsl/platform/blocking_counter.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/scoped_annotation.h"
#include "tsl/profiler/lib/traceme.h"

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

// It is important to fingerprint the entire module not just the autotuning
// candidates, to avoid collisions in the key-value store when several
// distinct modules have the same fusions, and are compiled at different
// times by the same PjRt client.
//
// TODO(b/394763704): Eliminate the sharding feature when we have offline
// autotuning. See below for an explanation of some issues.
//
// Theoretically, we also want to include the hash of the module config
// to ensure that a module compiled twice with different configs is
// autotuned twice.
//
// This is important since the config could e.g. affect codegen, or the
// space of possible parameters for autotuning. As a result, the autotuning
// results could look very different for the same module.
//
// Why is it not done here? Well, proto serialization is non-deterministic
// and may change across different builds. Which means that users who run
// on several hosts with different CPUs may end up generating different
// fingerprints for the same module config. They would then fail to
// exchange results through the key value store, which would lead to
// deadlocks. Therefore, we don't hash the module config here.
//
// The flip side is this: if we compile the same module twice in the same
// client, but with a different module config each time, we may hit the
// cache the second time and recover potentially inferior, or incomplete
// autotuning results.
std::string GetKvStoreKey(const HloModule* module, int shard_index) {
  return absl::StrCat("autotune_results_", module->GetFingerprint128(), "_",
                      shard_index);
}

}  // namespace

absl::StatusOr<Autotuner::Config> Autotuner::GetDefaultConfig(
    const HloInstruction& instr) {
  // TODO(b/446870267): Improve default backend selection. Currently we just
  // return the first backend that supports the instruction.
  for (auto& backend : codegen_backends_) {
    auto config = backend->GetDefaultConfig(instr);
    if (absl::IsUnimplemented(config.status())) {
      LOG(FATAL) << "GetDefaultConfig is not implemented for "
                 << backend->name();
    }
    if (config.ok()) {
      return Config{backend.get(), std::move(*config)};
    }
  }
  return absl::NotFoundError(
      absl::StrCat("No backend with default config found for instruction: ",
                   instr.ToString()));
}

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
  InstructionsByFingerprint instructions_by_fingerprint =
      GetAutotuningCandidates(module, should_autotune);
  if (instructions_by_fingerprint.empty()) {
    VLOG(1) << "No instructions to autotune.";
    return absl::OkStatus();
  }
  VLOG(1) << "Finding configs for " << instructions_by_fingerprint.size()
          << " unique instructions.";
  for (auto& [_, instructions] : instructions_by_fingerprint) {
    CHECK(!instructions.empty());
    TF_ASSIGN_OR_RETURN(Config config, GetConfig(instructions[0]));
    CodegenBackend* codegen_backend = config.codegen_backend;
    if (autotune_config_.dump_hlos) {
      TF_RETURN_IF_ERROR(DumpHlo(instructions[0], config));
    }
    for (auto* instr : instructions) {
      TF_RETURN_IF_ERROR(
          codegen_backend->ApplyConfig(*instr, *config.backend_config));
    }
  }
  return DumpLogsToFile();
}

absl::Status Autotuner::Autotune(HloModule* module,
                                 const InstructionFilterFn& should_autotune,
                                 MultiProcessKeyValueStore& sharding_kv_store) {
  CHECK(cache_ != nullptr) << "Sharding autotuning requires a cache.";
  int total_shards = sharding_kv_store.process_count;
  int my_shard_index = sharding_kv_store.process_index;

  // 1. Get all the instructions that could be autotuned.
  InstructionsByFingerprint all_instructions_by_fingerprint =
      GetAutotuningCandidates(module, should_autotune);
  if (all_instructions_by_fingerprint.empty()) {
    VLOG(1) << "No instructions to autotune.";
    return absl::OkStatus();
  }

  // 2. Shard and get instructions to autotune for current shard.
  // Sort the instructions by fingerprint to ensure deterministic sharding.
  std::vector<tsl::Fprint128> sorted_fingerprints;
  for (const auto& [fingerprint, _] : all_instructions_by_fingerprint) {
    sorted_fingerprints.push_back(fingerprint);
  }
  std::sort(sorted_fingerprints.begin(), sorted_fingerprints.end(),
            [](const tsl::Fprint128& a, const tsl::Fprint128& b) {
              if (a.high64 != b.high64) {
                return a.high64 < b.high64;
              }
              return a.low64 < b.low64;
            });

  const size_t bucket_size =
      std::ceil(static_cast<double>(sorted_fingerprints.size()) /
                static_cast<double>(total_shards));
  const size_t start = bucket_size * my_shard_index;
  const size_t end = std::min(start + bucket_size, sorted_fingerprints.size());

  InstructionsByFingerprint instructions_by_fingerprint;
  for (size_t i = start; i < end; ++i) {
    const tsl::Fprint128& fingerprint = sorted_fingerprints[i];
    instructions_by_fingerprint[fingerprint] =
        all_instructions_by_fingerprint.at(fingerprint);
  }

  // 3. Autotune instructions for this shard. Use cached configs if available,
  // otherwise autotune and cache the best config.
  VLOG(1) << "Shard " << my_shard_index << "/" << total_shards
          << ": finding configs for " << instructions_by_fingerprint.size()
          << "/" << all_instructions_by_fingerprint.size()
          << " unique instructions ";
  std::vector<const HloInstruction*> autotuned_instructions;
  for (auto& [_, instructions] : instructions_by_fingerprint) {
    CHECK(!instructions.empty());
    TF_ASSIGN_OR_RETURN(Config config, GetConfig(instructions[0]));
    autotuned_instructions.push_back(instructions[0]);
  }
  TF_RETURN_IF_ERROR(DumpLogsToFile());

  // 4. Store the results for this shard as a serialized string to the KV store.
  KeyValueStoreInterface& kv_store = *sharding_kv_store.key_value_store;
  const std::string local_key = GetKvStoreKey(module, my_shard_index);
  std::string local_results;
  if (!autotuned_instructions.empty()) {
    TF_ASSIGN_OR_RETURN(local_results,
                        cache_->Serialize(autotuned_instructions));
  }
  absl::StatusOr<std::string> stored_result = kv_store.TryGet(local_key);
  if (stored_result.status().code() == absl::StatusCode::kNotFound) {
    VLOG(2) << "Storing results for " << local_key;
    TF_RETURN_IF_ERROR(kv_store.Set(local_key, local_results));
    VLOG(2) << "Shard " << my_shard_index << " stored results at " << local_key;
  } else if (!stored_result.ok()) {
    return stored_result.status();
  } else {
    VLOG(2) << "Results already exist for " << local_key << ", skipping store.";
  }

  // 5. Load the autotune results of other shards from the KV store and update
  // the current shard's cache by deserializing the results.
  for (int i = 0; i < total_shards; ++i) {
    if (i == my_shard_index) {
      continue;
    }
    const std::string remote_key = GetKvStoreKey(module, i);
    VLOG(2) << "Shard " << my_shard_index << ": waiting for results from shard "
            << i << " / " << total_shards << " at " << remote_key;
    // TODO(b/361009609): reset to infinite duration once issue with MPI is
    // fixed. https://github.com/google/jax/issues/22995.
    TF_ASSIGN_OR_RETURN(std::string remote_results,
                        kv_store.Get(remote_key, absl::Hours(24)));
    if (!remote_results.empty()) {
      TF_RETURN_IF_ERROR(cache_->Deserialize(remote_results));
    }
  }

  // 6. Apply the results to all candidate instructions, must be already in
  // cache_ due to step 3 and 5 above.
  for (tsl::Fprint128 fingerprint : sorted_fingerprints) {
    std::vector<HloInstruction*>& instructions =
        all_instructions_by_fingerprint[fingerprint];
    CHECK(!instructions.empty());
    std::optional<Config> cached_config = LookUp(instructions[0]);
    CHECK(cached_config.has_value())
        << "Sharding autotuning failed: no config found for HLO: " +
               instructions[0]->ToString();
    if (autotune_config_.dump_hlos) {
      TF_RETURN_IF_ERROR(DumpHlo(instructions[0], *cached_config));
    }
    CodegenBackend* codegen_backend = cached_config->codegen_backend;
    for (auto* instr : instructions) {
      TF_RETURN_IF_ERROR(
          codegen_backend->ApplyConfig(*instr, *cached_config->backend_config));
    }
  }

  return absl::OkStatus();
}

absl::Status Autotuner::Autotune(HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(Config config, GetConfig(instr));
  CodegenBackend* codegen_backend = config.codegen_backend;
  if (autotune_config_.dump_hlos) {
    TF_RETURN_IF_ERROR(DumpHlo(instr, config));
  }
  TF_RETURN_IF_ERROR(
      codegen_backend->ApplyConfig(*instr, *config.backend_config));
  return DumpLogsToFile();
}

absl::StatusOr<Autotuner::Config> Autotuner::GetConfig(HloInstruction* instr) {
  VLOG(1) << "Getting config for HLO: " << instr->ToString();
  std::optional<Config> cached_config = LookUp(instr);
  if (cached_config.has_value()) {
    VLOG(1) << "Using cached config: " << cached_config->ToString();
    return std::move(cached_config.value());
  }

  if (autotune_config_.expect_all_instructions_in_cache) {
    absl::Status s = absl::NotFoundError(
        "No cached config found for HLO instr: " + instr->ToString());
    tsl::errors::InsertPayloads(
        s, {{std::string(gpu::kAutotuneCacheRequiredErrorPayloadKey), ""}});
    return s;
  }

  if (autotune_config_.use_default_config) {
    TF_ASSIGN_OR_RETURN(Config default_config, GetDefaultConfig(*instr));
    VLOG(1) << "Using default config: " << default_config.ToString();
    return default_config;
  }

  VLOG(1) << "Autotuning the HLO instruction to find best config.";
  TF_ASSIGN_OR_RETURN(Config best_config, TuneBestConfig(instr));
  Insert(instr, best_config);
  return best_config;
}

absl::Status Autotuner::IsValidExecutable(
    const absl::StatusOr<std::unique_ptr<Executable>>& executable) const {
  if (!executable.ok()) {
    return absl::Status(
        executable.status().code(),
        absl::StrCat("Compilation failed: ", executable.status().message()));
  }

  if (!autotune_config_.allow_reg_spills && executable.value()) {
    const auto spills_registers = [](const auto& pair) {
      const KernelStats& kernel_stats = pair.second;
      return kernel_stats.store_bytes_spilled > 0 ||
             kernel_stats.load_bytes_spilled > 0;
    };
    ModuleStats module_stats = executable.value()->module_stats();
    if (absl::c_any_of(module_stats, spills_registers)) {
      return absl::ResourceExhaustedError(
          "Discarding compilation due to register spilling.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Autotuner::Config> Autotuner::TuneBestConfig(
    HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(std::vector<Config> supported_configs,
                      GetSupportedConfigs(instr));
  if (supported_configs.empty()) {
    return absl::InternalError(
        absl::StrCat("Autotuner could not find any supported configs for HLO: ",
                     instr->ToString()));
  }
  VLOG(1) << "Found total of " << supported_configs.size()
          << " supported configs.";

  std::vector<absl::StatusOr<std::unique_ptr<Executable>>> executables =
      CompileAll(instr, supported_configs);

  std::vector<ExecutableCandidate> executable_candidates;
  for (int i = 0; i < executables.size(); ++i) {
    auto status = IsValidExecutable(executables[i]);
    if (status.ok()) {
      executable_candidates.push_back(
          {std::move(supported_configs[i]), std::move(executables[i].value())});
    } else {
      VLOG(4) << "Discarding config " << supported_configs[i].ToString()
              << " due to: " << status;
    }
  }

  if (autotune_config_.exclude_cublas_config) {
    executable_candidates.erase(
        std::remove_if(executable_candidates.begin(),
                       executable_candidates.end(),
                       [](const ExecutableCandidate& candidate) {
                         return candidate.config.codegen_backend->name() ==
                                "Cublas_fission";
                       }),
        executable_candidates.end());
  }

  if (executable_candidates.empty()) {
    return absl::InternalError(
        absl::StrCat("Autotuner could not compile any configs for HLO: ",
                     instr->ToString()));
  }
  VLOG(1) << "Successfully compiled " << executable_candidates.size()
          << " configs out of " << supported_configs.size() << " configs.";

  bool skip_profiling =
      executable_candidates.size() == 1 || autotune_config_.select_first_config;
  if (skip_profiling) {
    VLOG(1) << "Skipping profiling and using the "
            << (autotune_config_.select_first_config ? "first" : "only")
            << " config: " << executable_candidates[0].config.ToString();
    return std::move(executable_candidates[0].config);
  }

  TF_ASSIGN_OR_RETURN(std::vector<ConfigResult> results,
                      ProfileAll(executable_candidates));
  LogConfigResults(*instr, results);
  absl::StatusOr<ConfigResult> best_result = PickBestConfig(results);
  if (!best_result.ok()) {
    return absl::InternalError(
        absl::StrCat("Autotuning failed for HLO: ", instr->ToString(),
                     " with error: ", best_result.status().ToString()));
  }
  VLOG(1) << "Picked best config: "
          << best_result.value().ToString(/*verbose=*/true);
  return std::move(best_result.value().config);
}

Autotuner::InstructionsByFingerprint Autotuner::GetAutotuningCandidates(
    const HloModule* module, const InstructionFilterFn& should_autotune) {
  InstructionsByFingerprint instructions_by_fingerprint;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (should_autotune(*instr)) {
        instructions_by_fingerprint[GetFingerprint(instr)].push_back(instr);
      }
    }
  }
  return instructions_by_fingerprint;
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
    absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
        per_backend_configs = codegen_backend->GetSupportedConfigs(*instr);
    if (!per_backend_configs.ok()) {
      VLOG(3) << "Failed to get supported configs for backend "
              << codegen_backend->name() << ": "
              << per_backend_configs.status();
      continue;
    }
    VLOG(3) << "Found of " << per_backend_configs->size()
            << " supported configs for backend " << codegen_backend->name();
    for (auto& config : *per_backend_configs) {
      configs.push_back({codegen_backend.get(), std::move(config)});
    }
  }
  return configs;
}

std::vector<absl::StatusOr<std::unique_ptr<Executable>>> Autotuner::CompileAll(
    HloInstruction* instr, std::vector<Config>& configs) {
  XLA_SCOPED_LOGGING_TIMER_LEVEL("CompileAll", 5);
  tsl::profiler::TraceMe traceme("CompileAll");
  tsl::profiler::ScopedAnnotation annotation("XlaAutotunerCompilation");

  if (autotune_config_.select_first_config) {
    std::vector<absl::StatusOr<std::unique_ptr<Executable>>> executables;
    for (int i = 0; i < configs.size(); ++i) {
      absl::StatusOr<std::unique_ptr<Executable>> executable =
          configs[i].codegen_backend->Compile(*instr,
                                              *configs[i].backend_config);
      if (executable.ok() && IsValidExecutable(executable).ok()) {
        std::vector<absl::StatusOr<std::unique_ptr<Executable>>> success_result;
        success_result.push_back(std::move(executable));
        Config selected_config = std::move(configs[i]);
        configs.clear();
        configs.push_back(std::move(selected_config));
        return success_result;
      }
    }
    return executables;
  }

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
    VLOG(2) << "Checking buffers";
    reference_output = GetReferenceOutput(candidates, *input_buffers);
    if (!reference_output.has_value()) {
      LOG(WARNING) << "No reference output found even though buffer checking "
                      "was requested while autotuning";
    }
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
      if (autotune_config_.check_buffers && reference_output.has_value()) {
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

  if (best_result == nullptr) {
    return absl::NotFoundError("No valid config found!");
  }

  if (autotune_config_.optimize_scratch_bytes) {
    const ConfigResult* fastest_result = best_result;
    int64_t min_scratch_bytes = std::numeric_limits<int64_t>::max();
    absl::Duration duration_limit =
        min_duration +
        absl::Microseconds(autotune_config_.scratch_bytes_window_size_us);
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
              << "Tolerance: " << autotune_config_.scratch_bytes_window_size_us
              << "us.";
    }
  }

  return std::move(*best_result);
}

absl::Status Autotuner::DumpHlo(HloInstruction* instr, const Config& config) {
  const HloModule* parent_module = instr->GetModule();
  std::unique_ptr<HloModule> module = ExtractInstructionIntoNewModule(*instr);
  module->set_name(std::string(instr->name()));
  std::string id =
      absl::StrCat("autotuner_", dump_counter_++, ".", instr->name());
  DumpToFileInDirOrStdout(*parent_module, "", absl::StrCat(id, ".before.txt"),
                          module->ToString());
  HloInstruction* root = module->entry_computation()->root_instruction();
  TF_RETURN_IF_ERROR(
      config.codegen_backend->ApplyConfig(*root, *config.backend_config));
  DumpToFileInDirOrStdout(*parent_module, "", absl::StrCat(id, ".after.txt"),
                          module->ToString());
  return absl::OkStatus();
}

std::optional<ScopedShapedBuffer> Autotuner::GetReferenceOutput(
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
      VLOG(2) << "Found reference output for config: "
              << candidate.config.ToString();
      return std::move(profile_result.value().output_buffer.value());
    }
  }
  return std::nullopt;
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
  } else {
    result.mutable_other()->set_name(config.codegen_backend->name());
    *result.mutable_other()->mutable_config() = *config.backend_config;
  }
  if (failure.has_value()) {
    *result.mutable_failure() = failure->ToProto();
  }
  *result.mutable_run_time() = tsl::proto_utils::ToDurationProto(duration);
  result.set_scratch_bytes(scratch_bytes);
  return result;
}

std::string Autotuner::Config::ToString() const {
  return absl::StrFormat("%s : %s", codegen_backend->name(),
                         UnpackedAnyShortDebugString(*backend_config));
}

std::string AutotuneConfig::ToString() const {
  return absl::StrFormat(
      "{\n"
      "  \"check_buffers\": %s,\n"
      "  \"relative_tolerance\": %f,\n"
      "  \"crash_on_check_failure\": %s,\n"
      "  \"optimize_scratch_bytes\": %s,\n"
      "  \"scratch_bytes_window_size_us\": %d,\n"
      "  \"expect_all_instructions_in_cache\": %s,\n"
      "  \"dump_logs_to\": \"%s\",\n"
      "  \"exclude_cublas_config\": %s,\n"
      "  \"select_first_config\": %s,\n"
      "  \"use_default_config\": %s,\n"
      "  \"dump_hlos\": %s,\n"
      "  \"allow_reg_spills\": %s\n"
      "}",
      check_buffers ? "true" : "false", relative_tolerance,
      crash_on_check_failure ? "true" : "false",
      optimize_scratch_bytes ? "true" : "false", scratch_bytes_window_size_us,
      expect_all_instructions_in_cache ? "true" : "false", dump_logs_to,
      exclude_cublas_config ? "true" : "false",
      select_first_config ? "true" : "false",
      use_default_config ? "true" : "false", dump_hlos ? "true" : "false",
      allow_reg_spills ? "true" : "false");
}

}  // namespace xla
