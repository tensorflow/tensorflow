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

#include "xla/backends/autotuner/config_assigner.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
#include "absl/time/time.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/hlo_extractor.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/autotuner/tuner.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/autotuning/autotuner_status_key.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace {

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
std::string GetKvStoreKey(
    const HloModule* module, int shard_index,
    const std::vector<std::unique_ptr<CodegenBackend>>& codegen_backends) {
  std::vector<std::string> names;
  names.reserve(codegen_backends.size());
  for (const auto& backend : codegen_backends) {
    names.push_back(std::string(backend->name()));
  }
  absl::c_sort(names);
  std::string backend_names = absl::StrJoin(names, ",");
  uint32_t backend_fingerprint = tsl::Fingerprint32(backend_names);
  return absl::StrCat("autotune_results_", module->GetFingerprint128(), "_",
                      backend_fingerprint, "_", shard_index);
}

}  // namespace

absl::StatusOr<std::unique_ptr<ConfigAssigner>> ConfigAssigner::Create(
    Options options,
    std::unique_ptr<AutotunerCacheInterface> absl_nonnull cache,
    std::unique_ptr<CodegenOrchestrator> absl_nonnull orchestrator,
    std::unique_ptr<Profiler> profiler) {
  std::unique_ptr<Tuner> tuner = nullptr;
  if (profiler != nullptr) {
    Tuner::Options tuner_options;
    tuner_options.check_buffers = options.check_buffers;
    tuner_options.relative_tolerance = options.relative_tolerance;
    tuner_options.crash_on_check_failure = options.crash_on_check_failure;
    tuner_options.scratch_bytes_window_size_us =
        options.scratch_bytes_window_size_us;
    tuner_options.dump_logs_to = options.dump_logs_to;
    ASSIGN_OR_RETURN(tuner, Tuner::Create(std::move(profiler),
                                          orchestrator.get(), tuner_options));
  }
  return absl::WrapUnique(
      new ConfigAssigner(std::move(options), std::move(cache),
                         std::move(orchestrator), std::move(tuner)));
}

absl::Status ConfigAssigner::AssignConfigs(
    HloModule* module, const InstructionFilterFn& should_assign_config) {
  std::vector<EquivalentInstructions> instruction_groups =
      ExtractEquivalentInstructions(*module, should_assign_config);
  if (instruction_groups.empty()) {
    VLOG(1) << "No instructions to autotune.";
    return absl::OkStatus();
  }
  VLOG(1) << "Finding configs for " << instruction_groups.size()
          << " unique instructions.";

  ASSIGN_OR_RETURN(std::vector<Config> configs,
                   GetConfigsForAll(instruction_groups));

  for (int i = 0; i < instruction_groups.size(); i++) {
    auto& instructions = instruction_groups[i];
    Config config = std::move(configs[i]);
    if (options_.dump_hlos) {
      RETURN_IF_ERROR(DumpHlo(*instructions[0], config));
    }
    for (auto* instr : instructions) {
      RETURN_IF_ERROR(orchestrator_->ApplyConfig(*instr, config));
    }
  }
  if (tuner_ != nullptr) {
    RETURN_IF_ERROR(tuner_->DumpLogsToFile());
  }
  return absl::OkStatus();
}

absl::Status ConfigAssigner::AssignConfigs(
    HloModule* module, const InstructionFilterFn& should_assign_config,
    MultiProcessKeyValueStore& sharding_kv_store) {
  // Sharding the instructions only makes sense if we can have different
  // configs for different shards, which only happens due to online tuning.
  if (options_.select_first_config || options_.use_default_config) {
    VLOG(1) << "Falling back to non-sharded config assignment as online "
               "tuning is disabled.";
    return AssignConfigs(module, should_assign_config);
  }

  int total_shards = sharding_kv_store.process_count;
  int my_shard_index = sharding_kv_store.process_index;

  // 1. Get all the instructions that needs a config assignment.
  std::vector<EquivalentInstructions> all_instruction_groups =
      ExtractEquivalentInstructions(*module, should_assign_config);
  if (all_instruction_groups.empty()) {
    VLOG(1) << "No instructions to autotune.";
    return absl::OkStatus();
  }

  // 2. Shard and get configs for instructions in the current shard.
  const size_t bucket_size =
      std::ceil(static_cast<double>(all_instruction_groups.size()) /
                static_cast<double>(total_shards));
  const size_t start = bucket_size * my_shard_index;
  const size_t end =
      std::min(start + bucket_size, all_instruction_groups.size());
  std::vector<EquivalentInstructions> instruction_groups;
  instruction_groups.reserve(end - start);
  for (size_t i = start; i < end; ++i) {
    instruction_groups.push_back(all_instruction_groups[i]);
  }

  // 3. Autotune instructions for this shard. Use cached configs if available,
  // otherwise get and cache the best config.
  VLOG(1) << "Shard " << my_shard_index << "/" << total_shards
          << ": finding configs for " << instruction_groups.size() << "/"
          << all_instruction_groups.size() << " unique instructions ";

  ASSIGN_OR_RETURN(std::vector<Config> configs,
                   GetConfigsForAll(instruction_groups));

  std::vector<const HloInstruction*> autotuned_instructions;
  autotuned_instructions.reserve(instruction_groups.size());
  for (int i = 0; i < instruction_groups.size(); ++i) {
    autotuned_instructions.push_back(instruction_groups[i][0]);
  }
  if (tuner_ != nullptr) {
    RETURN_IF_ERROR(tuner_->DumpLogsToFile());
  }

  // 4. Store the results for this shard as a serialized string to the KV store.
  KeyValueStoreInterface& kv_store = *sharding_kv_store.key_value_store;
  const std::string local_key =
      GetKvStoreKey(module, my_shard_index, orchestrator_->codegen_backends());
  std::string local_results;
  if (!autotuned_instructions.empty()) {
    ASSIGN_OR_RETURN(local_results,
                     optimal_config_cache_->Serialize(autotuned_instructions));
  }
  absl::StatusOr<std::string> stored_result = kv_store.TryGet(local_key);
  if (stored_result.status().code() == absl::StatusCode::kNotFound) {
    VLOG(2) << "Storing results for " << local_key;
    absl::Status set_result = kv_store.Set(local_key, local_results);
    if (absl::IsAlreadyExists(set_result)) {
      VLOG(2) << "Shard " << my_shard_index << " tried to store results at "
              << local_key << " but lost a race to do so";
    } else if (!set_result.ok()) {
      return set_result;
    } else {
      VLOG(2) << "Shard " << my_shard_index << " stored results at "
              << local_key;
    }
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
    const std::string remote_key =
        GetKvStoreKey(module, i, orchestrator_->codegen_backends());
    VLOG(2) << "Shard " << my_shard_index << ": waiting for results from shard "
            << i << " / " << total_shards << " at " << remote_key;
    // TODO(b/361009609): reset to infinite duration once issue with MPI is
    // fixed. https://github.com/google/jax/issues/22995.
    ASSIGN_OR_RETURN(std::string remote_results,
                     kv_store.Get(remote_key, absl::Hours(24)));
    if (!remote_results.empty()) {
      RETURN_IF_ERROR(optimal_config_cache_->Deserialize(remote_results));
    }
  }

  // 6. Apply the results to all candidate instructions, must be already in
  // cache_ due to step 3 and 5 above.
  for (auto& instruction_group : all_instruction_groups) {
    CHECK(!instruction_group.empty());
    std::optional<Config> cached_config = LookUp(instruction_group[0]);
    if (!cached_config.has_value()) {
      return absl::InternalError(absl::StrCat(
          "Autotuning failed for HLO: ", instruction_group[0]->ToString(),
          ". No configuration found in cache after synchronizing results "
          "across all shards."));
    }
    if (options_.dump_hlos) {
      RETURN_IF_ERROR(DumpHlo(*instruction_group[0], *cached_config));
    }
    for (auto* instr : instruction_group) {
      RETURN_IF_ERROR(orchestrator_->ApplyConfig(*instr, *cached_config));
    }
  }

  return absl::OkStatus();
}

absl::Status ConfigAssigner::AssignConfig(HloInstruction* instr) {
  ASSIGN_OR_RETURN(Config config, GetConfig(instr).Await());
  if (options_.dump_hlos) {
    RETURN_IF_ERROR(DumpHlo(*instr, config));
  }
  RETURN_IF_ERROR(orchestrator_->ApplyConfig(*instr, config));
  return tuner_ != nullptr ? tuner_->DumpLogsToFile() : absl::OkStatus();
}

tsl::Future<ConfigAssigner::Config> ConfigAssigner::GetConfig(
    const HloInstruction* instr) {
  if (VLOG_IS_ON(1)) {
    HloPrintOptions print_options;
    if (VLOG_IS_ON(4)) {
      print_options.set_print_subcomputation_mode(
          HloPrintOptions::PrintSubcomputationMode::kFullBodies);
    }
    VLOG(1) << "Getting config for HLO: " << instr->ToString(print_options);
  }
  std::optional<Config> cached_config = LookUp(instr);
  if (cached_config.has_value()) {
    VLOG(1) << "Using cached config: " << cached_config->ToString();
    return std::move(cached_config.value());
  }

  if (options_.expect_all_instructions_in_cache) {
    absl::Status s = absl::NotFoundError(absl::StrCat(
        "No cached config found for HLO instr: ", instr->ToString()));
    tsl::errors::InsertPayloads(
        s, {{std::string(gpu::kAutotuneCacheRequiredErrorPayloadKey), ""}});
    return s;
  }

  // TODO (b/446870267): Improve the cache fallback logic as we move to offline
  // autotuning.
  if (options_.use_default_config) {
    ASSIGN_OR_RETURN(Config default_config,
                     orchestrator_->GetDefaultConfig(*instr));
    VLOG(1) << "Using default config: " << default_config.ToString();
    return default_config;
  }

  if (options_.select_first_config) {
    ASSIGN_OR_RETURN(std::vector<Config> supported_configs,
                     orchestrator_->GetSupportedConfigs(*instr));
    for (Config& config : supported_configs) {
      auto executable = orchestrator_->Compile(*instr, config);
      if (executable.ok()) {
        VLOG(1) << "Using first compilable config: " << config.ToString();
        return std::move(config);
      }
    }
    return absl::InternalError(
        absl::StrCat("No supported config found for HLO: ", instr->ToString()));
  }

  if (tuner_ == nullptr) {
    return absl::FailedPreconditionError(
        "Autotuning failed. Profiler is null (no Tuner available).");
  }
  VLOG(1) << "Autotuning the HLO instruction to find best config.";
  return tuner_->GetTunedConfig(instr).Map(
      [&, instr](ConfigAssigner::Config best_config) -> absl::StatusOr<Config> {
        RETURN_IF_ERROR(Insert(instr, best_config));
        return best_config;
      });
}

std::optional<ConfigAssigner::Config> ConfigAssigner::LookUp(
    const HloInstruction* instr) const {
  auto cached_config = optimal_config_cache_->Lookup(instr);
  if (!cached_config.has_value()) {
    return std::nullopt;
  }
  VLOG(1) << "Found cached config for HLO: " << instr->ToString();
  for (const auto& codegen_backend : orchestrator_->codegen_backends()) {
    if (codegen_backend->backend() == cached_config->codegen_backend) {
      auto backend_config =
          std::make_unique<BackendConfig>(cached_config->backend_config);
      return Config{codegen_backend.get(), std::move(backend_config)};
    }
  }
  LOG(WARNING) << "Ignoring cached config from backend "
               << Backend_Name(cached_config->codegen_backend) << " for HLO '"
               << instr->ToString() << "'"
               << ", because this backend is not registered with the "
                  "autotuner.";
  return std::nullopt;
}

absl::Status ConfigAssigner::Insert(const HloInstruction* instr,
                                    const ConfigAssigner::Config& config) {
  AutotunerCacheInterface::Config cached_config;
  cached_config.codegen_backend = config.codegen_backend->backend();
  cached_config.backend_config = *config.backend_config;
  return optimal_config_cache_->Insert(instr, cached_config);
}

absl::StatusOr<std::vector<ConfigAssigner::Config>>
ConfigAssigner::GetConfigsForAll(
    const std::vector<InstructionGroup>& instruction_groups) {
  std::vector<tsl::Future<Config>> future_configs;
  future_configs.reserve(instruction_groups.size());
  for (int i = 0; i < instruction_groups.size(); i++) {
    future_configs.push_back(GetConfig(instruction_groups[i][0]));
  }

  std::vector<absl::StatusOr<Config>> status_or_configs;
  status_or_configs.reserve(instruction_groups.size());
  absl::Status combined_status = absl::OkStatus();
  int num_failures = 0;
  for (int i = 0; i < instruction_groups.size(); i++) {
    absl::StatusOr<Config> config_or = std::move(future_configs[i]).Await();
    combined_status.Update(config_or.status());
    if (!config_or.ok()) {
      LOG(ERROR)
          << "Could not get config for HLO: "
          << instruction_groups[i][0]->ToString(
                 HloPrintOptions().set_print_subcomputation_mode(
                     HloPrintOptions::PrintSubcomputationMode::kFullBodies))
          << ". Status: " << config_or.status();
      num_failures++;
    }
    status_or_configs.push_back(std::move(config_or));
  }

  if (!combined_status.ok() && num_failures > 1) {
    return tsl::errors::CreateWithUpdatedMessage(
        combined_status,
        absl::StrCat(
            "Failed to get configs for: ", num_failures, " out of ",
            instruction_groups.size(),
            " instructions. See logs for all failures. Example failure: \n",
            combined_status.message()));
  }
  RETURN_IF_ERROR(combined_status);

  std::vector<Config> configs;
  for (auto& config_or : status_or_configs) {
    if (config_or.ok()) {
      configs.push_back(std::move(*config_or));
    }
  }
  return configs;
}

absl::Status ConfigAssigner::DumpHlo(const HloInstruction& instr,
                                     const Config& config) {
  const HloModule* parent_module = instr.GetModule();
  std::unique_ptr<HloModule> module = ExtractInstructionIntoNewModule(instr);
  module->set_name(std::string(instr.name()));
  std::string id =
      absl::StrCat("autotuner_", dump_counter_++, ".", instr.name());
  DumpToFileInDirOrStdout(*parent_module, "", absl::StrCat(id, ".before.txt"),
                          module->ToString());
  HloInstruction* root = module->entry_computation()->root_instruction();
  RETURN_IF_ERROR(orchestrator_->ApplyConfig(*root, config));
  DumpToFileInDirOrStdout(*parent_module, "", absl::StrCat(id, ".after.txt"),
                          module->ToString());
  return absl::OkStatus();
}

std::string AutotuneConfig::ToString() const {
  return absl::StrFormat(
      "{\n"
      "  \"check_buffers\": %s,\n"
      "  \"relative_tolerance\": %f,\n"
      "  \"crash_on_check_failure\": %s,\n"
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
      crash_on_check_failure ? "true" : "false", scratch_bytes_window_size_us,
      expect_all_instructions_in_cache ? "true" : "false", dump_logs_to,
      exclude_cublas_config ? "true" : "false",
      select_first_config ? "true" : "false",
      use_default_config ? "true" : "false", dump_hlos ? "true" : "false",
      allow_reg_spills_fn ? "dynamic" : "null");
}

AutotunerCacheInterface::CacheStats ConfigAssigner::GetCacheStats() const {
  return optimal_config_cache_->GetCacheStats();
}

}  // namespace xla
