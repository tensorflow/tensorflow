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

#include "xla/backends/autotuner/persistent_cache.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/autotuner/autotune_fingerprint.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

namespace {

std::string ToString(tsl::Fprint128 fingerprint) {
  return absl::StrCat(absl::Hex(fingerprint.high64, absl::kZeroPad16),
                      absl::Hex(fingerprint.low64, absl::kZeroPad16));
}

std::string GetCodegenOptionsFingerprint(const HloInstruction& instr) {
  if (instr.GetModule() != nullptr) {
    return GetCodegenOptionsFingerprint(
        instr.GetModule()->config().debug_options());
  }
  LOG(WARNING) << "Couldn't get debug options, module is null for instruction: "
               << instr.name();
  return "";
}
}  // namespace

PersistentCache::PersistentCache(AutotuneCacheContext context, CacheMode mode,
                                 KeyMatchingMode matching_mode)
    : context_(std::move(context)),
      mode_(mode),
      matching_mode_(matching_mode) {}

std::optional<AutotunerCacheInterface::Config> PersistentCache::Lookup(
    const HloInstruction* instr) {
  CHECK(instr != nullptr) << "Instruction cannot be null.";
  autotuner::AutotuneTargetKey target_key;
  target_key.set_device(context_.device());
  target_key.set_explicit_version(context_.explicit_version());
  target_key.set_hlo_fingerprint(ToString(GetHloFingerprint(*instr)));

  if (matching_mode_ == KeyMatchingMode::kStrict) {
    std::string codegen_options_fp = GetCodegenOptionsFingerprint(*instr);
    autotuner::AutotuneKey key;
    *key.mutable_target() = std::move(target_key);
    key.mutable_environment()->set_codegen_version(context_.codegen_version());
    key.mutable_environment()->set_codegen_options_fingerprint(
        std::move(codegen_options_fp));

    absl::StatusOr<std::optional<autotuner::AutotuneValue>> value = Read(key);
    if (!value.ok() || !value->has_value()) {
      absl::MutexLock lock(stats_mutex_);
      stats_.misses++;
      return std::nullopt;
    }
    const autotuner::Config& opt_config = value->value().optimal_config();
    {
      absl::MutexLock lock(stats_mutex_);
      stats_.hits++;
    }
    return Config{opt_config.backend(), opt_config.backend_config()};
  }
  if (matching_mode_ == KeyMatchingMode::kLoose) {
    absl::StatusOr<std::vector<autotuner::AutotuneEntry>> entries =
        Read(target_key);
    if (!entries.ok() || entries->empty()) {
      absl::MutexLock lock(stats_mutex_);
      stats_.misses++;
      return std::nullopt;
    }
    // Returns the first entry that matches the backend version. We can improve
    // this further when we support multiple entries per target key.
    for (const autotuner::AutotuneEntry& entry : *entries) {
      autotuner::Backend backend = entry.value().optimal_config().backend();
      absl::flat_hash_map<autotuner::Backend, std::string>::const_iterator it =
          context_.per_backend_versions().find(backend);
      if (it != context_.per_backend_versions().end() &&
          it->second == entry.value().optimal_backend_version()) {
        {
          absl::MutexLock lock(stats_mutex_);
          stats_.hits++;
        }
        return Config{backend, entry.value().optimal_config().backend_config()};
      }
    }
  }

  absl::MutexLock lock(stats_mutex_);
  stats_.misses++;
  return std::nullopt;
}

absl::Status PersistentCache::Insert(const HloInstruction* instr,
                                     const Config& config) {
  if (mode_ == CacheMode::kReadOnly) {
    return absl::PermissionDeniedError("Cache is in ReadOnly mode.");
  }

  autotuner::AutotuneTargetKey target_key;
  target_key.set_device(context_.device());
  target_key.set_explicit_version(context_.explicit_version());
  target_key.set_hlo_fingerprint(ToString(GetHloFingerprint(*instr)));

  std::string codegen_options_fp = GetCodegenOptionsFingerprint(*instr);

  autotuner::AutotuneKey key;
  *key.mutable_target() = std::move(target_key);
  key.mutable_environment()->set_codegen_version(context_.codegen_version());
  key.mutable_environment()->set_codegen_options_fingerprint(
      codegen_options_fp);

  if (mode_ == CacheMode::kReadAppend) {
    absl::StatusOr<std::optional<autotuner::AutotuneValue>> value = Read(key);
    if (value.ok() && value->has_value()) {
      return absl::AlreadyExistsError(
          "Entry already exists in ReadAppend mode.");
    }
  }

  autotuner::AutotuneValue val;
  val.mutable_optimal_config()->set_backend(config.codegen_backend);
  *val.mutable_optimal_config()->mutable_backend_config() =
      config.backend_config;

  absl::flat_hash_map<autotuner::Backend, std::string>::const_iterator it =
      context_.per_backend_versions().find(config.codegen_backend);
  if (it != context_.per_backend_versions().end()) {
    val.set_optimal_backend_version(it->second);
  } else {
    LOG(WARNING) << "Backend version not found in context for backend: "
                 << config.codegen_backend;
  }

  autotuner::AutotuneEntry entry;
  *entry.mutable_key() = std::move(key);
  *entry.mutable_value() = std::move(val);

  return Write(entry);
}

absl::StatusOr<std::optional<autotuner::AutotuneValue>> PersistentCache::Read(
    const autotuner::AutotuneKey& key) {
  ASSIGN_OR_RETURN(std::vector<autotuner::AutotuneEntry> entries,
                   Read(key.target()));
  for (const autotuner::AutotuneEntry& entry : entries) {
    if (entry.key().environment().codegen_version() ==
            key.environment().codegen_version() &&
        entry.key().environment().codegen_options_fingerprint() ==
            key.environment().codegen_options_fingerprint()) {
      return entry.value();
    }
  }
  return std::nullopt;
}

AutotunerCacheInterface::CacheStats PersistentCache::GetCacheStats() const {
  absl::MutexLock lock(stats_mutex_);
  return stats_;
}

}  // namespace xla
