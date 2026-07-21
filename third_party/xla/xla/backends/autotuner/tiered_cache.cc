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

#include "xla/backends/autotuner/tiered_cache.h"

#include <memory>
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
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/autotuner/autotune_cache_store.h"
#include "xla/backends/autotuner/autotune_fingerprint.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/protobuf.h"

namespace xla {

namespace {

std::string ToString(tsl::Fprint128 fingerprint) {
  return absl::StrCat(absl::Hex(fingerprint.high64, absl::kZeroPad16),
                      absl::Hex(fingerprint.low64, absl::kZeroPad16));
}

std::string CodegenOptionsFingerprintForInstr(const HloInstruction& instr) {
  if (instr.GetModule() == nullptr) {
    LOG(WARNING) << "Could not use DebugOptions for autotune cache key, "
                    "module is null for instruction: "
                 << instr.name();
    return "";
  }
  return GetCodegenOptionsFingerprint(
      instr.GetModule()->config().debug_options());
}

// Returns true if both full AutotuneKeys identify the same cache slot.
bool SameFullKey(const autotuner::AutotuneKey& a,
                 const autotuner::AutotuneKey& b) {
  return a.target().device() == b.target().device() &&
         a.target().explicit_version() == b.target().explicit_version() &&
         a.target().hlo_fingerprint() == b.target().hlo_fingerprint() &&
         a.environment().codegen_version() ==
             b.environment().codegen_version() &&
         a.environment().codegen_options_fingerprint() ==
             b.environment().codegen_options_fingerprint();
}

}  // namespace

TieredCache::TieredCache(AutotuneCacheContext context,
                         KeyMatchingMode matching_mode,
                         std::unique_ptr<AutotuneCacheStore> primary,
                         std::unique_ptr<AutotuneCacheStore> secondary)
    : context_(std::move(context)),
      matching_mode_(matching_mode),
      primary_(std::move(primary)),
      secondary_(std::move(secondary)) {
  CHECK(primary_ != nullptr) << "Primary store cannot be null.";
}

autotuner::AutotuneTargetKey TieredCache::BuildTargetKey(
    const HloInstruction& instr) const {
  autotuner::AutotuneTargetKey target_key;
  target_key.set_device(context_.device());
  target_key.set_explicit_version(context_.explicit_version());
  target_key.set_hlo_fingerprint(ToString(GetHloFingerprint(instr)));
  return target_key;
}

autotuner::AutotuneEntry TieredCache::BuildEntry(const HloInstruction& instr,
                                                 const Config& config) const {
  autotuner::AutotuneEntry entry;
  autotuner::AutotuneKey* key = entry.mutable_key();
  *key->mutable_target() = BuildTargetKey(instr);
  key->mutable_environment()->set_codegen_version(context_.codegen_version());
  // The codegen options fingerprint is only needed (and only matched) in strict
  // mode; computing it is expensive, so it is skipped for loose matching.
  if (matching_mode_ == KeyMatchingMode::kStrict) {
    std::string codegen_options_fp = CodegenOptionsFingerprintForInstr(instr);
    if (!codegen_options_fp.empty()) {
      key->mutable_environment()->set_codegen_options_fingerprint(
          codegen_options_fp);
    }
  }

  autotuner::AutotuneValue* value = entry.mutable_value();
  value->mutable_optimal_config()->set_backend(config.codegen_backend);
  *value->mutable_optimal_config()->mutable_backend_config() =
      config.backend_config;
  absl::flat_hash_map<autotuner::Backend, std::string>::const_iterator it =
      context_.per_backend_versions().find(config.codegen_backend);
  if (it != context_.per_backend_versions().end()) {
    value->set_optimal_backend_version(it->second);
  } else {
    LOG(WARNING) << "Backend version not found in context for backend: "
                 << config.codegen_backend;
  }
  return entry;
}

std::optional<autotuner::AutotuneEntry> TieredCache::MatchEntry(
    const std::vector<autotuner::AutotuneEntry>& entries,
    const std::optional<std::string>& codegen_options_fp) const {
  for (const autotuner::AutotuneEntry& entry : entries) {
    const autotuner::AutotuneEnvironmentKey& env = entry.key().environment();
    if (matching_mode_ == KeyMatchingMode::kStrict) {
      if (env.codegen_version() == context_.codegen_version() &&
          codegen_options_fp.has_value() &&
          env.codegen_options_fingerprint() == *codegen_options_fp) {
        return entry;
      }
      continue;
    }
    // Loose matching: accept if the combined codegen version matches, or if the
    // specific backend version of the cached optimal config still matches.
    if (env.codegen_version() == context_.codegen_version()) {
      return entry;
    }
    autotuner::Backend backend = entry.value().optimal_config().backend();
    absl::flat_hash_map<autotuner::Backend, std::string>::const_iterator it =
        context_.per_backend_versions().find(backend);
    if (it != context_.per_backend_versions().end() &&
        it->second == entry.value().optimal_backend_version()) {
      return entry;
    }
  }
  return std::nullopt;
}

absl::Status TieredCache::WriteToStore(
    AutotuneCacheStore& store, const autotuner::AutotuneEntry& entry) const {
  switch (store.GetMode()) {
    case CacheMode::kReadOnly:
      return absl::OkStatus();
    case CacheMode::kReadAppend: {
      absl::StatusOr<std::vector<autotuner::AutotuneEntry>> existing =
          store.Read(entry.key().target());
      if (existing.ok()) {
        for (const autotuner::AutotuneEntry& e : *existing) {
          if (SameFullKey(e.key(), entry.key())) {
            return absl::OkStatus();
          }
        }
      }
      return store.Write(entry);
    }
    case CacheMode::kReadWrite:
      return store.Write(entry);
  }
  return absl::OkStatus();
}

std::optional<AutotunerCacheInterface::Config> TieredCache::Lookup(
    const HloInstruction* instr) {
  CHECK(instr != nullptr) << "Instruction cannot be null.";
  autotuner::AutotuneTargetKey target_key = BuildTargetKey(*instr);

  // Compute the codegen options fingerprint once, only when strict matching
  // actually needs it.
  std::optional<std::string> codegen_options_fp;
  if (matching_mode_ == KeyMatchingMode::kStrict) {
    codegen_options_fp = CodegenOptionsFingerprintForInstr(*instr);
  }

  // 1. Look up in primary.
  absl::StatusOr<std::vector<autotuner::AutotuneEntry>> primary_entries =
      primary_->Read(target_key);
  if (primary_entries.ok()) {
    std::optional<autotuner::AutotuneEntry> matched =
        MatchEntry(*primary_entries, codegen_options_fp);
    if (matched.has_value()) {
      {
        absl::MutexLock lock(stats_mutex_);
        stats_.hits++;
      }
      const autotuner::Config& opt_config = matched->value().optimal_config();
      return Config{opt_config.backend(), opt_config.backend_config()};
    }
  }

  // 2. Look up in secondary.
  if (secondary_ != nullptr) {
    absl::StatusOr<std::vector<autotuner::AutotuneEntry>> secondary_entries =
        secondary_->Read(target_key);
    if (secondary_entries.ok()) {
      std::optional<autotuner::AutotuneEntry> matched =
          MatchEntry(*secondary_entries, codegen_options_fp);
      if (matched.has_value()) {
        // Promote the matched entry to the primary tier.
        WriteToStore(*primary_, *matched).IgnoreError();
        {
          absl::MutexLock lock(stats_mutex_);
          stats_.hits++;
        }
        const autotuner::Config& opt_config = matched->value().optimal_config();
        return Config{opt_config.backend(), opt_config.backend_config()};
      }
    }
  }

  absl::MutexLock lock(stats_mutex_);
  stats_.misses++;
  return std::nullopt;
}

absl::Status TieredCache::Insert(const HloInstruction* instr,
                                 const Config& config) {
  CHECK(instr != nullptr) << "Instruction cannot be null.";
  autotuner::AutotuneEntry entry = BuildEntry(*instr, config);
  absl::Status result = WriteToStore(*primary_, entry);
  if (secondary_ != nullptr) {
    absl::Status secondary_status = WriteToStore(*secondary_, entry);
    if (!secondary_status.ok() && result.ok()) {
      result = secondary_status;
    }
  }
  return result;
}

AutotunerCacheInterface::CacheStats TieredCache::GetCacheStats() const {
  absl::MutexLock lock(stats_mutex_);
  return stats_;
}

absl::StatusOr<std::string> TieredCache::Serialize(
    absl::Span<const HloInstruction* const> instructions_to_serialize) {
  autotuner::AutotuneCache cache;
  cache.set_device_scope(context_.device());
  cache.set_explicit_version_scope(context_.explicit_version());

  if (instructions_to_serialize.empty()) {
    ASSIGN_OR_RETURN(std::vector<autotuner::AutotuneEntry> all,
                     primary_->ReadAll());
    for (autotuner::AutotuneEntry& entry : all) {
      *cache.add_entries() = std::move(entry);
    }
  } else {
    for (const HloInstruction* instr : instructions_to_serialize) {
      autotuner::AutotuneTargetKey target_key = BuildTargetKey(*instr);
      absl::StatusOr<std::vector<autotuner::AutotuneEntry>> entries =
          primary_->Read(target_key);
      if (!entries.ok()) {
        continue;
      }
      for (autotuner::AutotuneEntry& entry : *entries) {
        *cache.add_entries() = std::move(entry);
      }
    }
  }
  return cache.SerializeAsString();
}

absl::Status TieredCache::Deserialize(absl::string_view serialized_cache) {
  autotuner::AutotuneCache cache;
  if (!cache.ParseFromString(serialized_cache)) {
    if (!tsl::protobuf::TextFormat::ParseFromString(serialized_cache, &cache)) {
      return absl::InvalidArgumentError(
          "Failed to parse AutotuneCache proto (binary or textproto).");
    }
  }
  // Populate every tier so that subsequent lookups hit the hottest tier.
  for (const autotuner::AutotuneEntry& entry : cache.entries()) {
    RETURN_IF_ERROR(primary_->Write(entry));
  }
  return absl::OkStatus();
}

CacheMode TieredCache::GetMode() const { return secondary_->GetMode(); }

}  // namespace xla
