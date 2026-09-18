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
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/autotune_cache.pb.h"
#include "xla/backends/autotuner/autotune_cache_store.h"
#include "xla/backends/autotuner/autotune_fingerprint.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "tsl/platform/protobuf.h"

namespace xla {

namespace {

// TODO(b/444398084): Use codegen options fingerprint when strict matching
// is enabled. Ignoring it for now since the always changing debug options
// make the cache useless in strict mode. We need to improve it to only use
// the parts of the debug options that are relevant for codegen.
[[maybe_unused]] std::string CodegenOptionsFingerprintForInstr(
    const HloInstruction& instr) {
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
bool IsSameCacheSlot(const autotuner::AutotuneKey& a,
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
  target_key.set_hlo_fingerprint(
      xla::AutotuneFingerprintToString(GetHloFingerprint(instr)));
  return target_key;
}

autotuner::AutotuneEntry TieredCache::BuildEntry(const HloInstruction& instr,
                                                 const Config& config) const {
  autotuner::AutotuneEntry entry;
  autotuner::AutotuneKey* key = entry.mutable_key();
  *key->mutable_target() = BuildTargetKey(instr);
  key->mutable_environment()->set_codegen_version(context_.codegen_version());
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

TieredCache::MissReason TieredCache::MostInformativeMissReason(MissReason a,
                                                               MissReason b) {
  // A version mismatch means the entry is cached but unusable, which is more
  // informative than a read error, which in turn is more informative than not
  // finding anything at all.
  if (a == MissReason::kVersionMismatch || b == MissReason::kVersionMismatch) {
    return MissReason::kVersionMismatch;
  }
  if (a == MissReason::kReadError || b == MissReason::kReadError) {
    return MissReason::kReadError;
  }
  return MissReason::kNotFound;
}

AutotunerCacheInterface::Config TieredCache::ToConfig(
    const autotuner::AutotuneEntry& entry) {
  const autotuner::Config& optimal_config = entry.value().optimal_config();
  return Config{optimal_config.backend(), optimal_config.backend_config()};
}

void TieredCache::RecordHit(const Hit& hit, bool in_memory_hit) {
  absl::MutexLock lock(stats_mutex_);
  stats_.hits++;
  if (hit.is_strict) {
    stats_.strict_hits++;
  }
  if (in_memory_hit) {
    stats_.in_memory_hits++;
  }
}

void TieredCache::RecordMiss(MissReason reason) {
  absl::MutexLock lock(stats_mutex_);
  stats_.RecordMiss(reason);
}

TieredCache::MatchResult TieredCache::MatchEntry(
    const std::vector<autotuner::AutotuneEntry>& entries) const {
  for (const autotuner::AutotuneEntry& entry : entries) {
    const autotuner::AutotuneEnvironmentKey& env = entry.key().environment();
    if (env.codegen_version() == context_.codegen_version()) {
      return Hit{entry, /*is_strict=*/true};
    }
    if (matching_mode_ == KeyMatchingMode::kStrict) {
      continue;
    }
    // Loose matching: accept if the specific backend version of the cached
    // optimal config still matches. Return the first match found.
    autotuner::Backend backend = entry.value().optimal_config().backend();
    absl::flat_hash_map<autotuner::Backend, std::string>::const_iterator it =
        context_.per_backend_versions().find(backend);
    if (it != context_.per_backend_versions().end() &&
        it->second == entry.value().optimal_backend_version()) {
      return Hit{entry, /*is_strict=*/false};
    }
  }

  if (entries.empty()) {
    return MissReason::kNotFound;
  }
  return MissReason::kVersionMismatch;
}

absl::Status TieredCache::MaybeWriteToStore(
    AutotuneCacheStore& store, const autotuner::AutotuneEntry& entry) const {
  switch (store.GetMode()) {
    case CacheMode::kReadOnly:
      return absl::OkStatus();
    case CacheMode::kReadAppend: {
      absl::StatusOr<std::vector<autotuner::AutotuneEntry>> existing =
          store.Read(entry.key().target());
      if (existing.ok()) {
        for (const autotuner::AutotuneEntry& e : *existing) {
          if (IsSameCacheSlot(e.key(), entry.key())) {
            return absl::OkStatus();
          }
        }
      }
      return store.Write(entry);
    }
    case CacheMode::kReadWrite:
    case CacheMode::kWriteOnly:
      return store.Write(entry);
  }
  return absl::OkStatus();
}

TieredCache::MatchResult TieredCache::LookupInStore(
    AutotuneCacheStore& store,
    const autotuner::AutotuneTargetKey& target_key) const {
  absl::StatusOr<std::vector<autotuner::AutotuneEntry>> entries =
      store.Read(target_key);
  if (!entries.ok()) {
    return MissReason::kReadError;
  }
  return MatchEntry(*entries);
}

std::optional<AutotunerCacheInterface::Config> TieredCache::Lookup(
    const HloInstruction* instr) {
  CHECK(instr != nullptr) << "Instruction cannot be null.";
  autotuner::AutotuneTargetKey target_key = BuildTargetKey(*instr);

  // 1. Look up in the primary (in-memory) tier.
  MatchResult primary_match = LookupInStore(*primary_, target_key);
  if (const Hit* hit = std::get_if<Hit>(&primary_match); hit != nullptr) {
    RecordHit(*hit, /*in_memory_hit=*/true);
    return ToConfig(hit->entry);
  }
  MissReason miss_reason = std::get<MissReason>(primary_match);

  // 2. Fall back to the secondary (persistent) tier, if there is one to read.
  if (secondary_ != nullptr && secondary_->GetMode() != CacheMode::kWriteOnly) {
    MatchResult secondary_match = LookupInStore(*secondary_, target_key);
    if (const Hit* hit = std::get_if<Hit>(&secondary_match); hit != nullptr) {
      // Promote the matched entry to the primary tier. Promotion is
      // best-effort; a failure only costs us a future secondary lookup.
      MaybeWriteToStore(*primary_, hit->entry).IgnoreError();
      RecordHit(*hit, /*in_memory_hit=*/false);
      return ToConfig(hit->entry);
    }
    miss_reason = MostInformativeMissReason(
        miss_reason, std::get<MissReason>(secondary_match));
  }

  RecordMiss(miss_reason);
  return std::nullopt;
}

absl::Status TieredCache::Insert(const HloInstruction* instr,
                                 const Config& config) {
  autotuner::AutotuneEntry entry = BuildEntry(*instr, config);
  absl::Status result = MaybeWriteToStore(*primary_, entry);
  if (secondary_ != nullptr) {
    result.Update(MaybeWriteToStore(*secondary_, entry));
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
    ABSL_ASSIGN_OR_RETURN(std::vector<autotuner::AutotuneEntry> all,
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
    ABSL_RETURN_IF_ERROR(primary_->Write(entry));
  }
  return absl::OkStatus();
}

CacheMode TieredCache::GetMode() const {
  if (secondary_ != nullptr) {
    return secondary_->GetMode();
  }
  return primary_->GetMode();
}

}  // namespace xla
