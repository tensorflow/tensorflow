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

#include "xla/backends/autotuner/local_cache.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/autotune_fingerprint.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

namespace {
struct LocalCacheKey {
  std::string hlo_fingerprint;
  std::string codegen_options_fp;

  std::string ToString(KeyMatchingMode matching_mode) const {
    if (matching_mode == KeyMatchingMode::kLoose ||
        codegen_options_fp.empty()) {
      return hlo_fingerprint;
    }
    return absl::StrCat(hlo_fingerprint, "|", codegen_options_fp);
  }

  static LocalCacheKey FromString(absl::string_view key) {
    LocalCacheKey cache_key;
    size_t pipe_pos = key.find('|');
    if (pipe_pos != absl::string_view::npos) {
      cache_key.hlo_fingerprint = std::string(key.substr(0, pipe_pos));
      cache_key.codegen_options_fp = std::string(key.substr(pipe_pos + 1));
    } else {
      cache_key.hlo_fingerprint = std::string(key);
    }
    return cache_key;
  }
};
}  // namespace

LocalCacheStorage& LocalCacheStorage::GetInstance(
    const AutotuneCacheContext& ctx) {
  static absl::Mutex mu(absl::kConstInit);
  using StorageMap =
      absl::flat_hash_map<std::string, std::unique_ptr<LocalCacheStorage>>;
  static auto* instances = new StorageMap();

  std::string key = ctx.GetId();
  absl::MutexLock lock(mu);
  auto [it, inserted] = instances->try_emplace(key, nullptr);
  if (inserted) {
    it->second = std::make_unique<LocalCacheStorage>();
  }
  return *it->second;
}

LocalCache::LocalCache(KeyMatchingMode matching_mode,
                       LocalCacheStorage* absl_nonnull storage)
    : matching_mode_(matching_mode), storage_(storage) {
  CHECK(storage_ != nullptr) << "LocalCacheStorage cannot be null.";
}

std::string LocalCache::GetCacheKey(const HloInstruction* instr) const {
  tsl::Fprint128 hlo_fp = GetHloFingerprint(*instr);
  std::string hlo_fingerprint =
      absl::StrCat(absl::Hex(hlo_fp.high64, absl::kZeroPad16),
                   absl::Hex(hlo_fp.low64, absl::kZeroPad16));

  std::string codegen_options_fp = "";
  if (matching_mode_ != KeyMatchingMode::kLoose) {
    if (instr->GetModule() == nullptr) {
      LOG(WARNING) << "Could not use DebugOptions for autotune cache key, "
                      "module is null for instruction: "
                   << instr->name();
    } else {
      codegen_options_fp = GetCodegenOptionsFingerprint(
          instr->GetModule()->config().debug_options());
    }
  }

  return LocalCacheKey{std::move(hlo_fingerprint),
                       std::move(codegen_options_fp)}
      .ToString(matching_mode_);
}

std::optional<AutotunerCacheInterface::Config> LocalCache::Lookup(
    const HloInstruction* instr) {
  std::string key = GetCacheKey(instr);
  absl::MutexLock lock(storage_->mutex_);
  auto it = storage_->cache_.find(key);
  if (it != storage_->cache_.end()) {
    storage_->stats_.hits++;
    return it->second;
  }
  storage_->stats_.misses++;
  return std::nullopt;
}

absl::Status LocalCache::Insert(const HloInstruction* instr,
                                const Config& config) {
  if (instr == nullptr) {
    return absl::InvalidArgumentError("Instruction cannot be null.");
  }
  std::string key = GetCacheKey(instr);
  absl::MutexLock lock(storage_->mutex_);
  storage_->cache_[key] = config;
  return absl::OkStatus();
}

absl::StatusOr<std::string> LocalCache::Serialize(
    absl::Span<const HloInstruction* const> instructions_to_serialize) {
  // We are only serializing the data currently in the local cache. This is
  // sufficient if we only provide (de)serialization support within the same
  // implementation. It can be extended by saving the scope in the local
  // cache, using it during serialization, and validating against it during
  // deserialization.
  autotuner::AutotuneCache proto_cache;
  absl::MutexLock lock(storage_->mutex_);

  auto add_entry = [&](absl::string_view key, const Config& config) {
    LocalCacheKey cache_key = LocalCacheKey::FromString(key);

    autotuner::AutotuneEntry* entry = proto_cache.add_entries();
    entry->mutable_key()->mutable_target()->set_hlo_fingerprint(
        cache_key.hlo_fingerprint);
    if (!cache_key.codegen_options_fp.empty()) {
      entry->mutable_key()
          ->mutable_environment()
          ->set_codegen_options_fingerprint(cache_key.codegen_options_fp);
    }
    entry->mutable_value()->mutable_optimal_config()->set_backend(
        config.codegen_backend);
    *entry->mutable_value()
         ->mutable_optimal_config()
         ->mutable_backend_config() = config.backend_config;
  };

  if (!instructions_to_serialize.empty()) {
    for (const HloInstruction* instr : instructions_to_serialize) {
      std::string key = GetCacheKey(instr);
      if (auto it = storage_->cache_.find(key); it != storage_->cache_.end()) {
        add_entry(key, it->second);
      }
    }
  } else {
    for (const auto& [key, config] : storage_->cache_) {
      add_entry(key, config);
    }
  }

  std::string serialized;
  if (!proto_cache.SerializeToString(&serialized)) {
    return absl::InternalError("Failed to serialize AutotuneCache proto.");
  }
  return serialized;
}

absl::Status LocalCache::Deserialize(absl::string_view serialized_cache) {
  autotuner::AutotuneCache proto_cache;
  if (!proto_cache.ParseFromString(serialized_cache)) {
    return absl::InvalidArgumentError("Failed to parse AutotuneCache proto.");
  }

  absl::MutexLock lock(storage_->mutex_);
  for (const autotuner::AutotuneEntry& entry : proto_cache.entries()) {
    LocalCacheKey cache_key{
        entry.key().target().hlo_fingerprint(),
        entry.key().environment().codegen_options_fingerprint()};
    std::string key = cache_key.ToString(matching_mode_);

    AutotunerCacheInterface::Config config;
    config.codegen_backend = entry.value().optimal_config().backend();
    config.backend_config = entry.value().optimal_config().backend_config();

    storage_->cache_[key] = config;
  }
  return absl::OkStatus();
}

AutotunerCacheInterface::CacheStats LocalCache::GetCacheStats() const {
  absl::MutexLock lock(storage_->mutex_);
  return storage_->stats_;
}

}  // namespace xla
