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

#include <memory>
#include <optional>
#include <string>

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

LocalCacheStorage& LocalCacheStorage::GetInstance(const AutotuneScope& scope) {
  static absl::Mutex mu(absl::kConstInit);
  using StorageMap =
      absl::flat_hash_map<std::string, std::unique_ptr<LocalCacheStorage>>;
  static auto* instances = new StorageMap();

  std::string key = scope.GetId();
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
  std::string key = absl::StrCat(absl::Hex(hlo_fp.high64, absl::kZeroPad16),
                                 absl::Hex(hlo_fp.low64, absl::kZeroPad16));
  if (matching_mode_ == KeyMatchingMode::kLoose) {
    return key;
  }
  if (instr->GetModule() == nullptr) {
    LOG(WARNING) << "Could not use DebugOptions for autotune cache key, "
                    "module is null for instruction: "
                 << instr->name();
    return key;
  }
  std::string codegen_options_fp = GetCodegenOptionsFingerprint(
      instr->GetModule()->config().debug_options());
  return absl::StrCat(key, "|", codegen_options_fp);
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
  return absl::UnimplementedError("Serialize is not implemented.");
}

absl::Status LocalCache::Deserialize(absl::string_view serialized_cache) {
  return absl::UnimplementedError("Deserialize is not implemented.");
}

AutotunerCacheInterface::CacheStats LocalCache::GetCacheStats() const {
  absl::MutexLock lock(storage_->mutex_);
  return storage_->stats_;
}

}  // namespace xla
