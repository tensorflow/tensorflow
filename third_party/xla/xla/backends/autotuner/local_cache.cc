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
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
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
#include "tsl/platform/protobuf.h"

namespace xla {

namespace {
std::string GetHloFingerprintString(const HloInstruction& instr) {
  tsl::Fprint128 hlo_fp = GetHloFingerprint(instr);
  return absl::StrCat(absl::Hex(hlo_fp.high64, absl::kZeroPad16),
                      absl::Hex(hlo_fp.low64, absl::kZeroPad16));
}

std::string GetCodegenOptionsFingerprintString(const HloInstruction& instr) {
  if (instr.GetModule() == nullptr) {
    LOG(WARNING) << "Could not use DebugOptions for autotune cache key, "
                    "module is null for instruction: "
                 << instr.name();
    return "";
  }
  return GetCodegenOptionsFingerprint(
      instr.GetModule()->config().debug_options());
}
}  // namespace

LocalCacheStorage& LocalCacheStorage::GetInstance() {
  static auto* instance = new LocalCacheStorage();
  return *instance;
}

LocalCacheStorage& LocalCacheStorage::GetInstance(
    const AutotuneCacheContext& ctx) {
  return GetInstance();
}

std::optional<AutotunerCacheInterface::Config> LocalCacheStorage::Lookup(
    absl::string_view hlo_fingerprint,
    absl::FunctionRef<std::string()> codegen_options_fp_generator,
    const AutotuneCacheContext& context, KeyMatchingMode matching_mode) {
  absl::MutexLock lock(mutex_);
  auto it = entries_by_hlo_.find(hlo_fingerprint);
  if (it == entries_by_hlo_.end()) {
    stats_.misses++;
    return std::nullopt;
  }

  for (const autotuner::AutotuneEntry& entry : it->second) {
    // TargetKey matching
    if (entry.key().target().device() != context.device() ||
        entry.key().target().explicit_version() != context.explicit_version()) {
      continue;
    }

    if (matching_mode == KeyMatchingMode::kStrict) {
      if (entry.key().environment().codegen_version() ==
          context.codegen_version()) {
        std::string codegen_options_fp = codegen_options_fp_generator();
        if (entry.key().environment().codegen_options_fingerprint() ==
            codegen_options_fp) {
          stats_.hits++;
          return AutotunerCacheInterface::Config{
              entry.value().optimal_config().backend(),
              entry.value().optimal_config().backend_config()};
        }
      }
    } else if (matching_mode == KeyMatchingMode::kLoose) {
      autotuner::Backend backend = entry.value().optimal_config().backend();
      auto backend_it = context.per_backend_versions().find(backend);
      if (backend_it != context.per_backend_versions().end() &&
          backend_it->second == entry.value().optimal_backend_version()) {
        stats_.hits++;
        return AutotunerCacheInterface::Config{
            backend, entry.value().optimal_config().backend_config()};
      }
    }
  }

  stats_.misses++;
  return std::nullopt;
}

absl::Status LocalCacheStorage::Insert(const autotuner::AutotuneEntry& entry) {
  absl::MutexLock lock(mutex_);
  auto& entries = entries_by_hlo_[entry.key().target().hlo_fingerprint()];
  for (auto& existing : entries) {
    if (existing.key().target().device() == entry.key().target().device() &&
        existing.key().target().explicit_version() ==
            entry.key().target().explicit_version() &&
        existing.key().environment().codegen_version() ==
            entry.key().environment().codegen_version() &&
        existing.key().environment().codegen_options_fingerprint() ==
            entry.key().environment().codegen_options_fingerprint()) {
      *existing.mutable_value() = entry.value();
      return absl::OkStatus();
    }
  }
  entries.push_back(entry);
  return absl::OkStatus();
}

absl::Status LocalCacheStorage::Deserialize(
    absl::string_view serialized_cache) {
  autotuner::AutotuneCache proto_cache;
  if (!proto_cache.ParseFromString(serialized_cache)) {
    if (!tsl::protobuf::TextFormat::ParseFromString(
            std::string(serialized_cache), &proto_cache)) {
      return absl::InvalidArgumentError(
          "Failed to parse AutotuneCache proto (binary or textproto).");
    }
  }

  absl::MutexLock lock(mutex_);
  int deserialized_entries = 0;
  for (const autotuner::AutotuneEntry& entry : proto_cache.entries()) {
    auto& entries = entries_by_hlo_[entry.key().target().hlo_fingerprint()];
    bool updated = false;
    for (auto& existing : entries) {
      if (existing.key().target().device() == entry.key().target().device() &&
          existing.key().target().explicit_version() ==
              entry.key().target().explicit_version() &&
          existing.key().environment().codegen_version() ==
              entry.key().environment().codegen_version() &&
          existing.key().environment().codegen_options_fingerprint() ==
              entry.key().environment().codegen_options_fingerprint()) {
        *existing.mutable_value() = entry.value();
        updated = true;
        break;
      }
    }
    if (!updated) {
      entries.push_back(entry);
    }
    deserialized_entries++;
  }
  VLOG(1) << "LocalCacheStorage::Deserialize: Deserialized "
          << deserialized_entries << " entries, out of "
          << proto_cache.entries_size() << " entries in the cache.";
  return absl::OkStatus();
}

absl::StatusOr<std::string> LocalCacheStorage::Serialize(
    absl::Span<const std::string> hlo_fingerprints,
    const AutotuneCacheContext& context, KeyMatchingMode matching_mode) {
  autotuner::AutotuneCache proto_cache;
  proto_cache.set_device_scope(context.device());
  proto_cache.set_explicit_version_scope(context.explicit_version());

  absl::MutexLock lock(mutex_);
  auto add_matching_entries =
      [&](const std::vector<autotuner::AutotuneEntry>& candidate_entries) {
        for (const auto& entry : candidate_entries) {
          if (entry.key().target().device() != context.device() ||
              entry.key().target().explicit_version() !=
                  context.explicit_version()) {
            continue;
          }

          if (matching_mode == KeyMatchingMode::kStrict) {
            if (entry.key().environment().codegen_version() ==
                context.codegen_version()) {
              *proto_cache.add_entries() = entry;
            }
          } else if (matching_mode == KeyMatchingMode::kLoose) {
            autotuner::Backend backend =
                entry.value().optimal_config().backend();
            auto backend_it = context.per_backend_versions().find(backend);
            if (backend_it != context.per_backend_versions().end() &&
                backend_it->second == entry.value().optimal_backend_version()) {
              *proto_cache.add_entries() = entry;
            }
          }
        }
      };

  if (!hlo_fingerprints.empty()) {
    for (const std::string& fp : hlo_fingerprints) {
      auto it = entries_by_hlo_.find(fp);
      if (it != entries_by_hlo_.end()) {
        add_matching_entries(it->second);
      }
    }
  } else {
    for (const auto& [fp, candidate_entries] : entries_by_hlo_) {
      add_matching_entries(candidate_entries);
    }
  }

  std::string serialized;
  if (!proto_cache.SerializeToString(&serialized)) {
    return absl::InternalError("Failed to serialize AutotuneCache proto.");
  }
  return serialized;
}

AutotunerCacheInterface::CacheStats LocalCacheStorage::GetCacheStats() const {
  absl::MutexLock lock(mutex_);
  return stats_;
}

void LocalCacheStorage::Clear() {
  absl::MutexLock lock(mutex_);
  entries_by_hlo_.clear();
  stats_ = {};
}

LocalCache::LocalCache(AutotuneCacheContext context,
                       KeyMatchingMode matching_mode)
    : matching_mode_(matching_mode),
      context_(std::move(context)),
      storage_(&LocalCacheStorage::GetInstance()) {}

LocalCache::LocalCache(AutotuneCacheContext context,
                       KeyMatchingMode matching_mode,
                       LocalCacheStorage* absl_nonnull storage)
    : matching_mode_(matching_mode),
      context_(std::move(context)),
      storage_(storage) {
  CHECK(storage_ != nullptr) << "LocalCacheStorage cannot be null.";
}

std::optional<AutotunerCacheInterface::Config> LocalCache::Lookup(
    const HloInstruction* instr) {
  if (instr == nullptr) {
    return std::nullopt;
  }
  std::string hlo_fp = GetHloFingerprintString(*instr);
  return storage_->Lookup(
      hlo_fp, [instr]() { return GetCodegenOptionsFingerprintString(*instr); },
      context_, matching_mode_);
}

absl::Status LocalCache::Insert(const HloInstruction* instr,
                                const Config& config) {
  if (instr == nullptr) {
    return absl::InvalidArgumentError("Instruction cannot be null.");
  }
  std::string hlo_fp = GetHloFingerprintString(*instr);
  std::string codegen_options_fp = GetCodegenOptionsFingerprintString(*instr);

  autotuner::AutotuneEntry entry;
  auto* target = entry.mutable_key()->mutable_target();
  target->set_hlo_fingerprint(hlo_fp);
  target->set_device(context_.device());
  if (!context_.explicit_version().empty()) {
    target->set_explicit_version(context_.explicit_version());
  }

  auto* env = entry.mutable_key()->mutable_environment();
  env->set_codegen_version(context_.codegen_version());
  if (!codegen_options_fp.empty()) {
    env->set_codegen_options_fingerprint(codegen_options_fp);
  }

  auto* value = entry.mutable_value();
  value->mutable_optimal_config()->set_backend(config.codegen_backend);
  *value->mutable_optimal_config()->mutable_backend_config() =
      config.backend_config;

  auto it = context_.per_backend_versions().find(config.codegen_backend);
  if (it != context_.per_backend_versions().end()) {
    value->set_optimal_backend_version(it->second);
  }

  return storage_->Insert(entry);
}

absl::StatusOr<std::string> LocalCache::Serialize(
    absl::Span<const HloInstruction* const> instructions_to_serialize) {
  std::vector<std::string> fingerprints;
  fingerprints.reserve(instructions_to_serialize.size());
  for (const HloInstruction* instr : instructions_to_serialize) {
    if (instr != nullptr) {
      fingerprints.push_back(GetHloFingerprintString(*instr));
    }
  }
  return storage_->Serialize(fingerprints, context_, matching_mode_);
}

absl::Status LocalCache::Deserialize(absl::string_view serialized_cache) {
  return storage_->Deserialize(serialized_cache);
}

AutotunerCacheInterface::CacheStats LocalCache::GetCacheStats() const {
  return storage_->GetCacheStats();
}

}  // namespace xla
