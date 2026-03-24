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

#include "xla/backends/autotuner/file_based_autotuner_cache.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/autotuner/autotuner_cache.pb.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/base64.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {

absl::StatusOr<std::unique_ptr<AutotunerCacheInterface>>
FileBasedAutotunerCache::Create(const FileBasedCacheConfig& cache_config) {
  auto cache = absl::WrapUnique(new FileBasedAutotunerCache(cache_config));
  if (!cache_config.autotune_cache_dir.empty()) {
    TF_RETURN_IF_ERROR(cache->Load());
  }
  return cache;
}

FileBasedAutotunerCache::FileBasedAutotunerCache(
    const FileBasedCacheConfig& cache_config)
    : cache_config_(cache_config), version_(cache_config.cache_version) {}

std::string FileBasedAutotunerCache::DeviceDescriptionToString(
    const se::DeviceDescription& device_desc) {
  std::string compute_capability;
  if (auto* ccc =
          device_desc.gpu_compute_capability().cuda_compute_capability()) {
    compute_capability = absl::StrCat("CUDA: ", ccc->major, ".", ccc->minor);
  } else {
    auto* rcc = device_desc.gpu_compute_capability().rocm_compute_capability();
    compute_capability = absl::StrCat("ROCM: ", rcc->gfx_version());
  }

  double memory_bandwidth = device_desc.memory_bandwidth() / 1e9;
  memory_bandwidth = std::round(memory_bandwidth);

  constexpr double kBytesPerMegabyte = 1 << 20;
  double l2_cache_size = device_desc.l2_cache_size() / kBytesPerMegabyte;

  return absl::StrCat(compute_capability, ", Cores: ", device_desc.core_count(),
                      ", GPU clock: ", device_desc.clock_rate_ghz(),
                      " GHz, Memory bandwidth: ", memory_bandwidth,
                      " GB/s, L2 cache: ", l2_cache_size, " MB");
}

namespace {
// We use SHA256 instead of the tsl fingerprint method to avoid collisions, as
// this is industry standard.
absl::StatusOr<std::string> GetBase64EncodedSha256Hash(absl::string_view s) {
  llvm::SHA256 sha256;
  sha256.update(llvm::StringRef(s.data(), s.size()));
  std::array<uint8_t, 32> hash = sha256.final();
  // C++ strict aliasing rules allow reinterpret casting to (const) char*.
  absl::string_view hash_view(reinterpret_cast<const char*>(hash.data()),
                              hash.size());
  std::string base64_encoded_hash;
  TF_RETURN_IF_ERROR(tsl::Base64Encode(hash_view, &base64_encoded_hash));
  return base64_encoded_hash;
}

absl::StatusOr<std::string> GetHloHash(const HloInstruction* instr) {
  auto options = HloPrintOptions::Fingerprint();
  options.set_print_backend_config(true);
  options.set_sort_backend_config(true);
  options.set_print_operand_shape(true);
  return GetBase64EncodedSha256Hash(instr->ToString(options));
}
}  // namespace

absl::StatusOr<std::string> FileBasedAutotunerCache::GetMapKey(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(const std::string hlo_hash, GetHloHash(instr));
  return absl::StrCat(hlo_hash, "|",
                      DeviceDescriptionToString(cache_config_.device_desc), "|",
                      version_);
}

absl::StatusOr<AutotunerCacheKey> FileBasedAutotunerCache::GetProtoKey(
    const HloInstruction* instr) {
  TF_ASSIGN_OR_RETURN(const std::string hlo_hash, GetHloHash(instr));
  AutotunerCacheKey key;
  key.set_hlo_fingerprint(hlo_hash);
  key.set_device_str(DeviceDescriptionToString(cache_config_.device_desc));
  key.set_version(version_);
  return key;
}

std::optional<AutotunerCacheInterface::Config> FileBasedAutotunerCache::Lookup(
    const HloInstruction* instr) {
  absl::StatusOr<std::string> map_key = GetMapKey(instr);
  if (!map_key.ok()) {
    LOG(ERROR) << "Failed to get map key: " << map_key.status();
    return std::nullopt;
  }
  absl::MutexLock lock(mutex_);
  auto it = in_memory_cache_.find(*map_key);
  if (it == in_memory_cache_.end()) {
    return std::nullopt;
  }
  AutotunerCacheInterface::Config config;
  if (!autotuner::Backend_Parse(it->second.codegen_backend(),
                                &config.codegen_backend)) {
    LOG(ERROR) << "Failed to parse codegen backend: "
               << it->second.codegen_backend();
    return std::nullopt;
  }
  config.backend_config = it->second.backend_config();
  return config;
}

absl::Status FileBasedAutotunerCache::Insert(
    const HloInstruction* instr,
    const AutotunerCacheInterface::Config& best_config) {
  if (cache_config_.autotune_cache_mode ==
      FileBasedCacheConfig::CacheMode::READ) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(const std::string map_key, GetMapKey(instr));
  TF_ASSIGN_OR_RETURN(AutotunerCacheKey proto_key, GetProtoKey(instr));
  absl::MutexLock lock(mutex_);
  AutotunerCacheEntry entry;
  *entry.mutable_key() = proto_key;
  entry.set_codegen_backend(Backend_Name(best_config.codegen_backend));
  *entry.mutable_backend_config() = best_config.backend_config;
  in_memory_cache_[map_key] = entry;
  if (!cache_config_.autotune_cache_dir.empty()) {
    return Save(map_key, entry);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> FileBasedAutotunerCache::GetCacheFilePath(
    absl::string_view map_key) {
  TF_ASSIGN_OR_RETURN(const std::string key_hash,
                      GetBase64EncodedSha256Hash(map_key));
  return tsl::io::JoinPath(cache_config_.autotune_cache_dir,
                           absl::StrCat(key_hash, ".textproto"));
}

std::string FileBasedAutotunerCache::GetCacheFilePattern() {
  return tsl::io::JoinPath(cache_config_.autotune_cache_dir, "*.textproto");
}

absl::Status FileBasedAutotunerCache::Load() {
  absl::MutexLock lock(mutex_);
  const std::string file_pattern = GetCacheFilePattern();
  VLOG(1) << "Loading autotuner cache from: " << file_pattern;

  std::vector<std::string> cache_files;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->GetMatchingPaths(file_pattern, &cache_files));

  if (cache_files.empty()) {
    VLOG(1) << "No autotuner cache files found.";
    return absl::OkStatus();
  }

  for (const auto& file_path : cache_files) {
    std::string proto_string;
    absl::Status status =
        tsl::ReadFileToString(tsl::Env::Default(), file_path, &proto_string);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to read autotuner cache file: " << file_path << ": "
                 << status;
      continue;
    }

    AutotunerCacheEntry entry;
    if (!tsl::protobuf::TextFormat::ParseFromString(proto_string, &entry)) {
      LOG(ERROR) << "Failed to parse autotuner cache file: " << file_path;
      continue;
    }

    const AutotunerCacheKey& key = entry.key();
    if (key.device_str() ==
            DeviceDescriptionToString(cache_config_.device_desc) &&
        key.version() == version_) {
      in_memory_cache_[absl::StrCat(key.hlo_fingerprint(), "|",
                                    key.device_str(), "|", key.version())] =
          entry;
    }
  }
  VLOG(1) << "Loaded " << in_memory_cache_.size()
          << " entries from autotuner cache.";
  return absl::OkStatus();
}

absl::Status FileBasedAutotunerCache::Save(absl::string_view map_key,
                                           const AutotunerCacheEntry& entry) {
  TF_ASSIGN_OR_RETURN(const std::string file_path, GetCacheFilePath(map_key));
  VLOG(1) << "Saving autotuner entry to: " << file_path;

  std::string proto_string;
  if (!tsl::protobuf::TextFormat::PrintToString(entry, &proto_string)) {
    return absl::InternalError("Failed to serialize autotuner entry.");
  }

  tsl::Env* default_env = tsl::Env::Default();
  TF_RETURN_IF_ERROR(
      default_env->RecursivelyCreateDir(cache_config_.autotune_cache_dir));

  // Rename trick: Write to a temporary file, then rename it to the final file
  // to avoid mingled files when multiple threads are writing to the same
  // file. Also avoids reading incomplete files.
  const std::string tmp_dir =
      tsl::io::JoinPath(cache_config_.autotune_cache_dir, "tmp");
  TF_RETURN_IF_ERROR(default_env->RecursivelyCreateDir(tmp_dir));

  TF_ASSIGN_OR_RETURN(const std::string key_hash,
                      GetBase64EncodedSha256Hash(map_key));
  int64_t time_stamp = default_env->NowNanos();
  const std::string temp_file_path = tsl::io::JoinPath(
      tmp_dir, absl::StrCat("tmp_cache_", key_hash, "_",
                            std::to_string(time_stamp), ".textproto"));

  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(default_env, temp_file_path, proto_string));
  TF_RETURN_IF_ERROR(default_env->RenameFile(temp_file_path, file_path));
  VLOG(1) << "Saved entry to autotuner cache: " << file_path;
  return absl::OkStatus();
}

}  // namespace xla
