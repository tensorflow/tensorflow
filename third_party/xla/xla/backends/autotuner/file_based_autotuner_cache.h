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

#ifndef XLA_BACKENDS_AUTOTUNER_FILE_BASED_AUTOTUNER_CACHE_H_
#define XLA_BACKENDS_AUTOTUNER_FILE_BASED_AUTOTUNER_CACHE_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/autotuner/autotune_config.h"
#include "xla/backends/autotuner/autotuner_cache.pb.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/device_description.h"

namespace xla {

// File-based implementation of the AutotunerCacheInterface.
// This class stores autotuner cache entries as textproto files in a directory
// specified by AutotuneConfig.autotune_cache_dir. It supports any file system
// accessible via tsl::Env.
//
// Each cache entry is stored in a separate file. The filename is a SHA256 hash
// of the cache key, which includes the HLO instruction, device description,
// and cache version. This design prevents large cache files and minimizes
// corruption risks.
//
// To ensure atomicity and prevent reading incomplete files, writes are first
// directed to a temporary file in a "tmp" subdirectory. The temporary file is
// then renamed to its final destination.
//
// Key Management:
// - In-memory map: Uses a string key "hlo_hash|device_str|version" for
//   efficient lookups in an absl::flat_hash_map.
// - Persistent storage: Uses an AutotunerCacheKey protobuf message for on-disk
//   serialization, enabling structured and language-independent data access.
class FileBasedAutotunerCache : public AutotunerCacheInterface {
 public:
  static absl::StatusOr<std::unique_ptr<AutotunerCacheInterface>> Create(
      const AutotuneConfig& autotune_config,
      const se::DeviceDescription& device_desc, const std::string& version);

  std::optional<AutotunerCacheEntry> Lookup(
      const HloInstruction* instr) override ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status Insert(const HloInstruction* instr,
                      AutotunerCacheEntry& entry) override
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  FileBasedAutotunerCache(const AutotuneConfig& autotune_config,
                          const se::DeviceDescription& device_desc,
                          const std::string& version);

  static std::string DeviceDescriptionToString(
      const se::DeviceDescription& device_desc);

  absl::StatusOr<std::string> GetMapKey(const HloInstruction* instr);

  absl::StatusOr<AutotunerCacheKey> GetProtoKey(const HloInstruction* instr);

  absl::StatusOr<std::string> GetCacheFilePath(absl::string_view map_key);

  std::string GetCacheFilePattern();

  absl::Status Load() ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status Save(absl::string_view map_key, const AutotunerCacheEntry& entry)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  AutotuneConfig autotune_config_;
  const se::DeviceDescription device_desc_;
  const std::string version_;
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, AutotunerCacheEntry> in_memory_cache_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_FILE_BASED_AUTOTUNER_CACHE_H_
