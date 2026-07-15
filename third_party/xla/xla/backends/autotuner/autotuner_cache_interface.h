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

#ifndef XLA_BACKENDS_AUTOTUNER_AUTOTUNER_CACHE_INTERFACE_H_
#define XLA_BACKENDS_AUTOTUNER_AUTOTUNER_CACHE_INTERFACE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/stream_executor/device_description.h"

namespace xla {

enum class KeyMatchingMode {
  // Matched the entire AutotuneKey.
  kStrict,
  // Matches only on `AutotuneTargetKey` (ignoring `AutotuneEnvironmentKey`).
  // Loose key matching allows retrieving cache entries even if the general
  // codegen version or options differ, as long as the specific backend version
  // used by the cached optimal configuration matches the current backend.
  //
  // Although this guarantees a compatible, working configuration, it might
  // be suboptimal. For example, if an instruction is supported by backends
  // A and B:
  // - The cached optimal config is for backend A.
  // - If backend B has since improved enough to outperform backend A, a loose
  //   match will still reuse backend A's configuration.
  kLoose,
};

enum class CacheMode {
  kReadOnly,    // Lookup only (e.g., immutable cache formats)
  kReadAppend,  // Lookup and Append (does not update existing entries)
  kReadWrite,   // Lookup, Insert, and update existing entries.
};

// AutotuneCacheContext defines the compilation and hardware context used for
// the autotuning cache lookup/insert operations (such as the target device
// description and active codegen backend versions). Because this context
// remains invariant during the lifetime of a compiler process, it is
// passed directly to the constructors of cache implementations. Cache
// implementations use this to construct keys and match entries.
class AutotuneCacheContext {
 public:
  // Creates an AutotuneCacheContext from the given device description and
  // codegen backends. Optional explicit version can be provided to further
  // differentiate cache entries, and is accepted by all cache implementations.
  static AutotuneCacheContext Create(
      const stream_executor::DeviceDescription& device_description,
      absl::Span<const std::unique_ptr<CodegenBackend>> backends,
      std::string explicit_version = "");

  // Returns a unique identifier for the cache context.
  std::string GetId() const;

  bool operator==(const AutotuneCacheContext& other) const;
  bool operator!=(const AutotuneCacheContext& other) const;

  const std::string& device() const { return device_; }
  const std::string& explicit_version() const { return explicit_version_; }
  const std::string& codegen_version() const { return codegen_version_; }
  // TODO(b/444398084): change to an api which returns version for a given
  // backend.
  const absl::flat_hash_map<autotuner::Backend, std::string>&
  per_backend_versions() const {
    return per_backend_versions_;
  }

 private:
  // Private constructor to enforce creation via the static Create method.
  AutotuneCacheContext(
      std::string device, std::string explicit_version,
      std::string codegen_version,
      absl::flat_hash_map<autotuner::Backend, std::string> per_backend_versions)
      : device_(std::move(device)),
        explicit_version_(std::move(explicit_version)),
        codegen_version_(std::move(codegen_version)),
        per_backend_versions_(std::move(per_backend_versions)) {}

  std::string device_;
  std::string explicit_version_;
  std::string codegen_version_;
  // Already represented by the combined codegen, but we maintain them
  // separately for backend specific version matching.
  absl::flat_hash_map<autotuner::Backend, std::string> per_backend_versions_;
};

// AutotunerCacheInterface is an interface for managing autotuning cache.
// It provides methods for looking up and inserting configs, serializing and
// deserializing the cache, and retrieving cache statistics and mode.
// Cross implementation cache lookup/insert and serialize/deserialize are not
// compatible.
class AutotunerCacheInterface {
 public:
  // Serializable config. Will be changed to a proto in the future.
  struct Config {
    autotuner::Backend codegen_backend;
    autotuner::BackendConfig backend_config;
  };

  struct CacheStats {
    int64_t hits = 0;
    int64_t misses = 0;
  };

  virtual ~AutotunerCacheInterface() = default;

  // TODO(b/444398084): Migrate to use StatusOr<std::optional<Config>>.
  virtual std::optional<Config> Lookup(const HloInstruction* instr) = 0;

  virtual absl::Status Insert(const HloInstruction* instr,
                              const Config& config) = 0;

  virtual CacheStats GetCacheStats() const = 0;

  // Serializes the cache to a string. If instructions are provided, only the
  // cache entries corresponding to the instructions will be serialized,
  // otherwise all cache entries will be serialized.
  virtual absl::StatusOr<std::string> Serialize(
      absl::Span<const HloInstruction* const> instructions_to_serialize) {
    return absl::UnimplementedError("Serialize is not implemented.");
  };

  // Deserializes the string and updates the cache, overwriting the keys if they
  // already exist.
  virtual absl::Status Deserialize(absl::string_view serialized_cache) {
    return absl::UnimplementedError("Deserialize is not implemented.");
  };
  // TODO(b/444398084): Remove default implementations once all implementations
  // are updated.
  virtual CacheMode GetMode() const { return CacheMode::kReadWrite; };
  virtual KeyMatchingMode GetKeyMatchingMode() const {
    return KeyMatchingMode::kStrict;
  };
};

// A no-op implementation of the autotuner cache.
class NoOpAutotunerCache : public AutotunerCacheInterface {
 public:
  std::optional<Config> Lookup(const HloInstruction* instr) override {
    return std::nullopt;
  }

  absl::Status Insert(const HloInstruction* instr,
                      const Config& config) override {
    return absl::OkStatus();
  }

  CacheStats GetCacheStats() const override { return {}; }

  absl::StatusOr<std::string> Serialize(
      absl::Span<const HloInstruction* const> instructions_to_serialize)
      override {
    return "";
  }

  absl::Status Deserialize(absl::string_view serialized_cache) override {
    return absl::OkStatus();
  }
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNER_CACHE_INTERFACE_H_
