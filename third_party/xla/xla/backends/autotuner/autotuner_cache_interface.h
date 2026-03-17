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
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/autotuner_cache.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// Class to manage the autotuning cache.
// Note: We do not enforce cross implementation compatibility. E.g. different
// AutotunerCacheInterface implementations may have different cache keys.
class AutotunerCacheInterface {
 public:
  // Serializable config. Will be changed to a proto in the future.
  struct Config {
    autotuner::Backend codegen_backend;
    google::protobuf::Any backend_config;
  };

  struct CacheStats {
    int64_t hits = 0;
    int64_t misses = 0;
  };

  virtual ~AutotunerCacheInterface() = default;

  virtual std::optional<Config> Lookup(const HloInstruction* instr) = 0;

  virtual absl::Status Insert(const HloInstruction* instr,
                              const Config& best_config) = 0;

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
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNER_CACHE_INTERFACE_H_
