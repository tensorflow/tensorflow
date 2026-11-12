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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_COMPILED_CUDNN_GRAPH_CACHE_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_COMPILED_CUDNN_GRAPH_CACHE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Fingerprint of all properties of a device that can affect building a cuDNN
// graph, excluding properties that identify a particular device of the same
// kind (e.g. its PCI bus ID).
class DeviceFingerprint {
 public:
  explicit DeviceFingerprint(
      const stream_executor::DeviceDescription& device_info);

  const std::string& value() const { return value_; }

  bool operator==(const DeviceFingerprint& other) const {
    return value_ == other.value_;
  }

  template <typename H>
  friend H AbslHashValue(H h, const DeviceFingerprint& f) {
    return H::combine(std::move(h), f.value_);
  }

 private:
  std::string value_;
};

// Fingerprint of a fused computation.
struct ComputationFingerprint {
  std::string value;

  bool operator==(const ComputationFingerprint& other) const {
    return value == other.value;
  }

  template <typename H>
  friend H AbslHashValue(H h, const ComputationFingerprint& f) {
    return H::combine(std::move(h), f.value);
  }
};

// A built and serialized cuDNN graph.
struct CompiledCudnnGraph {
  int64_t plan_id = 0;
  int64_t workspace_size = 0;
  std::string serialized_graph;
};

// Identifies a built cuDNN graph: contains everything that affects building
// the graph of a fusion.
struct CompiledCudnnGraphCacheKey {
  DeviceFingerprint device_fingerprint;
  // Whether the graph is built without a cuDNN handle.
  bool deviceless = false;
  bool require_determinism = false;
  bool allow_tf32 = false;
  bool force_tensor_ir = false;
  // The requested execution plan, or nullopt for the first plan that builds.
  std::optional<int64_t> plan_id;
  ComputationFingerprint computation_fingerprint;

 private:
  // Defined before its users, since its return type is deduced.
  auto Tie() const {
    return std::tie(device_fingerprint, deviceless, require_determinism,
                    allow_tf32, force_tensor_ir, plan_id,
                    computation_fingerprint);
  }

 public:
  bool operator==(const CompiledCudnnGraphCacheKey& other) const {
    return Tie() == other.Tie();
  }

  template <typename H>
  friend H AbslHashValue(H h, const CompiledCudnnGraphCacheKey& key) {
    return H::combine(std::move(h), key.Tie());
  }
};

// Thread-safe cache of built cuDNN graphs. Building a cuDNN graph is
// expensive, and the same fusion is usually built several times: by the
// autotuner and by the CuDnnFusionCompiler run in GpuCompiler::RunBackend, and
// again for every compilation of the same module. The cache is cleared when
// inserting an entry would exceed `max_size_bytes`.
class CompiledCudnnGraphCache {
 public:
  explicit CompiledCudnnGraphCache(int64_t max_size_bytes = int64_t{256} << 20)
      : max_size_bytes_(max_size_bytes) {}

  std::optional<CompiledCudnnGraph> Lookup(
      const CompiledCudnnGraphCacheKey& key) const;

  // Does nothing if `key` is already cached.
  void Insert(CompiledCudnnGraphCacheKey key, CompiledCudnnGraph graph);

  // Returns the number of cached graphs.
  int64_t size() const;

 private:
  int64_t max_size_bytes_;
  mutable absl::Mutex mu_;
  absl::flat_hash_map<CompiledCudnnGraphCacheKey, CompiledCudnnGraph> graphs_
      ABSL_GUARDED_BY(mu_);
  int64_t size_bytes_ ABSL_GUARDED_BY(mu_) = 0;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_COMPILED_CUDNN_GRAPH_CACHE_H_
