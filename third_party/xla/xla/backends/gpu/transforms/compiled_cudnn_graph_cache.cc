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

#include "xla/backends/gpu/transforms/compiled_cudnn_graph_cache.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/lib/strings/proto_serialization.h"

namespace xla {
namespace gpu {

DeviceFingerprint::DeviceFingerprint(
    const stream_executor::DeviceDescription& device_info) {
  stream_executor::GpuDeviceInfoProto proto = device_info.ToProto();
  proto.clear_pci_bus_id();
  proto.clear_numa_node();
  proto.clear_device_interconnect_info();
  // The proto contains maps, so the serialization must be deterministic.
  CHECK(tsl::SerializeToStringDeterministic(proto, &value_));
}

std::optional<CompiledCudnnGraph> CompiledCudnnGraphCache::Lookup(
    const CompiledCudnnGraphCacheKey& key) const {
  absl::MutexLock lock(mu_);
  auto it = graphs_.find(key);
  if (it == graphs_.end()) {
    return std::nullopt;
  }
  return it->second;
}

void CompiledCudnnGraphCache::Insert(CompiledCudnnGraphCacheKey key,
                                     CompiledCudnnGraph graph) {
  const int64_t entry_size_bytes = key.device_fingerprint.value().size() +
                                   key.computation_fingerprint.value.size() +
                                   graph.serialized_graph.size();
  if (entry_size_bytes > max_size_bytes_) {
    return;
  }
  absl::MutexLock lock(mu_);
  if (graphs_.contains(key)) {
    return;
  }
  if (size_bytes_ + entry_size_bytes > max_size_bytes_) {
    graphs_.clear();
    size_bytes_ = 0;
  }
  graphs_.emplace(std::move(key), std::move(graph));
  size_bytes_ += entry_size_bytes;
}

int64_t CompiledCudnnGraphCache::size() const {
  absl::MutexLock lock(mu_);
  return graphs_.size();
}

}  // namespace gpu
}  // namespace xla
