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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla {
namespace gpu {
namespace {

stream_executor::DeviceDescription H100() {
  stream_executor::DeviceDescription device;
  device.set_name("NVIDIA H100");
  device.set_gpu_compute_capability(
      stream_executor::CudaComputeCapability(9, 0));
  device.set_core_count(132);
  device.set_pci_bus_id("0000:01:00.0");
  return device;
}

CompiledCudnnGraphCacheKey Key(std::string computation) {
  return CompiledCudnnGraphCacheKey{
      DeviceFingerprint(H100()),
      /*deviceless=*/false,
      /*require_determinism=*/false,
      /*allow_tf32=*/false,
      /*force_tensor_ir=*/false,
      /*plan_id=*/0,
      ComputationFingerprint{std::move(computation)}};
}

CompiledCudnnGraph Graph(int64_t plan_id, int size = 16) {
  return CompiledCudnnGraph{plan_id, /*workspace_size=*/128,
                            std::string(size, 'x')};
}

TEST(CompiledCudnnGraphCacheTest, LookupReturnsInsertedGraph) {
  CompiledCudnnGraphCache cache;
  EXPECT_EQ(cache.Lookup(Key("a")), std::nullopt);
  cache.Insert(Key("a"), Graph(3));
  cache.Insert(Key("a"), Graph(4));  // Does not overwrite.

  std::optional<CompiledCudnnGraph> graph = cache.Lookup(Key("a"));
  ASSERT_TRUE(graph.has_value());
  EXPECT_EQ(graph->plan_id, 3);
  EXPECT_EQ(cache.Lookup(Key("b")), std::nullopt);
  EXPECT_EQ(cache.size(), 1);
}

TEST(CompiledCudnnGraphCacheTest, AllKeyFieldsDistinguishEntries) {
  CompiledCudnnGraphCache cache;
  cache.Insert(Key("a"), Graph(0));
  for (auto modify : std::vector<void (*)(CompiledCudnnGraphCacheKey&)>{
           [](auto& k) {
             stream_executor::DeviceDescription other = H100();
             other.set_core_count(66);
             k.device_fingerprint = DeviceFingerprint(other);
           },
           [](auto& k) { k.deviceless = true; },
           [](auto& k) { k.require_determinism = true; },
           [](auto& k) { k.allow_tf32 = true; },
           [](auto& k) { k.force_tensor_ir = true; },
           [](auto& k) { k.plan_id = 1; },
           [](auto& k) { k.plan_id = std::nullopt; },
           [](auto& k) { k.computation_fingerprint.value = "b"; },
       }) {
    CompiledCudnnGraphCacheKey key = Key("a");
    modify(key);
    EXPECT_EQ(cache.Lookup(key), std::nullopt);
  }
  EXPECT_NE(cache.Lookup(Key("a")), std::nullopt);
}

TEST(CompiledCudnnGraphCacheTest, ClearsWhenLimitIsExceeded) {
  const int64_t entry_size =
      DeviceFingerprint(H100()).value().size() + 1 + 1000;
  CompiledCudnnGraphCache cache(2 * entry_size);
  cache.Insert(Key("a"), Graph(0, 1000));
  cache.Insert(Key("b"), Graph(0, 1000));
  EXPECT_EQ(cache.size(), 2);

  cache.Insert(Key("c"), Graph(0, 1000));

  EXPECT_EQ(cache.size(), 1);
  EXPECT_NE(cache.Lookup(Key("c")), std::nullopt);
}

TEST(CompiledCudnnGraphCacheTest, DeviceFingerprintIgnoresDeviceIdentity) {
  stream_executor::DeviceDescription same_kind = H100();
  same_kind.set_pci_bus_id("0000:02:00.0");
  same_kind.set_numa_node(1);
  stream_executor::DeviceDescription fewer_cores = H100();
  fewer_cores.set_core_count(66);
  stream_executor::DeviceDescription other_dnn = H100();
  other_dnn.set_dnn_version(stream_executor::SemanticVersion(9, 23, 0));

  EXPECT_EQ(DeviceFingerprint(same_kind), DeviceFingerprint(H100()));
  EXPECT_NE(DeviceFingerprint(fewer_cores), DeviceFingerprint(H100()));
  EXPECT_NE(DeviceFingerprint(other_dnn), DeviceFingerprint(H100()));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
