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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

// Helper to create an IFRT array sharded across the given devices.
absl::StatusOr<ArrayRef> MakeShardedArray(Client* client,
                                          absl::Span<Device* const> devices,
                                          MemoryKind memory_kind,
                                          int64_t per_device_bytes) {
  const int64_t num_devices = static_cast<int64_t>(devices.size());
  ASSIGN_OR_RETURN(DeviceListRef device_list, client->MakeDeviceList(devices));

  DType dtype(DType::kF32);
  const int64_t bytes_per_element = *dtype.byte_size();
  int64_t per_device_elements = per_device_bytes / bytes_per_element;

  const int64_t total_elements = per_device_elements * num_devices;
  Shape shape({total_elements});
  Shape shard_shape({per_device_elements});

  ShardingRef sharding = xla::ifrt::HloSharding::Create(
      device_list, memory_kind, xla::HloSharding::IotaTile({num_devices}));

  ASSIGN_OR_RETURN(PrimitiveType element_type, ToPrimitiveType(dtype));
  const Literal literal =
      LiteralUtil::CreateFromDimensions(element_type, shard_shape.dims());

  absl::InlinedVector<int64_t, 1> shard_indices(num_devices);
  for (int64_t i = 0; i < num_devices; ++i) {
    shard_indices[i] = i;
  }

  Client::HostBuffer host_buffer{
      /*data=*/literal.untyped_data(),
      /*dtype=*/dtype,
      /*shape=*/shard_shape,
      /*byte_strides=*/std::nullopt,
      /*on_done=*/nullptr,
  };

  ArraySpec array_spec{
      /*dtype=*/dtype,
      /*shape=*/shape,
      /*sharding=*/sharding,
  };

  Client::MakeArraysFromHostBufferShardsSpec spec{
      /*buffers=*/{{std::move(shard_indices), std::move(host_buffer)}},
      /*array_spec=*/std::move(array_spec),
  };

  ASSIGN_OR_RETURN(std::vector<ArrayRef> arrays,
                   client->MakeArraysFromHostBufferShards(
                       absl::MakeSpan(&spec, 1),
                       Client::HostBufferSemantics::kImmutableOnlyDuringCall));
  // Block on each array construction to reduce the memory use.
  arrays[0]->GetReadyFuture().Await().IgnoreError();
  return std::move(arrays[0]);
}

Client* GetSharedClient() {
  static absl::NoDestructor<absl::StatusOr<std::shared_ptr<Client>>> client(
      test_util::GetClient());
  CHECK(client->ok()) << "Failed to get IFRT client: " << client->status();
  return client->value().get();
}

void BM_HashValues(benchmark::State& state, Client::HashMode hash_mode,
                   MemoryKind memory_kind) {
  const int64_t num_arrays = state.range(0);
  const int64_t num_devices = state.range(1);
  const int64_t per_device_bytes = state.range(2);

  Client* client = GetSharedClient();
  CHECK(client != nullptr) << "Failed to obtain IFRT client";

  absl::Span<Device* const> available_devices = client->addressable_devices();
  if (static_cast<int64_t>(available_devices.size()) < num_devices) {
    CHECK(false) << "Not enough devices available for benchmark. Requested: "
                 << num_devices << ", available: " << available_devices.size();
  }

  absl::Span<Device* const> selected_devices =
      available_devices.subspan(0, num_devices);

  std::vector<ValueRef> values;
  values.reserve(num_arrays);
  for (int64_t i = 0; i < num_arrays; ++i) {
    absl::StatusOr<ArrayRef> array = MakeShardedArray(
        client, selected_devices, memory_kind, per_device_bytes);
    CHECK(array.ok()) << "Failed to create sharded array: " << array.status();
    values.push_back(*std::move(array));
  }

  absl::Status ready_status =
      client->GetReadyFuture(absl::MakeSpan(values)).Await();
  if (!ready_status.ok()) {
    LOG(ERROR) << "Values ready future failed: " << ready_status;
    return;
  }

  for (auto _ : state) {
    tsl::Future<std::vector<uint64_t>> future =
        client->HashValues(absl::MakeSpan(values), hash_mode);
    absl::StatusOr<std::vector<uint64_t>> hash = future.Await();
    CHECK(hash.ok()) << "HashValues failed: " << hash.status();
  }

  const int64_t total_bytes_per_iteration =
      num_arrays * per_device_bytes * num_devices;
  state.SetBytesProcessed(state.iterations() * total_bytes_per_iteration);
  state.SetItemsProcessed(state.iterations() * num_arrays);
}

void CustomBenchmarkArgs(benchmark::internal::Benchmark* b) {
  for (int num_arrays : {1, 16}) {
    for (int num_devices : {1, 4}) {
      for (int64_t size : {1 << 10, 1 << 20, 64 << 20}) {
        b->Args({num_arrays, num_devices, static_cast<int>(size)});
      }
    }
  }
}

static const MemoryKind kDeviceMemoryKind("device");
static const MemoryKind kPinnedHostMemoryKind("pinned_host");

void BM_HashValues_Physical_Device(benchmark::State& state) {
  BM_HashValues(state, Client::HashMode::kPhysical, kDeviceMemoryKind);
}

void BM_HashValues_Physical_PinnedHost(benchmark::State& state) {
  BM_HashValues(state, Client::HashMode::kPhysical, kPinnedHostMemoryKind);
}

void BM_HashValues_Logical_Device(benchmark::State& state) {
  BM_HashValues(state, Client::HashMode::kLogical, kDeviceMemoryKind);
}

void BM_HashValues_Logical_PinnedHost(benchmark::State& state) {
  BM_HashValues(state, Client::HashMode::kLogical, kPinnedHostMemoryKind);
}

BENCHMARK(BM_HashValues_Physical_Device)
    ->UseRealTime()
    ->Apply(CustomBenchmarkArgs);
BENCHMARK(BM_HashValues_Physical_PinnedHost)
    ->UseRealTime()
    ->Apply(CustomBenchmarkArgs);
BENCHMARK(BM_HashValues_Logical_Device)
    ->UseRealTime()
    ->Apply(CustomBenchmarkArgs);
BENCHMARK(BM_HashValues_Logical_PinnedHost)
    ->UseRealTime()
    ->Apply(CustomBenchmarkArgs);

}  // namespace
}  // namespace ifrt
}  // namespace xla
