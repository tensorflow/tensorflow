/* Copyright 2022 The OpenXLA Authors.

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
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::tsl::testing::StatusIs;

// Returns a list of non-addressable devices in the client.
std::vector<Device*> GetNonAddressableDevices(Client* client) {
  std::vector<Device*> devices;
  for (auto* device : client->devices()) {
    if (!device->IsAddressable()) {
      devices.push_back(device);
    }
  }
  return devices;
}

TEST(ArrayImplTest, MakeArrayFromHostBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data->data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/nullptr));

  EXPECT_EQ(array->dtype(), dtype);
  EXPECT_EQ(array->shape(), shape);
  EXPECT_EQ(array->shared_ptr_sharding().get(), sharding.get());
}

TEST(ArrayImplTest,
     MakeArrayFromHostBufferWithAddressableAndNonAddressableDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  std::vector<Device*> non_addressable_devices =
      GetNonAddressableDevices(client.get());
  if (non_addressable_devices.empty()) {
    GTEST_SKIP() << "Skipping test; needs at least 1 non-addressable device.";
  }

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);

  std::vector<Device*> devices;
  devices.reserve(2);
  devices.push_back(non_addressable_devices.at(0));
  devices.push_back(client->addressable_devices().at(0));
  ShardingRef sharding = xla::ifrt::ConcreteEvenSharding::Create(
      client->MakeDeviceList(devices), xla::ifrt::MemoryKind(), shape,
      /*shard_shape=*/shape,
      /*is_fully_replicated=*/true);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data->data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/nullptr));

  EXPECT_EQ(array->dtype(), dtype);
  EXPECT_EQ(array->shape(), shape);
  EXPECT_EQ(array->shared_ptr_sharding().get(), sharding.get());
}

class ArrayImplWithHostBufferSemanticsTest
    : public testing::TestWithParam<Client::HostBufferSemantics> {};

TEST_P(ArrayImplWithHostBufferSemanticsTest,
       MakeArrayFromHostBufferCallsWithOnDoneWithHostBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Client::HostBufferSemantics semantics = GetParam();

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  absl::Notification done_with_host_buffer;
  auto on_done_with_host_buffer = [&]() { done_with_host_buffer.Notify(); };

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data->data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      std::move(on_done_with_host_buffer)));

  // Regardless of the host buffer semantics chosen, the host buffer must not be
  // used by the runtime once `on_done_with_host_buffer` has been called.
  if (semantics == Client::HostBufferSemantics::kImmutableZeroCopy) {
    // `on_done_with_host_buffer` is called only when the `Array` is destroyed
    // if the runtime implements `kZeroCopy`. A deadlock will occur if we keep
    // the `Array` instance.
    array.reset();

    // `done_with_host_buffer` is very likely to have been called after
    // sleeping. This method has false positives (sleeping was not long enough
    // for the callback to be called asynchronously), but it may greatly
    // increases the chance of detecting an incorrect implementation as a form
    // of test flakes.
    absl::SleepFor(absl::Seconds(3));
    ASSERT_TRUE(done_with_host_buffer.HasBeenNotified());
  } else {
    done_with_host_buffer.WaitForNotification();
  }
  data = nullptr;
  // There should be no use-after-free.
}

INSTANTIATE_TEST_CASE_P(
    AllHostBufferSemantics, ArrayImplWithHostBufferSemanticsTest,
    testing::Values(
        Client::HostBufferSemantics::kImmutableOnlyDuringCall,
        Client::HostBufferSemantics::kImmutableUntilTransferCompletes,
        Client::HostBufferSemantics::kImmutableZeroCopy));

TEST(ArrayImplTest, MakeArrayFromHostBufferImmutableOnlyDuringCall) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  absl::Notification done_with_host_buffer;
  auto on_done_with_host_buffer = [&]() {
    // Sleeping facilitates testing if `MakeArrayFromHostBuffer()` calls
    // `on_done_with_host_buffer` synchronously before returning. This method
    // has false negatives (when a call to
    // `done_with_host_buffer.HasBeenNotified()` is delayed), but it may greatly
    // increases the chance of detecting an incorrect implementation as a form
    // of test flakes.
    absl::SleepFor(absl::Seconds(3));

    done_with_host_buffer.Notify();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data->data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      std::move(on_done_with_host_buffer)));

  // `on_done_with_host_buffer` should have been called before returning from
  // `MakeArrayFromHostBuffer`.
  ASSERT_TRUE(done_with_host_buffer.HasBeenNotified());
  data = nullptr;
  // There should be no use-after-free.
}

TEST(ArrayImplTest, MakeArrayFromHostBufferImmutableUntilTransferCompletes) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      client->MakeArrayFromHostBuffer(
          data->data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding,
          Client::HostBufferSemantics::kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr));

  // Once the `Array` has become ready, the host buffer is not accessed.
  TF_ASSERT_OK(array->GetReadyFuture().Await());
  data = nullptr;
  // There should be no use-after-free.
}

TEST(ArrayImplTest, MakeArrayFromHostBufferZeroCopy) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(auto array,
                          client->MakeArrayFromHostBuffer(
                              data->data(), dtype, shape,
                              /*byte_strides=*/std::nullopt, sharding,
                              Client::HostBufferSemantics::kImmutableZeroCopy,
                              /*on_done_with_host_buffer=*/nullptr));

  // The `Array` may alias the host buffer, but once the transfer is done and
  // the `Array` is destroyed, the host buffer is not accessed. This test would
  // pass trivially on the implementations that downgrade `kZeroCopy`, if
  // `MakeArrayFromHostBufferImmutableUntilTransferCompletes` already passes.
  TF_ASSERT_OK(array->GetReadyFuture().Await());
  array.reset();
  data = nullptr;
  // There should be no use-after-free.
}

TEST(ArrayImplTest, MakeArrayFromHostBufferAndCopyToHostBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  std::vector<float> out_data(6);
  auto future =
      array->CopyToHostBuffer(out_data.data(), /*byte_strides=*/std::nullopt,
                              ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());
  EXPECT_THAT(out_data, ElementsAreArray(data));
}

TEST(ArrayImplTest, MakeArrayFromHostBufferWithByteStridesAndCopyToHostBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  // The input data layout is minor-to-major.
  std::vector<float> data = {0, 3, 1, 4, 2, 5};
  std::vector<int64_t> byte_strides = {4, 8};
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape, byte_strides, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  std::vector<float> out_data(6);
  // The expected output data layout is major-to-minor.
  std::vector<float> expected_out_data = {0, 1, 2, 3, 4, 5};
  auto future =
      array->CopyToHostBuffer(out_data.data(), /*byte_strides=*/std::nullopt,
                              ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());
  EXPECT_THAT(out_data, ElementsAreArray(expected_out_data));
}

TEST(ArrayImplTest, MakeArrayFromHostBufferAndCopyToHostBufferWithByteStrides) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  // The input data layout is major-to-minor.
  std::vector<float> data = {0, 1, 2, 3, 4, 5};
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  std::vector<float> out_data(6);
  // The requested output data layout is minor-to-major.
  std::vector<int64_t> byte_strides = {4, 8};
  std::vector<float> expected_out_data = {0, 3, 1, 4, 2, 5};
  auto future = array->CopyToHostBuffer(out_data.data(), byte_strides,
                                        ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());
  EXPECT_THAT(out_data, ElementsAreArray(expected_out_data));
}

TEST(ArrayImplTest, MakeArrayFromHostBufferReplicated) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);
  absl::Span<Device* const> devices = client->addressable_devices();
  ShardingRef sharding = ConcreteEvenSharding::Create(
      client->MakeDeviceList(devices), MemoryKind(), shape,
      /*shard_shape=*/shape, /*is_fully_replicated=*/true);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      client->MakeArrayFromHostBuffer(
          data->data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding,
          Client::HostBufferSemantics::kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/nullptr));

  // Once the `Array` has become ready, the host buffer is not accessed.
  TF_ASSERT_OK(array->GetReadyFuture().Await());
  data = nullptr;
  // There should be no use-after-free.

  TF_ASSERT_OK_AND_ASSIGN(auto single_device_arrays,
                          array->DisassembleIntoSingleDeviceArrays(
                              ArrayCopySemantics::kAlwaysCopy,
                              SingleDeviceShardSemantics::kAddressableShards));
  ASSERT_EQ(single_device_arrays.size(), devices.size());
  for (int i = 0; i < single_device_arrays.size(); ++i) {
    EXPECT_THAT(single_device_arrays[i]->sharding().devices()->devices(),
                ElementsAre(devices[i]));

    std::vector<float> out_data(6);
    auto future = single_device_arrays[i]->CopyToHostBuffer(
        out_data.data(),
        /*byte_strides=*/std::nullopt, ArrayCopySemantics::kAlwaysCopy);
    TF_ASSERT_OK(future.Await());
    EXPECT_THAT(out_data, ElementsAre(0, 1, 2, 3, 4, 5));
  }
}

TEST(ArrayImplTest, MakeArraysFromHostBufferShardsAndCopyToHostBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  if (client->addressable_devices().size() < 2) {
    GTEST_SKIP() << "This test is relevant only for clients with devices that "
                    "have at least 2 devices";
  }

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  Shape shard_shape({1, 3});
  auto data0 = std::make_unique<std::vector<float>>(3);
  std::iota(data0->begin(), data0->end(), 0);
  auto data1 = std::make_unique<std::vector<float>>(3);
  std::iota(data1->begin(), data1->end(), 3);
  absl::Span<Device* const> devices =
      client->addressable_devices().subspan(0, 2);
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef sharding,
      ShardingParamSharding::Create(
          ShardingParam(
              /*dim_shards=*/{2, 1},
              {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 1}}),
          client->MakeDeviceList(devices), MemoryKind()));

  std::vector<Client::MakeArraysFromHostBufferShardsSpec> specs;
  // Create two arrays with the same sharding, but swapped host buffers (data0
  // and data1).
  specs.push_back({
      /*buffers=*/{
          {{0},
           {data0->data(), dtype, shard_shape, /*byte_strides=*/std::nullopt,
            /*on_done_with_host_buffer=*/nullptr}},
          {{1},
           {data1->data(), dtype, shard_shape, /*byte_strides=*/std::nullopt,
            /*on_done_with_host_buffer=*/nullptr}}},
      /*array_spec=*/{dtype, shape, sharding, /*layout=*/nullptr},
  });
  specs.push_back({
      /*buffers=*/{
          {{0},
           {data1->data(), dtype, shard_shape, /*byte_strides=*/std::nullopt,
            /*on_done_with_host_buffer=*/nullptr}},
          {{1},
           {data0->data(), dtype, shard_shape, /*byte_strides=*/std::nullopt,
            /*on_done_with_host_buffer=*/nullptr}}},
      /*array_spec=*/{dtype, shape, sharding, /*layout=*/nullptr},
  });

  TF_ASSERT_OK_AND_ASSIGN(
      auto arrays, client->MakeArraysFromHostBufferShards(
                       absl::MakeSpan(specs),
                       Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                       client->CreateUserContext()));
  ASSERT_THAT(arrays, SizeIs(2));

  // Once the `Array` has become ready, the host buffer is not accessed.
  TF_ASSERT_OK(arrays[0]->GetReadyFuture().Await());
  TF_ASSERT_OK(arrays[1]->GetReadyFuture().Await());
  data0 = nullptr;
  data1 = nullptr;
  // There should be no use-after-free.

  for (int i = 0; i < arrays.size(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto single_device_arrays,
        arrays[i]->DisassembleIntoSingleDeviceArrays(
            ArrayCopySemantics::kAlwaysCopy,
            SingleDeviceShardSemantics::kAddressableShards));
    ASSERT_EQ(single_device_arrays.size(), devices.size());
    for (int j = 0; j < single_device_arrays.size(); ++j) {
      EXPECT_THAT(single_device_arrays[j]->sharding().devices()->devices(),
                  ElementsAre(devices[j]));

      std::vector<float> out_data(3);
      auto future = single_device_arrays[j]->CopyToHostBuffer(
          out_data.data(),
          /*byte_strides=*/std::nullopt, ArrayCopySemantics::kAlwaysCopy);
      TF_ASSERT_OK(future.Await());
      if ((i + j) % 2 == 0) {
        EXPECT_THAT(out_data, ElementsAre(0, 1, 2));
      } else {
        EXPECT_THAT(out_data, ElementsAre(3, 4, 5));
      }
    }
  }
}

TEST(ArrayImplTest, MakeArraysFromHostBufferShardsWithDifferentDevices) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  if (client->addressable_devices().size() < 2) {
    GTEST_SKIP() << "This test is relevant only for clients with devices that "
                    "have at least 2 devices";
  }

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  Shape shard_shape = shape;
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);

  ShardingRef sharding0 = SingleDeviceSharding::Create(
      client->addressable_devices()[0], MemoryKind());
  ShardingRef sharding1 = SingleDeviceSharding::Create(
      client->addressable_devices()[1], MemoryKind());

  std::vector<Client::MakeArraysFromHostBufferShardsSpec> specs;
  // Create two arrays with different shardings.
  specs.push_back({
      /*buffers=*/{
          {{0},
           {data->data(), dtype, shard_shape, /*byte_strides=*/std::nullopt,
            /*on_done_with_host_buffer=*/nullptr}}},
      /*array_spec=*/{dtype, shape, sharding0, /*layout=*/nullptr},
  });
  specs.push_back({
      /*buffers=*/{
          {{0},
           {data->data(), dtype, shard_shape, /*byte_strides=*/std::nullopt,
            /*on_done_with_host_buffer=*/nullptr}}},
      /*array_spec=*/{dtype, shape, sharding1, /*layout=*/nullptr},
  });

  absl::Status status;
  auto result = client->MakeArraysFromHostBufferShards(
      absl::MakeSpan(specs),
      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
      client->CreateUserContext());
  if (result.ok()) {
    // Implementations may poison outputs instead of immediately returning an
    // error.
    status = result->at(0)->GetReadyFuture().Await();
  } else {
    status = result.status();
  }
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ArrayImplTest, MakeArraysFromHostBufferShardsWithDifferentMemoryKinds) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  if (client->addressable_devices().front()->Memories().size() < 2) {
    GTEST_SKIP() << "This test is relevant only for clients with a device that "
                    "have at least 2 memories";
  }

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  Shape shard_shape = shape;
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);

  std::vector<MemoryKind> memory_kinds;
  for (const Memory* memory :
       client->addressable_devices().front()->Memories()) {
    memory_kinds.push_back(memory->Kind());
  }

  ShardingRef sharding0 = SingleDeviceSharding::Create(
      client->addressable_devices().front(), memory_kinds[0]);
  ShardingRef sharding1 = SingleDeviceSharding::Create(
      client->addressable_devices().front(), memory_kinds[1]);

  std::vector<Client::MakeArraysFromHostBufferShardsSpec> specs;
  // Create two arrays with different shardings.
  specs.push_back({
      /*buffers=*/{
          {{0},
           {data->data(), dtype, shard_shape, /*byte_strides=*/std::nullopt,
            /*on_done_with_host_buffer=*/nullptr}}},
      /*array_spec=*/{dtype, shape, sharding0, /*layout=*/nullptr},
  });
  specs.push_back({
      /*buffers=*/{
          {{0},
           {data->data(), dtype, shard_shape, /*byte_strides=*/std::nullopt,
            /*on_done_with_host_buffer=*/nullptr}}},
      /*array_spec=*/{dtype, shape, sharding1, /*layout=*/nullptr},
  });

  absl::Status status;
  auto result = client->MakeArraysFromHostBufferShards(
      absl::MakeSpan(specs),
      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
      client->CreateUserContext());
  if (result.ok()) {
    // Implementations may poison outputs instead of immediately returning an
    // error.
    status = result->at(0)->GetReadyFuture().Await();
  } else {
    status = result.status();
  }
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ArrayImplTest, MakeErrorArrays) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  xla::ifrt::DeviceListRef device_list =
      client->MakeDeviceList(client->addressable_devices());

  Shape shape({2, 2});
  ArraySpec array_spec = {
      /*dtype=*/xla::ifrt::DType(xla::ifrt::DType::kS8),
      /*shape=*/shape,
      /*sharding=*/
      xla::ifrt::ConcreteEvenSharding::Create(
          device_list, xla::ifrt::MemoryKind(), shape, /*shard_shape=*/shape,
          /*is_fully_replicated=*/true),
  };

  const absl::Status error = absl::InternalError("injected error");
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<tsl::RCReference<xla::ifrt::Array>> arrays,
      client->MakeErrorArrays(error, {array_spec, array_spec},
                              client->CreateUserContext()));
  ASSERT_EQ(arrays.size(), 2);

  EXPECT_THAT(arrays[0]->GetReadyFuture().Await(),
              StatusIs(_, HasSubstr("injected error")));
  EXPECT_THAT(arrays[1]->GetReadyFuture().Await(),
              StatusIs(_, HasSubstr("injected error")));
}

TEST(ArrayImplTest, MakeErrorArraysWithAddressableAndNonAddressableDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  std::vector<Device*> non_addressable_devices =
      GetNonAddressableDevices(client.get());
  if (non_addressable_devices.empty()) {
    GTEST_SKIP() << "Skipping test; needs at least 1 non-addressable device.";
  }

  Shape shape({2, 2});

  std::vector<Device*> devices;
  devices.reserve(2);
  devices.push_back(client->addressable_devices().at(0));
  devices.push_back(non_addressable_devices.at(0));
  ShardingRef sharding =
      ConcreteEvenSharding::Create(client->MakeDeviceList(devices),
                                   MemoryKind(), shape, /*shard_shape=*/shape,
                                   /*is_fully_replicated=*/true);

  ArraySpec array_spec = {/*dtype=*/xla::ifrt::DType(xla::ifrt::DType::kS8),
                          /*shape=*/shape,
                          /*sharding=*/sharding};

  const absl::Status error = absl::InternalError("injected error");
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<tsl::RCReference<xla::ifrt::Array>> arrays,
      client->MakeErrorArrays(error, {array_spec, array_spec},
                              client->CreateUserContext()));
  ASSERT_EQ(arrays.size(), 2);

  EXPECT_THAT(arrays[0]->GetReadyFuture().Await(),
              StatusIs(_, HasSubstr("injected error")));
  EXPECT_THAT(arrays[1]->GetReadyFuture().Await(),
              StatusIs(_, HasSubstr("injected error")));
}

TEST(ArrayImplTest, AssembleArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device0 = client->addressable_devices().at(0);
  ShardingRef sharding0 = SingleDeviceSharding::Create(device0, MemoryKind());
  Device* device1 = client->addressable_devices().at(1);
  ShardingRef sharding1 = SingleDeviceSharding::Create(device1, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto array0, client->MakeArrayFromHostBuffer(
                       data.data(), dtype, shape,
                       /*byte_strides=*/std::nullopt, sharding0,
                       Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                       /*on_done_with_host_buffer=*/{}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto array1, client->MakeArrayFromHostBuffer(
                       data.data(), dtype, shape,
                       /*byte_strides=*/std::nullopt, sharding1,
                       Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                       /*on_done_with_host_buffer=*/{}));

  std::vector<tsl::RCReference<Array>> arrays({array0, array1});
  Shape assembled_shape({4, 3});
  ShardingRef assembled_sharding = OpaqueSharding::Create(
      client->MakeDeviceList({array0->sharding().devices()->devices().front(),
                              array1->sharding().devices()->devices().front()}),
      MemoryKind());
  TF_ASSERT_OK_AND_ASSIGN(
      auto assembled_array,
      client->AssembleArrayFromSingleDeviceArrays(
          dtype, assembled_shape, assembled_sharding, absl::MakeSpan(arrays),
          ArrayCopySemantics::kAlwaysCopy,
          SingleDeviceShardSemantics::kAddressableShards));

  EXPECT_EQ(assembled_array->dtype(), dtype);
  EXPECT_EQ(assembled_array->shape(), assembled_shape);
  EXPECT_EQ(assembled_array->shared_ptr_sharding().get(),
            assembled_sharding.get());
}

TEST(ArrayImplTest, AssembleAndDisassembleArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device0 = client->addressable_devices().at(0);
  ShardingRef sharding0 = SingleDeviceSharding::Create(device0, MemoryKind());
  Device* device1 = client->addressable_devices().at(1);
  ShardingRef sharding1 = SingleDeviceSharding::Create(device1, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto array0, client->MakeArrayFromHostBuffer(
                       data.data(), dtype, shape,
                       /*byte_strides=*/std::nullopt, sharding0,
                       Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                       /*on_done_with_host_buffer=*/{}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto array1, client->MakeArrayFromHostBuffer(
                       data.data(), dtype, shape,
                       /*byte_strides=*/std::nullopt, sharding1,
                       Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                       /*on_done_with_host_buffer=*/{}));

  std::vector<tsl::RCReference<Array>> arrays({array0, array1});
  Shape assembled_shape({4, 3});
  ShardingParam sharding_param(
      /*dim_shards=*/{2, 1}, {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 1}});
  auto ifrt_device_list =
      client->MakeDeviceList({array0->sharding().devices()->devices().front(),
                              array1->sharding().devices()->devices().front()});
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef sharding_param_sharding,
      ShardingParamSharding::Create(std::move(sharding_param), ifrt_device_list,
                                    MemoryKind()));
  ShardingRef assembled_shardings[] = {
      ConcreteEvenSharding::Create(ifrt_device_list, MemoryKind(),
                                   assembled_shape, shape),
      sharding_param_sharding};
  for (auto& assembled_sharding : assembled_shardings) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto assembled_array,
        client->AssembleArrayFromSingleDeviceArrays(
            assembled_shape, assembled_sharding, absl::MakeSpan(arrays),
            ArrayCopySemantics::kAlwaysCopy,
            SingleDeviceShardSemantics::kAddressableShards));

    TF_ASSERT_OK_AND_ASSIGN(
        auto single_device_arrays,
        assembled_array->DisassembleIntoSingleDeviceArrays(
            ArrayCopySemantics::kAlwaysCopy,
            SingleDeviceShardSemantics::kAddressableShards));

    ASSERT_THAT(single_device_arrays, SizeIs(2));
    EXPECT_EQ(single_device_arrays[0]->dtype(), array0->dtype());
    EXPECT_EQ(single_device_arrays[0]->shape(), array0->shape());
    EXPECT_THAT(single_device_arrays[0]->sharding().devices()->devices(),
                ElementsAreArray(array0->sharding().devices()->devices()));
    EXPECT_EQ(single_device_arrays[1]->dtype(), array1->dtype());
    EXPECT_EQ(single_device_arrays[1]->shape(), array1->shape());
    EXPECT_THAT(single_device_arrays[1]->sharding().devices()->devices(),
                ElementsAreArray(array1->sharding().devices()->devices()));
  }
}

TEST(ArrayImplTest, AssembleAndDisassembleSingleDeviceArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  std::vector<tsl::RCReference<Array>> arrays({array});

  TF_ASSERT_OK_AND_ASSIGN(auto assembled_array,
                          client->AssembleArrayFromSingleDeviceArrays(
                              dtype, shape, sharding, absl::MakeSpan(arrays),
                              ArrayCopySemantics::kAlwaysCopy,
                              SingleDeviceShardSemantics::kAddressableShards));

  ASSERT_EQ(assembled_array->dtype(), array->dtype());
  ASSERT_EQ(assembled_array->shape(), array->shape());
  ASSERT_THAT(assembled_array->sharding().devices()->devices(),
              ElementsAreArray(array->sharding().devices()->devices()));

  TF_ASSERT_OK_AND_ASSIGN(auto single_device_arrays,
                          assembled_array->DisassembleIntoSingleDeviceArrays(
                              ArrayCopySemantics::kAlwaysCopy,
                              SingleDeviceShardSemantics::kAddressableShards));

  ASSERT_THAT(single_device_arrays, SizeIs(1));
  ASSERT_EQ(single_device_arrays[0]->dtype(), array->dtype());
  ASSERT_EQ(single_device_arrays[0]->shape(), array->shape());
  EXPECT_THAT(single_device_arrays[0]->sharding().devices()->devices(),
              ElementsAreArray(array->sharding().devices()->devices()));
}

TEST(ArrayImplTest, CopyToSameDevices) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      /*on_done_with_host_buffer=*/{}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto new_arrays,
      client->CopyArrays(absl::MakeSpan(&array, 1), sharding->devices(),
                         MemoryKind(), ArrayCopySemantics::kAlwaysCopy));

  std::vector<float> out_data(6);
  auto future = new_arrays[0]->CopyToHostBuffer(
      out_data.data(),
      /*byte_strides=*/std::nullopt, ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());
  EXPECT_THAT(out_data, ElementsAreArray(data));
}

TEST(ArrayImplTest, AssembleAndDisassembleNonAddressableArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  std::vector<Device*> non_addressable_devices =
      GetNonAddressableDevices(client.get());
  if (non_addressable_devices.size() < 2) {
    GTEST_SKIP() << "Skipping test; needs at least 2 non-addressable devices.";
  }

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device0 = client->addressable_devices().at(0);
  ShardingRef sharding0 = SingleDeviceSharding::Create(device0, MemoryKind());
  Device* device1 = client->addressable_devices().at(1);
  ShardingRef sharding1 = SingleDeviceSharding::Create(device1, MemoryKind());

  std::vector<tsl::RCReference<Array>> arrays;
  Shape assembled_shape({4, 3});
  ShardingParam sharding_param(
      /*dim_shards=*/{2, 1}, {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 1}});

  absl::flat_hash_set<DeviceId> addressable_device_ids;
  for (auto* device : client->addressable_devices()) {
    addressable_device_ids.insert(device->Id());
  }
  auto ifrt_device_list = client->MakeDeviceList(
      absl::MakeConstSpan(non_addressable_devices).subspan(0, 2));
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef sharding_param_sharding,
      ShardingParamSharding::Create(std::move(sharding_param), ifrt_device_list,
                                    MemoryKind()));
  ShardingRef assembled_shardings[] = {
      ConcreteEvenSharding::Create(ifrt_device_list, MemoryKind(),
                                   assembled_shape, shape),
      sharding_param_sharding};
  for (auto& assembled_sharding : assembled_shardings) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto assembled_array,
        client->AssembleArrayFromSingleDeviceArrays(
            dtype, assembled_shape, assembled_sharding, absl::MakeSpan(arrays),
            ArrayCopySemantics::kAlwaysCopy,
            SingleDeviceShardSemantics::kAddressableShards));

    TF_ASSERT_OK_AND_ASSIGN(
        auto single_device_arrays,
        assembled_array->DisassembleIntoSingleDeviceArrays(
            ArrayCopySemantics::kAlwaysCopy,
            SingleDeviceShardSemantics::kAddressableShards));

    ASSERT_THAT(single_device_arrays, SizeIs(0));
  }
}

TEST(ArrayImplTest, CopyToDifferentDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  DeviceListRef devices = client->MakeDeviceList(client->addressable_devices());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;
  std::vector<tsl::RCReference<Array>> shards;
  for (auto* device : devices->devices()) {
    ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
    TF_ASSERT_OK_AND_ASSIGN(shards.emplace_back(),
                            client->MakeArrayFromHostBuffer(
                                data.data(), dtype, shape,
                                /*byte_strides=*/std::nullopt, sharding,
                                semantics, /*on_done_with_host_buffer=*/{}));
  }

  // Intentionally use different shardings to verify that each result array has
  // the correct sharding.
  std::vector<tsl::RCReference<Array>> arrays;
  {
    std::vector<Shape> shapes(shards.size(), shape);
    ShardingRef sharding =
        ConcreteSharding::Create(devices, MemoryKind(), shape, shapes);
    TF_ASSERT_OK_AND_ASSIGN(
        arrays.emplace_back(),
        client->AssembleArrayFromSingleDeviceArrays(
            dtype, shape, sharding, absl::MakeSpan(shards),
            ArrayCopySemantics::kAlwaysCopy,
            SingleDeviceShardSemantics::kAddressableShards));
  }
  {
    ShardingRef sharding =
        ConcreteEvenSharding::Create(devices, MemoryKind(), shape, shape);
    TF_ASSERT_OK_AND_ASSIGN(
        arrays.emplace_back(),
        client->AssembleArrayFromSingleDeviceArrays(
            dtype, shape, sharding, absl::MakeSpan(shards),
            ArrayCopySemantics::kAlwaysCopy,
            SingleDeviceShardSemantics::kAddressableShards));
  }

  absl::InlinedVector<xla::ifrt::Device*, 1> new_devices;
  for (auto it = devices->devices().rbegin(); it != devices->devices().rend();
       ++it) {
    new_devices.push_back(*it);
  }
  TF_ASSERT_OK_AND_ASSIGN(
      auto new_arrays,
      client->CopyArrays(absl::MakeSpan(arrays),
                         client->MakeDeviceList(new_devices), MemoryKind(),
                         ArrayCopySemantics::kAlwaysCopy));

  for (int i = 0; i < arrays.size(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto expected_sharding,
        arrays[i]->sharding().WithDeviceAssignment(
            client->MakeDeviceList(new_devices), MemoryKind()));
    EXPECT_EQ(new_arrays[i]->sharding(), *expected_sharding);

    TF_ASSERT_OK_AND_ASSIGN(
        auto shards, arrays[i]->DisassembleIntoSingleDeviceArrays(
                         ArrayCopySemantics::kAlwaysCopy,
                         SingleDeviceShardSemantics::kAddressableShards));
    for (const auto& shard : shards) {
      std::vector<float> out_data(6);
      auto future = shard->CopyToHostBuffer(out_data.data(),
                                            /*byte_strides=*/std::nullopt,
                                            ArrayCopySemantics::kAlwaysCopy);
      TF_ASSERT_OK(future.Await());
      EXPECT_THAT(out_data, ElementsAreArray(data));
    }
  }
}

TEST(ArrayImplTest, CopyMixedSourceDevices) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  std::vector<tsl::RCReference<Array>> arrays;
  for (auto* device : client->addressable_devices()) {
    ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
    TF_ASSERT_OK_AND_ASSIGN(
        arrays.emplace_back(),
        client->MakeArrayFromHostBuffer(data.data(), dtype, shape,
                                        /*byte_strides=*/std::nullopt, sharding,
                                        semantics,
                                        /*on_done_with_host_buffer=*/{}));
  }

  Device* new_device = client->addressable_devices().at(1);
  EXPECT_THAT(client
                  ->CopyArrays(absl::MakeSpan(arrays),
                               client->MakeDeviceList({new_device}),
                               MemoryKind(), ArrayCopySemantics::kAlwaysCopy)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ArrayImplTest, CopyMixedSourceMemoryKind) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  if (client->addressable_devices()[0]->Memories().size() <= 1) {
    GTEST_SKIP() << "This test is relevant only for clients with devices that "
                    "have more than one memory kind";
  }

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  std::vector<tsl::RCReference<Array>> arrays;
  for (auto* memory : device->Memories()) {
    ShardingRef sharding = SingleDeviceSharding::Create(device, memory->Kind());
    TF_ASSERT_OK_AND_ASSIGN(arrays.emplace_back(),
                            client->MakeArrayFromHostBuffer(
                                data.data(), dtype, shape,
                                /*byte_strides=*/std::nullopt, sharding,
                                semantics, /*on_done_with_host_buffer=*/{}));
  }

  Device* new_device = client->addressable_devices().at(1);
  EXPECT_THAT(client
                  ->CopyArrays(absl::MakeSpan(arrays),
                               client->MakeDeviceList({new_device}),
                               MemoryKind(), ArrayCopySemantics::kAlwaysCopy)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ArrayImplTest, GetReadyFuture) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      /*on_done_with_host_buffer=*/{}));
  TF_EXPECT_OK(array->GetReadyFuture().Await());
}

TEST(ArrayImplTest, BatchedGetReadyFuture) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  std::vector<tsl::RCReference<Value>> values;
  for (int i = 0; i < 4; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(values.emplace_back(),
                            client->MakeArrayFromHostBuffer(
                                data.data(), dtype, shape,
                                /*byte_strides=*/std::nullopt, sharding,
                                semantics, /*on_done_with_host_buffer=*/{}));
  }
  TF_EXPECT_OK(client->GetReadyFuture(values).Await());
}

TEST(ArrayImplTest, Delete) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      /*on_done_with_host_buffer=*/{}));
  TF_EXPECT_OK(array->Delete().Await());
}

TEST(ArrayImplTest, DeleteIsIdempotent) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      /*on_done_with_host_buffer=*/{}));

  auto future_1 = array->Delete();
  auto future_2 = array->Delete();

  TF_EXPECT_OK(future_1.Await());
  TF_EXPECT_OK(future_2.Await());
}

TEST(ArrayImplTest, IsDeleted) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  std::iota(data.begin(), data.end(), 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      /*on_done_with_host_buffer=*/{}));
  EXPECT_FALSE(array->IsDeleted());
  auto future = array->Delete();
  EXPECT_TRUE(array->IsDeleted());
  TF_EXPECT_OK(future.Await());
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
