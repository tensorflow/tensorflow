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
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::SizeIs;

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

// Returns all addressable CPU devices in the client.
std::vector<Device*> GetAddressableCpuDevices(Client* client) {
  std::vector<Device*> cpu_devices;
  for (const auto& device : client->GetAllDevices()) {
    if (device->IsAddressable() && device->Kind() == "cpu") {
      cpu_devices.push_back(device);
    }
  }
  return cpu_devices;
}

TEST(ArrayImplTest, MakeArrayFromHostBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  absl::c_iota(*data, 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
  UserContextScope user_context_scope(test_util::MakeUserContext(100));

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data->data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/nullptr));

  EXPECT_EQ(array->dtype(), dtype);
  EXPECT_EQ(array->shape(), shape);
  EXPECT_EQ(array->shared_ptr_sharding().get(), sharding.get());
  EXPECT_EQ(array->user_context()->Id(), UserContextId(100));
}

TEST(ArrayImplTest, MakeArrayFromHostBufferWithCustomLayout) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  absl::c_iota(*data, 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
  TF_ASSERT_OK_AND_ASSIGN(CustomLayoutRef layout,
                          client->GetDefaultLayout(dtype, shape, sharding));
  UserContextScope user_context_scope(test_util::MakeUserContext(100));

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data->data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, layout,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/nullptr));

  EXPECT_EQ(array->dtype(), dtype);
  EXPECT_EQ(array->shape(), shape);
  EXPECT_EQ(array->shared_ptr_sharding().get(), sharding.get());
  // The created array currently only needs to have an equivalent layout to the
  // requested layout.
  // TODO(hyeontaek): Tighten this check to verify `array->layout().get() ==
  // layout.get()` to make it consistent with array sharding handling.
  EXPECT_THAT(EquivalentLayouts(dtype, shape, sharding, array->layout(), dtype,
                                shape, sharding, layout),
              IsOkAndHolds(true));
  EXPECT_EQ(array->user_context()->Id(), UserContextId(100));
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
  absl::c_iota(*data, 0);

  std::vector<Device*> devices;
  devices.reserve(2);
  devices.push_back(non_addressable_devices.at(0));
  devices.push_back(client->addressable_devices().at(0));
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client->MakeDeviceList(devices));
  ShardingRef sharding =
      ConcreteEvenSharding::Create(std::move(device_list), MemoryKind(), shape,
                                   /*shard_shape=*/shape,
                                   /*is_fully_replicated=*/true);
  UserContextScope user_context_scope(test_util::MakeUserContext(100));

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data->data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/nullptr));

  EXPECT_EQ(array->dtype(), dtype);
  EXPECT_EQ(array->shape(), shape);
  EXPECT_EQ(array->shared_ptr_sharding().get(), sharding.get());
  EXPECT_EQ(array->user_context()->Id(), UserContextId(100));
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
  absl::c_iota(*data, 0);
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
  absl::c_iota(*data, 0);
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
  absl::c_iota(*data, 0);
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
  absl::c_iota(*data, 0);
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

TEST(ArrayImplTest, MakeArrayFromHostBufferDefaultLayout) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
  Device* device = client->addressable_devices()[0];

  for (Memory* const memory : device->Memories()) {
    SCOPED_TRACE(absl::StrCat(memory->Kind()));

    TF_ASSERT_OK_AND_ASSIGN(
        auto array,
        client->MakeArrayFromHostBuffer(
            data.data(), dtype, shape, /*byte_strides=*/std::nullopt,
            SingleDeviceSharding::Create(device, memory->Kind()),
            Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr));
    TF_ASSERT_OK(array->GetReadyFuture().Await());

    TF_ASSERT_OK_AND_ASSIGN(auto layout, array->pjrt_layout());
    // `layout` should be either nullptr or a concrete default layout.
    if (layout != nullptr) {
      TF_ASSERT_OK_AND_ASSIGN(auto default_layout,
                              client->GetDefaultPjRtLayout(
                                  dtype, shape.dims(), device, memory->Kind()));

      EXPECT_EQ(*layout, *default_layout);
    }
  }
}

TEST(ArrayImplTest, MakeArrayFromHostBufferAndCopyToHostBuffer) {
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

TEST(ArrayImplTest, MakeArrayFromHostBufferWithNonCompactByteStrides) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef device_list,
      client->MakeDeviceList(client->addressable_devices()));
  ASSERT_GT(device_list->size(), 1);

  DType dtype(DType::kS8);
  Shape shape({2, 2});
  std::vector<int8_t> data = {0, -1, 1, -1, 2, -1, 3, -1};
  std::vector<int64_t> byte_strides = {2, 4};
  ShardingRef sharding = ConcreteEvenSharding::Create(
      device_list, MemoryKind(), shape, /*shard_shape=*/shape,
      /*is_fully_replicated=*/true);

  TF_ASSERT_OK_AND_ASSIGN(
      ArrayRef array, client->MakeArrayFromHostBuffer(
                          data.data(), dtype, shape, byte_strides, sharding,
                          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                          /*on_done_with_host_buffer=*/nullptr));
  TF_ASSERT_OK(array->GetReadyFuture().Await());

  std::vector<int8_t> out_data(4);
  tsl::Future<> future =
      array->CopyToHostBuffer(out_data.data(), /*byte_strides=*/std::nullopt,
                              ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());
  EXPECT_THAT(out_data, ElementsAre(0, 2, 1, 3));
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
  absl::c_iota(*data, 0);
  absl::Span<Device* const> devices = client->addressable_devices();
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client->MakeDeviceList(devices));
  ShardingRef sharding = ConcreteEvenSharding::Create(
      std::move(device_list), MemoryKind(), shape,
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
                              ArrayCopySemantics::kReuseInput,
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
  absl::c_iota(*data0, 0);
  auto data1 = std::make_unique<std::vector<float>>(3);
  absl::c_iota(*data1, 3);
  absl::Span<Device* const> devices =
      client->addressable_devices().subspan(0, 2);
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client->MakeDeviceList(devices));
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef sharding,
      ShardingParamSharding::Create(
          ShardingParam(
              /*dim_shards=*/{2, 1},
              {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 1}}),
          std::move(device_list), MemoryKind()));

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

  UserContextScope user_context_scope(test_util::MakeUserContext(100));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arrays, client->MakeArraysFromHostBufferShards(
                       absl::MakeSpan(specs),
                       Client::HostBufferSemantics::kImmutableOnlyDuringCall));
  ASSERT_THAT(arrays, SizeIs(2));

  // Once the `Array` has become ready, the host buffer is not accessed.
  TF_ASSERT_OK(arrays[0]->GetReadyFuture().Await());
  TF_ASSERT_OK(arrays[1]->GetReadyFuture().Await());
  data0 = nullptr;
  data1 = nullptr;
  // There should be no use-after-free.

  for (int i = 0; i < arrays.size(); ++i) {
    EXPECT_EQ(arrays[i]->user_context()->Id(), UserContextId(100));
    TF_ASSERT_OK_AND_ASSIGN(
        auto single_device_arrays,
        arrays[i]->DisassembleIntoSingleDeviceArrays(
            ArrayCopySemantics::kReuseInput,
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
  absl::c_iota(*data, 0);

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
  UserContextScope user_context_scope(test_util::MakeUserContext(100));
  auto result = client->MakeArraysFromHostBufferShards(
      absl::MakeSpan(specs),
      Client::HostBufferSemantics::kImmutableOnlyDuringCall);
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
  absl::c_iota(*data, 0);

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
  UserContextScope user_context_scope(test_util::MakeUserContext(100));
  auto result = client->MakeArraysFromHostBufferShards(
      absl::MakeSpan(specs),
      Client::HostBufferSemantics::kImmutableOnlyDuringCall);
  if (result.ok()) {
    // Implementations may poison outputs instead of immediately returning an
    // error.
    status = result->at(0)->GetReadyFuture().Await();
  } else {
    status = result.status();
  }
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ArrayImplTest, MakeArraysFromHostBufferShardsWithLayout) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
  Device* device = client->addressable_devices()[0];

  auto layout = std::make_shared<xla::PjRtLayout>(
      xla::LayoutUtil::MakeDescendingLayout(shape.dims().size()));

  ArrayRef array;
  {
    Client::HostBuffer host_buffer = {
        /*data=*/data.data(),
        /*dtype=*/dtype,
        /*shape=*/shape,
    };
    Client::MakeArraysFromHostBufferShardsSpec spec = {
        /*buffers=*/{{{0}, host_buffer}},
        /*array_spec=*/
        {
            /*dtype=*/dtype,
            /*shape=*/shape,
            /*sharding=*/
            SingleDeviceSharding::Create(device, MemoryKind()),
            /*layout=*/layout,
        },
    };
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<ArrayRef> arrays,
        client->MakeArraysFromHostBufferShards(
            absl::MakeSpan(&spec, 1),
            Client::HostBufferSemantics::kImmutableOnlyDuringCall));
    array = std::move(arrays.front());
  }

  TF_ASSERT_OK(array->GetReadyFuture().Await());
  TF_ASSERT_OK_AND_ASSIGN(auto result_layout, array->pjrt_layout());
  ASSERT_NE(result_layout, nullptr);
  EXPECT_EQ(*result_layout, *layout);
}

TEST(ArrayImplTest, MakeArrayFromHostBufferAndCopyToHostBufferWithString) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto cpu_devices = GetAddressableCpuDevices(client.get());
  if (cpu_devices.empty()) {
    GTEST_SKIP()
        << "This test is relevant only for clients with at least 1 CPU device";
  }

  DType dtype(DType::kString);
  Shape shape({2, 3});
  auto cords = std::make_shared<std::vector<absl::Cord>>();
  cords->reserve(shape.num_elements());
  for (int64_t k = 0; k < shape.num_elements(); ++k) {
    cords->push_back(absl::Cord(absl::StrCat("string-", k)));
  }
  void* data_ptr = static_cast<void*>(cords->data());
  Device* device = cpu_devices.front();
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
  UserContextScope user_context_scope(test_util::MakeUserContext(100));

  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      client->MakeArrayFromHostBuffer(
          data_ptr, dtype, shape,
          /*byte_strides=*/std::nullopt, std::move(sharding),
          Client::HostBufferSemantics::kImmutableUntilTransferCompletes,
          /*on_done_with_host_buffer=*/[cords = std::move(cords)]() {}));
  EXPECT_EQ(array->user_context()->Id(), UserContextId(100));

  std::vector<absl::Cord> out_data(shape.num_elements());
  auto future =
      array->CopyToHostBuffer(out_data.data(), /*byte_strides=*/std::nullopt,
                              ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());
  for (int k = 0; k < shape.num_elements(); ++k) {
    EXPECT_EQ(out_data[k].Flatten(), absl::StrCat("string-", k))
        << "Unexpected data at element " << k;
  }
}

TEST(ArrayImplTest,
     MakeArraysFromHostBufferShardsAndCopyToHostBufferWithString) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto cpu_devices = GetAddressableCpuDevices(client.get());
  if (cpu_devices.size() < 2) {
    GTEST_SKIP()
        << "This test is relevant only for clients with at least 2 CPU devices";
  }

  DType dtype(DType::kString);
  Shape shape({2, 3});
  Shape shard_shape({1, 3});

  auto cords0 = std::make_shared<std::vector<absl::Cord>>();
  cords0->reserve(shard_shape.num_elements());
  for (int64_t k = 0; k < shard_shape.num_elements(); ++k) {
    cords0->push_back(absl::Cord(absl::StrCat("string-", k)));
  }
  void* data_ptr0 = static_cast<void*>(cords0->data());

  auto cords1 = std::make_shared<std::vector<absl::Cord>>();
  cords1->reserve(shard_shape.num_elements());
  for (int64_t k = 0; k < shard_shape.num_elements(); ++k) {
    cords1->push_back(absl::Cord(absl::StrCat("string-", k + 100)));
  }
  void* data_ptr1 = static_cast<void*>(cords1->data());

  absl::Span<Device* const> devices =
      absl::MakeConstSpan(cpu_devices).subspan(0, 2);
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client->MakeDeviceList(devices));
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef sharding,
      ShardingParamSharding::Create(
          ShardingParam(
              /*dim_shards=*/{2, 1},
              {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 1}}),
          std::move(device_list), MemoryKind()));

  std::vector<Client::MakeArraysFromHostBufferShardsSpec> specs;
  // Create two arrays with the same sharding, but swapped host buffers (data0
  // and data1).
  specs.push_back({
      /*buffers=*/{
          {{0},
           {data_ptr0, dtype, shard_shape, /*byte_strides=*/std::nullopt,
            /*on_done_with_host_buffer=*/[cords0]() {}}},
          {{1},
           {data_ptr1, dtype, shard_shape, /*byte_strides=*/std::nullopt,
            /*on_done_with_host_buffer=*/[cords1]() {}}}},
      /*array_spec=*/{dtype, shape, sharding, /*layout=*/nullptr},
  });
  specs.push_back({
      /*buffers=*/{
          {{0},
           {data_ptr1, dtype, shard_shape, /*byte_strides=*/std::nullopt,
            /*on_done_with_host_buffer=*/[cords1]() {}}},
          {{1},
           {data_ptr0, dtype, shard_shape, /*byte_strides=*/std::nullopt,
            /*on_done_with_host_buffer=*/[cords0]() {}}}},
      /*array_spec=*/{dtype, shape, sharding, /*layout=*/nullptr},
  });

  UserContextScope user_context_scope(test_util::MakeUserContext(100));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arrays,
      client->MakeArraysFromHostBufferShards(
          absl::MakeSpan(specs),
          Client::HostBufferSemantics::kImmutableUntilTransferCompletes));
  ASSERT_THAT(arrays, SizeIs(2));

  // Resetting these references does not necessarily destroy host buffers
  // immediately. The host buffer will be destroyed once the transfer also
  // finishes and `on_done_with_host_buffer` is destroyed.
  //
  // There is no need to reset references, but doing so in this test may shorten
  // the lifetime of host buffers and help detect an incorrect API
  // implementation as a form of use-after-free if the implementation destroyed
  // `on_done_with_host_buffer` prematurely.
  cords0 = nullptr;
  cords1 = nullptr;

  for (int i = 0; i < arrays.size(); ++i) {
    EXPECT_EQ(arrays[i]->user_context()->Id(), UserContextId(100));
    TF_ASSERT_OK_AND_ASSIGN(
        auto single_device_arrays,
        arrays[i]->DisassembleIntoSingleDeviceArrays(
            ArrayCopySemantics::kReuseInput,
            SingleDeviceShardSemantics::kAddressableShards));
    ASSERT_EQ(single_device_arrays.size(), devices.size());
    for (int j = 0; j < single_device_arrays.size(); ++j) {
      EXPECT_THAT(single_device_arrays[j]->sharding().devices()->devices(),
                  ElementsAre(devices[j]))
          << "Unexpected array sharding devices for array " << i << " shard "
          << j;

      std::vector<absl::Cord> out_data(shard_shape.num_elements());
      auto future = single_device_arrays[j]->CopyToHostBuffer(
          out_data.data(),
          /*byte_strides=*/std::nullopt, ArrayCopySemantics::kAlwaysCopy);
      TF_ASSERT_OK(future.Await());
      if ((i + j) % 2 == 0) {
        for (int k = 0; k < shard_shape.num_elements(); ++k) {
          EXPECT_EQ(out_data[k].Flatten(), absl::StrCat("string-", k))
              << "Unexpected data at array " << i << " shard " << j
              << " element " << k;
        }
      } else {
        for (int k = 0; k < shard_shape.num_elements(); ++k) {
          EXPECT_EQ(out_data[k].Flatten(), absl::StrCat("string-", k + 100))
              << "Unexpected data at array " << i << " shard " << j
              << " element " << k;
        }
      }
    }
  }
}

TEST(ArrayImplTest, HostBufferRoundTripAllMemoryKinds) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
  Device* device = client->addressable_devices()[0];

  for (Memory* const memory : device->Memories()) {
    SCOPED_TRACE(absl::StrCat(memory->Kind()));

    ShardingRef sharding = SingleDeviceSharding::Create(device, memory->Kind());
    TF_ASSERT_OK_AND_ASSIGN(
        auto array,
        client->MakeArrayFromHostBuffer(
            data.data(), dtype, shape, /*byte_strides=*/std::nullopt, sharding,
            Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr));

    EXPECT_EQ(array->dtype(), dtype);
    EXPECT_EQ(array->shape(), shape);
    EXPECT_EQ(array->sharding(), *sharding);
    TF_ASSERT_OK(array->GetReadyFuture().Await());

    std::vector<float> new_data(6);
    tsl::Future<> future = array->CopyToHostBuffer(
        static_cast<void*>(new_data.data()), /*byte_strides=*/std::nullopt,
        ArrayCopySemantics::kAlwaysCopy);
    TF_ASSERT_OK(future.Await());
    EXPECT_THAT(new_data, ElementsAreArray(data));
  }
}

TEST(ArrayImplTest, HostBufferInt4) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef device_list,
      client->MakeDeviceList(client->addressable_devices()));
  ASSERT_GT(device_list->size(), 1);

  DType dtype(DType::kS4);
  Shape shape({2, 2});
  std::vector<int8_t> data = {1, 2, 3, 4};

  for (Memory* const memory : device_list->devices().front()->Memories()) {
    SCOPED_TRACE(absl::StrCat(memory->Kind()));

    ShardingRef sharding = ConcreteEvenSharding::Create(
        device_list, memory->Kind(), shape,
        /*shard_shape=*/shape, /*is_fully_replicated=*/true);
    TF_ASSERT_OK_AND_ASSIGN(
        ArrayRef array,
        client->MakeArrayFromHostBuffer(
            data.data(), dtype, shape,
            /*byte_strides=*/std::nullopt, sharding,
            Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr));
    TF_ASSERT_OK(array->GetReadyFuture().Await());

    std::vector<int8_t> out_data(4);
    tsl::Future<> future =
        array->CopyToHostBuffer(out_data.data(), /*byte_strides=*/std::nullopt,
                                ArrayCopySemantics::kAlwaysCopy);
    TF_ASSERT_OK(future.Await());
    EXPECT_THAT(out_data, ElementsAreArray(data));
  }
}

TEST(ArrayImplTest, MakeErrorArrays) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef device_list,
      client->MakeDeviceList(client->addressable_devices()));

  Shape shape({2, 2});
  ArraySpec array_spec = {
      /*dtype=*/DType(DType::kS8),
      /*shape=*/shape,
      /*sharding=*/
      ConcreteEvenSharding::Create(device_list, MemoryKind(), shape,
                                   /*shard_shape=*/shape,
                                   /*is_fully_replicated=*/true),
  };

  const absl::Status error = absl::InternalError("injected error");
  UserContextScope user_context_scope(test_util::MakeUserContext(100));
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<ArrayRef> arrays,
      client->MakeErrorArrays(error, {array_spec, array_spec}));
  ASSERT_EQ(arrays.size(), 2);

  EXPECT_THAT(arrays[0]->GetReadyFuture().Await(),
              StatusIs(_, HasSubstr("injected error")));
  EXPECT_THAT(arrays[1]->GetReadyFuture().Await(),
              StatusIs(_, HasSubstr("injected error")));
  EXPECT_EQ(arrays[0]->user_context()->Id(), UserContextId(100));
  EXPECT_EQ(arrays[1]->user_context()->Id(), UserContextId(100));
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
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client->MakeDeviceList(devices));
  ShardingRef sharding = ConcreteEvenSharding::Create(
      std::move(device_list), MemoryKind(), shape, /*shard_shape=*/shape,
      /*is_fully_replicated=*/true);

  ArraySpec array_spec = {/*dtype=*/DType(DType::kS8),
                          /*shape=*/shape,
                          /*sharding=*/sharding};

  const absl::Status error = absl::InternalError("injected error");
  UserContextScope user_context_scope(test_util::MakeUserContext(100));
  TF_ASSERT_OK_AND_ASSIGN(
      const std::vector<ArrayRef> arrays,
      client->MakeErrorArrays(error, {array_spec, array_spec}));
  ASSERT_EQ(arrays.size(), 2);

  EXPECT_THAT(arrays[0]->GetReadyFuture().Await(),
              StatusIs(_, HasSubstr("injected error")));
  EXPECT_THAT(arrays[1]->GetReadyFuture().Await(),
              StatusIs(_, HasSubstr("injected error")));
  EXPECT_EQ(arrays[0]->user_context()->Id(), UserContextId(100));
  EXPECT_EQ(arrays[1]->user_context()->Id(), UserContextId(100));
}

TEST(ArrayImplTest, AssembleArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
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

  std::vector<ArrayRef> arrays({array0, array1});
  Shape assembled_shape({4, 3});
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef device_list,
      client->MakeDeviceList(
          {array0->sharding().devices()->devices().front(),
           array1->sharding().devices()->devices().front()}));
  ShardingRef assembled_sharding =
      OpaqueSharding::Create(std::move(device_list), MemoryKind());
  UserContextScope user_context_scope(test_util::MakeUserContext(100));
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
  EXPECT_EQ(assembled_array->user_context()->Id(), UserContextId(100));
}

TEST(ArrayImplTest, AssembleAndDisassembleArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
  Device* device0 = client->addressable_devices().at(0);
  ShardingRef sharding0 = SingleDeviceSharding::Create(device0, MemoryKind());
  Device* device1 = client->addressable_devices().at(1);
  ShardingRef sharding1 = SingleDeviceSharding::Create(device1, MemoryKind());

  // TODO(b/318709106): Make this broad `UserContextScope` to more specific to
  // assembly/diassembly calls once IFRT implementations stop reusing the input
  // single-device `Array` instance as-is when assembling/disassembling it.
  UserContextScope user_context_scope(test_util::MakeUserContext(100));

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

  std::vector<ArrayRef> arrays({array0, array1});
  Shape assembled_shape({4, 3});
  ShardingParam sharding_param(
      /*dim_shards=*/{2, 1}, {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 1}});
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef ifrt_device_list,
      client->MakeDeviceList(
          {array0->sharding().devices()->devices().front(),
           array1->sharding().devices()->devices().front()}));
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
            ArrayCopySemantics::kReuseInput,
            SingleDeviceShardSemantics::kAddressableShards));

    ASSERT_THAT(single_device_arrays, SizeIs(2));
    EXPECT_EQ(single_device_arrays[0]->dtype(), array0->dtype());
    EXPECT_EQ(single_device_arrays[0]->shape(), array0->shape());
    EXPECT_THAT(single_device_arrays[0]->sharding().devices()->devices(),
                ElementsAreArray(array0->sharding().devices()->devices()));
    EXPECT_EQ(single_device_arrays[0]->user_context()->Id(),
              UserContextId(100));
    EXPECT_EQ(single_device_arrays[1]->dtype(), array1->dtype());
    EXPECT_EQ(single_device_arrays[1]->shape(), array1->shape());
    EXPECT_THAT(single_device_arrays[1]->sharding().devices()->devices(),
                ElementsAreArray(array1->sharding().devices()->devices()));
    EXPECT_EQ(single_device_arrays[1]->user_context()->Id(),
              UserContextId(100));
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

  // TODO(b/318709106): Make this broad `UserContextScope` to more specific to
  // assembly/diassembly calls once IFRT implementations stop reusing the input
  // single-device `Array` instance as-is when assembling/disassembling it.
  UserContextScope user_context_scope(test_util::MakeUserContext(100));

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  std::vector<ArrayRef> arrays({array});

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
                              ArrayCopySemantics::kReuseInput,
                              SingleDeviceShardSemantics::kAddressableShards));

  ASSERT_THAT(single_device_arrays, SizeIs(1));
  ASSERT_EQ(single_device_arrays[0]->dtype(), array->dtype());
  ASSERT_EQ(single_device_arrays[0]->shape(), array->shape());
  EXPECT_THAT(single_device_arrays[0]->sharding().devices()->devices(),
              ElementsAreArray(array->sharding().devices()->devices()));
  EXPECT_EQ(single_device_arrays[0]->user_context()->Id(), UserContextId(100));
}

TEST(ArrayImplTest, CopyToSameDevices) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;
  UserContextScope user_context_scope(test_util::MakeUserContext(100));

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding, semantics,
                      /*on_done_with_host_buffer=*/{}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto new_arrays,
      client->CopyArrays(absl::MakeSpan(&array, 1), sharding->devices(),
                         MemoryKind(), ArrayCopySemantics::kAlwaysCopy));
  ASSERT_THAT(new_arrays, SizeIs(1));
  EXPECT_EQ(new_arrays[0]->user_context()->Id(), UserContextId(100));

  std::vector<float> out_data(6);
  auto future = new_arrays[0]->CopyToHostBuffer(
      out_data.data(),
      /*byte_strides=*/std::nullopt, ArrayCopySemantics::kAlwaysCopy);
  TF_ASSERT_OK(future.Await());
  EXPECT_THAT(out_data, ElementsAreArray(data));
}

TEST(ArrayImplTest, BitcastArrays) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({});
  Shape shard_shape({});
  std::vector<float> data(1);
  absl::c_iota(data, 3);
  absl::Span<Device* const> devices = client->addressable_devices();
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client->MakeDeviceList(devices));
  ShardingRef sharding = ConcreteEvenSharding::Create(
      device_list, MemoryKind(), shape,
      /*shard_shape=*/shape, /*is_fully_replicated=*/true);

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->MakeArrayFromHostBuffer(
                      data.data(), dtype, shape,
                      /*byte_strides=*/std::nullopt, sharding,
                      /*layout=*/nullptr,
                      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                      /*on_done_with_host_buffer=*/{}));

  // Bitcast a fully-replicated F32 [] array into a fully-partitioned F32 [1]
  // array.
  Shape new_shape({2});
  Shape new_shard_shape({1});
  ShardingRef new_sharding = ConcreteEvenSharding::Create(
      device_list, MemoryKind(), new_shape, new_shard_shape,
      /*is_fully_replicated=*/false);
  ArraySpec new_array_spec = {dtype, new_shape, new_sharding};

  auto new_arrays =
      client->BitcastArrays(absl::MakeSpan(&array, 1), {new_array_spec},
                            ArrayCopySemantics::kReuseInput);
  if (absl::IsUnimplemented(new_arrays.status())) {
    // This specific bitcast operation may not be supported by the runtime.
    GTEST_SKIP() << "Skipping test; bitcast is unimplemented.";
  }
  TF_ASSERT_OK(new_arrays.status());

  // If the bitcast was successful, the output array should have the new spec.
  ASSERT_THAT(*new_arrays, SizeIs(1));

  ASSERT_EQ(new_arrays->front()->dtype(), new_array_spec.dtype);
  ASSERT_EQ(new_arrays->front()->shape(), new_array_spec.shape);
  ASSERT_EQ(new_arrays->front()->shared_ptr_sharding(),
            new_array_spec.sharding);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const PjRtLayout> actual_pjrt_layout,
                          new_arrays->front()->pjrt_layout());
  ASSERT_EQ(actual_pjrt_layout, new_array_spec.layout);

  // The data in the new array should be retrievable using the new spec.
  TF_ASSERT_OK_AND_ASSIGN(
      auto single_device_arrays,
      new_arrays->front()->DisassembleIntoSingleDeviceArrays(
          ArrayCopySemantics::kReuseInput,
          SingleDeviceShardSemantics::kAddressableShards));
  ASSERT_EQ(single_device_arrays.size(), devices.size());
  for (int i = 0; i < single_device_arrays.size(); ++i) {
    EXPECT_THAT(single_device_arrays[i]->sharding().devices()->devices(),
                ElementsAre(devices[i]));

    std::vector<float> out_data(1);
    auto future = single_device_arrays[i]->CopyToHostBuffer(
        out_data.data(),
        /*byte_strides=*/std::nullopt, ArrayCopySemantics::kAlwaysCopy);
    TF_ASSERT_OK(future.Await());
    EXPECT_THAT(out_data, ElementsAre(3));
  }
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
  absl::c_iota(data, 0);
  Device* device0 = client->addressable_devices().at(0);
  ShardingRef sharding0 = SingleDeviceSharding::Create(device0, MemoryKind());
  Device* device1 = client->addressable_devices().at(1);
  ShardingRef sharding1 = SingleDeviceSharding::Create(device1, MemoryKind());
  UserContextScope user_context_scope(test_util::MakeUserContext(100));

  std::vector<ArrayRef> arrays;
  Shape assembled_shape({4, 3});
  ShardingParam sharding_param(
      /*dim_shards=*/{2, 1}, {/*permutation=*/{0, 1}, /*axis_sizes=*/{2, 1}});

  absl::flat_hash_set<DeviceId> addressable_device_ids;
  for (auto* device : client->addressable_devices()) {
    addressable_device_ids.insert(device->Id());
  }
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef ifrt_device_list,
      client->MakeDeviceList(
          absl::MakeConstSpan(non_addressable_devices).subspan(0, 2)));
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
    EXPECT_EQ(assembled_array->user_context()->Id(), UserContextId(100));

    TF_ASSERT_OK_AND_ASSIGN(
        auto single_device_arrays,
        assembled_array->DisassembleIntoSingleDeviceArrays(
            ArrayCopySemantics::kReuseInput,
            SingleDeviceShardSemantics::kAddressableShards));

    ASSERT_THAT(single_device_arrays, SizeIs(0));
  }
}

TEST(ArrayImplTest, CopyToDifferentDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceListRef devices,
      client->MakeDeviceList(client->addressable_devices()));

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;
  std::vector<ArrayRef> shards;
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
  std::vector<ArrayRef> arrays;
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

  absl::InlinedVector<Device*, 1> new_devices;
  for (auto it = devices->devices().rbegin(); it != devices->devices().rend();
       ++it) {
    new_devices.push_back(*it);
  }
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client->MakeDeviceList(new_devices));
  UserContextScope user_context_scope(test_util::MakeUserContext(100));
  TF_ASSERT_OK_AND_ASSIGN(
      auto new_arrays,
      client->CopyArrays(absl::MakeSpan(arrays), device_list, MemoryKind(),
                         ArrayCopySemantics::kAlwaysCopy));

  for (int i = 0; i < arrays.size(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto expected_sharding,
        arrays[i]->sharding().WithDeviceAssignment(device_list, MemoryKind()));
    EXPECT_EQ(new_arrays[i]->sharding(), *expected_sharding);
    EXPECT_EQ(new_arrays[i]->user_context()->Id(), UserContextId(100));

    TF_ASSERT_OK_AND_ASSIGN(
        auto shards, arrays[i]->DisassembleIntoSingleDeviceArrays(
                         ArrayCopySemantics::kReuseInput,
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
  absl::c_iota(data, 0);
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  std::vector<ArrayRef> arrays;
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
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client->MakeDeviceList({new_device}));
  EXPECT_THAT(client
                  ->CopyArrays(absl::MakeSpan(arrays), std::move(device_list),
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
  absl::c_iota(data, 0);
  Device* device = client->addressable_devices().at(0);
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  std::vector<ArrayRef> arrays;
  for (auto* memory : device->Memories()) {
    ShardingRef sharding = SingleDeviceSharding::Create(device, memory->Kind());
    TF_ASSERT_OK_AND_ASSIGN(arrays.emplace_back(),
                            client->MakeArrayFromHostBuffer(
                                data.data(), dtype, shape,
                                /*byte_strides=*/std::nullopt, sharding,
                                semantics, /*on_done_with_host_buffer=*/{}));
  }

  Device* new_device = client->addressable_devices().at(1);
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client->MakeDeviceList({new_device}));
  EXPECT_THAT(client
                  ->CopyArrays(absl::MakeSpan(arrays), std::move(device_list),
                               MemoryKind(), ArrayCopySemantics::kAlwaysCopy)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ArrayImplTest, CopyPreservesDefaultLayouts) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
  Device* device = client->addressable_devices()[0];

  for (Memory* const src_memory : device->Memories()) {
    for (Memory* const dst_memory : device->Memories()) {
      SCOPED_TRACE(
          absl::StrCat(src_memory->Kind(), " -> ", dst_memory->Kind()));

      ShardingRef sharding =
          SingleDeviceSharding::Create(device, src_memory->Kind());
      TF_ASSERT_OK_AND_ASSIGN(
          auto array,
          client->MakeArrayFromHostBuffer(
              data.data(), dtype, shape, /*byte_strides=*/std::nullopt,
              sharding, Client::HostBufferSemantics::kImmutableOnlyDuringCall,
              /*on_done_with_host_buffer=*/nullptr));
      TF_ASSERT_OK(array->GetReadyFuture().Await());

      TF_ASSERT_OK_AND_ASSIGN(auto src_layout, array->pjrt_layout());
      // `layout` should be either nullptr or a concrete default layout.
      if (src_layout != nullptr) {
        TF_ASSERT_OK_AND_ASSIGN(
            auto src_default_layout,
            client->GetDefaultPjRtLayout(dtype, shape.dims(), device,
                                         src_memory->Kind()));
        EXPECT_EQ(*src_layout, *src_default_layout);
      }

      TF_ASSERT_OK_AND_ASSIGN(
          auto new_arrays, client->CopyArrays(absl::MakeSpan(&array, 1),
                                              std::nullopt, dst_memory->Kind(),
                                              ArrayCopySemantics::kAlwaysCopy));
      ASSERT_THAT(new_arrays, SizeIs(1));
      TF_ASSERT_OK_AND_ASSIGN(auto dst_layout, new_arrays[0]->pjrt_layout());
      // `layout` should be either nullptr or a concrete default layout.
      if (dst_layout != nullptr) {
        TF_ASSERT_OK_AND_ASSIGN(
            auto dst_default_layout,
            client->GetDefaultPjRtLayout(dtype, shape.dims(), device,
                                         dst_memory->Kind()));
        EXPECT_EQ(*dst_layout, *dst_default_layout);
      }
    }
  }
}

TEST(ArrayImplTest, MakeAndCopyZeroSizedBuffers) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  Device* const device = client->addressable_devices().front();
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client->MakeDeviceList({device}));

  DType dtype(DType::kF32);
  Shape shape({0, 1});

  for (Memory* const memory : device->Memories()) {
    SCOPED_TRACE(absl::StrCat(memory->Kind()));

    ShardingRef sharding = ConcreteEvenSharding::Create(
        device_list, memory->Kind(), shape,
        /*shard_shape=*/shape, /*is_fully_replicated=*/true);
    TF_ASSERT_OK_AND_ASSIGN(
        ArrayRef array,
        client->MakeArrayFromHostBuffer(
            nullptr, dtype, shape,
            /*byte_strides=*/std::nullopt, sharding,
            Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr));
    TF_ASSERT_OK(array->GetReadyFuture().Await());

    for (Device* const device : client->addressable_devices()) {
      TF_ASSERT_OK_AND_ASSIGN(DeviceListRef single_device_list,
                              client->MakeDeviceList({device}));
      TF_ASSERT_OK_AND_ASSIGN(
          auto copied,
          client->CopyArrays(absl::MakeSpan(&array, 1),
                             std::move(single_device_list), std::nullopt,
                             ArrayCopySemantics::kReuseInput));
      TF_ASSERT_OK(copied[0]->GetReadyFuture().Await());

      tsl::Future<> future =
          copied[0]->CopyToHostBuffer(nullptr, /*byte_strides=*/std::nullopt,
                                      ArrayCopySemantics::kAlwaysCopy);
      TF_ASSERT_OK(future.Await());
    }
  }
}

TEST(ArrayImplTest, PoisonedZeroSizedBuffers) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  Device* const device = client->addressable_devices().front();
  TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                          client->MakeDeviceList({device}));

  DType dtype(DType::kF32);
  Shape shape({0, 1});
  const absl::Status error = absl::InternalError("injected error");

  for (Memory* const memory : device->Memories()) {
    SCOPED_TRACE(absl::StrCat(memory->Kind()));

    ShardingRef sharding = ConcreteEvenSharding::Create(
        device_list, memory->Kind(), shape,
        /*shard_shape=*/shape, /*is_fully_replicated=*/true);
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<ArrayRef> arrays,
        client->MakeErrorArrays(error, {{
                                           /*dtype=*/dtype,
                                           /*shape=*/shape,
                                           /*sharding=*/sharding,
                                       }}));
    EXPECT_THAT(arrays[0]->GetReadyFuture().Await(),
                StatusIs(error.code(), HasSubstr(error.message())));

    for (Device* const device : client->addressable_devices()) {
      TF_ASSERT_OK_AND_ASSIGN(DeviceListRef single_device_list,
                              client->MakeDeviceList({device}));
      TF_ASSERT_OK_AND_ASSIGN(
          auto copied,
          client->CopyArrays(absl::MakeSpan(arrays),
                             std::move(single_device_list), std::nullopt,
                             ArrayCopySemantics::kReuseInput));
      EXPECT_THAT(copied[0]->GetReadyFuture().Await(),
                  StatusIs(error.code(), HasSubstr(error.message())));

      tsl::Future<> future =
          copied[0]->CopyToHostBuffer(nullptr, /*byte_strides=*/std::nullopt,
                                      ArrayCopySemantics::kAlwaysCopy);
      EXPECT_THAT(future.Await(),
                  StatusIs(error.code(), HasSubstr(error.message())));
    }
  }
}

TEST(ArrayImplTest, CopyArraysExhaustive) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Device* const src_device = client->addressable_devices().front();

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);

  for (Memory* const src_memory : src_device->Memories()) {
    for (Device* const dst_device : client->addressable_devices()) {
      for (Memory* const dst_memory : dst_device->Memories()) {
        SCOPED_TRACE(absl::StrCat(
            src_device->DebugString(), " ", src_memory->DebugString(), " -> ",
            dst_device->DebugString(), " ", dst_memory->DebugString()));

        ShardingRef sharding =
            SingleDeviceSharding::Create(src_device, src_memory->Kind());
        TF_ASSERT_OK_AND_ASSIGN(
            auto array,
            client->MakeArrayFromHostBuffer(
                data.data(), dtype, shape,
                /*byte_strides=*/std::nullopt, sharding,
                Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                /*on_done_with_host_buffer=*/{}));
        EXPECT_EQ(array->sharding(), *sharding);

        ArrayRef new_array;
        {
          TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                                  client->MakeDeviceList({dst_device}));
          TF_ASSERT_OK_AND_ASSIGN(
              auto copied,
              client->CopyArrays(absl::MakeSpan(&array, 1),
                                 std::move(device_list), dst_memory->Kind(),
                                 ArrayCopySemantics::kAlwaysCopy));
          new_array = std::move(copied[0]);
        }
        EXPECT_THAT(new_array->sharding().devices()->devices(),
                    ElementsAre(dst_device));
        EXPECT_EQ(new_array->sharding().memory_kind(), dst_memory->Kind());

        std::vector<float> out_data(6);
        tsl::Future<void> future = new_array->CopyToHostBuffer(
            out_data.data(), /*byte_strides=*/std::nullopt,
            ArrayCopySemantics::kAlwaysCopy);
        TF_ASSERT_OK(future.Await());
        EXPECT_THAT(out_data, ElementsAreArray(data));
      }
    }
  }
}

TEST(ArrayImplTest, CopyArraysSubByteDType) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Device* const src_device = client->addressable_devices().front();

  DType dtype(DType::kS4);
  Shape shape({2, 3});
  std::vector<int8_t> data(6);
  absl::c_iota(data, 0);

  for (Memory* const src_memory : src_device->Memories()) {
    for (Device* const dst_device : client->addressable_devices()) {
      for (Memory* const dst_memory : dst_device->Memories()) {
        SCOPED_TRACE(absl::StrCat(
            src_device->DebugString(), " ", src_memory->DebugString(), " -> ",
            dst_device->DebugString(), " ", dst_memory->DebugString()));

        ShardingRef sharding =
            SingleDeviceSharding::Create(src_device, src_memory->Kind());
        TF_ASSERT_OK_AND_ASSIGN(
            auto array,
            client->MakeArrayFromHostBuffer(
                data.data(), dtype, shape,
                /*byte_strides=*/std::nullopt, sharding,
                Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                /*on_done_with_host_buffer=*/{}));
        EXPECT_EQ(array->sharding(), *sharding);

        ArrayRef new_array;
        {
          TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                                  client->MakeDeviceList({dst_device}));
          TF_ASSERT_OK_AND_ASSIGN(
              auto copied,
              client->CopyArrays(absl::MakeSpan(&array, 1),
                                 std::move(device_list), dst_memory->Kind(),
                                 ArrayCopySemantics::kAlwaysCopy));
          new_array = std::move(copied[0]);
        }
        EXPECT_THAT(new_array->sharding().devices()->devices(),
                    ElementsAre(dst_device));
        EXPECT_EQ(new_array->sharding().memory_kind(), dst_memory->Kind());

        std::vector<int8_t> out_data(6);
        auto future = new_array->CopyToHostBuffer(
            out_data.data(), /*byte_strides=*/std::nullopt,
            ArrayCopySemantics::kAlwaysCopy);
        TF_ASSERT_OK(future.Await());
        EXPECT_THAT(out_data, ElementsAreArray(data));
      }
    }
  }
}

TEST(ArrayImplTest, CopyPoisonedArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Device* const src_device = client->addressable_devices().front();

  for (Memory* const src_memory : src_device->Memories()) {
    for (Device* const dst_device : client->addressable_devices()) {
      for (Memory* const dst_memory : dst_device->Memories()) {
        SCOPED_TRACE(absl::StrCat(
            src_device->DebugString(), " ", src_memory->DebugString(), " -> ",
            dst_device->DebugString(), " ", dst_memory->DebugString()));

        ArraySpec array_spec = {
            /*dtype=*/DType(DType::kF32),
            /*shape=*/Shape({2, 3}),
            /*sharding=*/
            SingleDeviceSharding::Create(src_device, src_memory->Kind()),
        };

        const absl::Status error = absl::InternalError("injected error");
        TF_ASSERT_OK_AND_ASSIGN(auto arrays,
                                client->MakeErrorArrays(error, {array_spec}));

        EXPECT_EQ(arrays.front()->dtype(), array_spec.dtype);
        EXPECT_EQ(arrays.front()->shape(), array_spec.shape);
        EXPECT_EQ(arrays.front()->sharding(), *array_spec.sharding);
        ASSERT_THAT(arrays.front()->GetReadyFuture().Await(),
                    StatusIs(error.code(), HasSubstr(error.message())));

        TF_ASSERT_OK_AND_ASSIGN(DeviceListRef device_list,
                                client->MakeDeviceList({dst_device}));
        TF_ASSERT_OK_AND_ASSIGN(
            auto copied,
            client->CopyArrays(absl::MakeSpan(arrays), std::move(device_list),
                               dst_memory->Kind(),
                               ArrayCopySemantics::kAlwaysCopy));
        ASSERT_EQ(copied.size(), 1);

        EXPECT_THAT(copied[0]->GetReadyFuture().Await(),
                    StatusIs(error.code(), HasSubstr(error.message())));
      }
    }
  }
}

TEST(ArrayImplTest, CopyArraysWithPartialReuse) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  absl::Span<Device* const> devices = client->addressable_devices();
  if (devices.size() < 4) {
    GTEST_SKIP() << "This test requires at least 4 devices";
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto src_devices,
      client->MakeDeviceList({devices[0], devices[1], devices[2], devices[3]}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto dst_devices,
      client->MakeDeviceList({devices[0], devices[2], devices[1], devices[3]}));

  DType dtype(DType::kS32);
  Shape shape({4, 3});
  Shape shard_shape({1, 3});
  TF_ASSERT_OK_AND_ASSIGN(
      ShardingRef sharding,
      ShardingParamSharding::Create(
          ShardingParam(
              /*dim_shards=*/{4, 1},
              {/*permutation=*/{0, 1}, /*axis_sizes=*/{4, 1}}),
          src_devices, MemoryKind()));

  std::vector<ArrayRef> src_arrays;
  {
    auto data0 = std::make_shared<std::vector<int32_t>>(3, 0);
    auto data1 = std::make_shared<std::vector<int32_t>>(3, 1);
    auto data2 = std::make_shared<std::vector<int32_t>>(3, 2);
    auto data3 = std::make_shared<std::vector<int32_t>>(3, 3);

    std::vector<Client::MakeArraysFromHostBufferShardsSpec> specs;
    specs.push_back({
        /*buffers=*/{
            {{0},
             {data0->data(), dtype, shard_shape, /*byte_strides=*/std::nullopt,
              /*on_done_with_host_buffer=*/[data0]() {}}},
            {{1},
             {data1->data(), dtype, shard_shape, /*byte_strides=*/std::nullopt,
              /*on_done_with_host_buffer=*/[data1]() {}}},
            {{2},
             {data2->data(), dtype, shard_shape, /*byte_strides=*/std::nullopt,
              /*on_done_with_host_buffer=*/[data2]() {}}},
            {{3},
             {data3->data(), dtype, shard_shape, /*byte_strides=*/std::nullopt,
              /*on_done_with_host_buffer=*/[data3]() {}}},
        },
        /*array_spec=*/{dtype, shape, sharding, /*layout=*/nullptr},
    });
    TF_ASSERT_OK_AND_ASSIGN(
        src_arrays, client->MakeArraysFromHostBufferShards(
                        absl::MakeSpan(specs),
                        Client::HostBufferSemantics::kImmutableOnlyDuringCall));
    ASSERT_EQ(src_arrays.size(), 1);
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto copied,
      client->CopyArrays(absl::MakeSpan(src_arrays), dst_devices, MemoryKind(),
                         ArrayCopySemantics::kReuseInput));
  ASSERT_EQ(copied.size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(auto single_device_arrays,
                          copied[0]->DisassembleIntoSingleDeviceArrays(
                              ArrayCopySemantics::kReuseInput,
                              SingleDeviceShardSemantics::kAddressableShards));
  ASSERT_EQ(single_device_arrays.size(), 4);

  for (int i = 0; i < single_device_arrays.size(); ++i) {
    EXPECT_THAT(single_device_arrays[i]->sharding().devices()->devices(),
                ElementsAre(dst_devices->devices()[i]));

    std::vector<int32_t> out_data(3);
    auto future = single_device_arrays[i]->CopyToHostBuffer(
        out_data.data(),
        /*byte_strides=*/std::nullopt, ArrayCopySemantics::kAlwaysCopy);
    TF_ASSERT_OK(future.Await());
    EXPECT_THAT(out_data, Each(i));
  }
}

TEST(ArrayImplTest, GetReadyFuture) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  DType dtype(DType::kF32);
  Shape shape({2, 3});
  std::vector<float> data(6);
  absl::c_iota(data, 0);
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
  absl::c_iota(data, 0);
  Device* device = client->addressable_devices().at(0);
  ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());
  auto semantics = Client::HostBufferSemantics::kImmutableOnlyDuringCall;

  std::vector<ValueRef> values;
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
  absl::c_iota(data, 0);
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
  absl::c_iota(data, 0);
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
  absl::c_iota(data, 0);
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
