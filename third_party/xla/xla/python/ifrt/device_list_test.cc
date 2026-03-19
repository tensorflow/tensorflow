/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/device_list.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device.pb.h"
#include "xla/python/ifrt/device_test_util.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/cpu_info.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAreArray;

class DeviceListTest
    : public testing::TestWithParam<test_util::DeviceTestParam> {
 public:
  DeviceListTest() : fixture_(GetParam()) {}

  DeviceListRef GetDevices(absl::Span<const int> device_indices) {
    return fixture_.GetDevices(device_indices);
  }

 private:
  test_util::DeviceTestFixture fixture_;
};

TEST_P(DeviceListTest, AddressableDevices) {
  auto device_list = GetDevices({0, 1});
  std::vector<Device*> addressable_devices;
  for (Device* device : device_list->devices()) {
    if (device->IsAddressable()) {
      addressable_devices.push_back(device);
    }
  }
  EXPECT_THAT(device_list->AddressableDeviceList()->devices(),
              ElementsAreArray(addressable_devices));
}

TEST_P(DeviceListTest, AddressableDevicesFromConcurrentCalls) {
  auto device_list = GetDevices({0, 1});

  const int num_threads = 16;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "test_pool",
      std::min(num_threads, tsl::port::MaxParallelism()));
  std::vector<DeviceList*> addressable_device_lists(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    thread_pool->Schedule([&, i]() {
      addressable_device_lists[i] = device_list->AddressableDeviceList();
      // Touch a device in the list so that tsan can verify access to the
      // content of the addressable device list.
      addressable_device_lists[i]->devices().front()->Id();
    });
  }

  thread_pool.reset();
  for (int i = 0; i < num_threads; ++i) {
    EXPECT_EQ(*addressable_device_lists[i],
              *device_list->AddressableDeviceList());
  }
}

TEST_P(DeviceListTest, IsFullyAddressable) {
  auto device_list = GetDevices({0, 1});
  int num_addressable_devices = 0;
  for (Device* device : device_list->devices()) {
    if (device->IsAddressable()) {
      ++num_addressable_devices;
    }
  }
  if (num_addressable_devices == device_list->size()) {
    EXPECT_TRUE(device_list->IsFullyAddressable());
  } else {
    EXPECT_FALSE(device_list->IsFullyAddressable());
  }
}

TEST_P(DeviceListTest, IdenticalHashFromConcurrentCalls) {
  auto device_list = GetDevices({0, 1});

  const int num_threads = 16;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "test_pool",
      std::min(num_threads, tsl::port::MaxParallelism()));
  std::vector<uint64_t> hashes(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    thread_pool->Schedule([&, i]() { hashes[i] = device_list->hash(); });
  }

  thread_pool.reset();
  for (int i = 0; i < num_threads; ++i) {
    EXPECT_EQ(hashes[i], device_list->hash());
  }
  EXPECT_NE(device_list->hash(), 0);
}

TEST_P(DeviceListTest, EqualityTest) {
  auto device_list1 = GetDevices({0, 1});
  auto device_list2 = GetDevices({0, 1});
  EXPECT_EQ(*device_list1, *device_list2);

  auto device_list3 = device_list1;
  EXPECT_EQ(*device_list1, *device_list3);

  auto device_list4 = std::move(device_list2);
  EXPECT_EQ(*device_list1, *device_list4);

  auto device_list5 = GetDevices({0});
  EXPECT_NE(*device_list1, *device_list5);

  auto device_list6 = GetDevices({1, 0});
  EXPECT_NE(*device_list1, *device_list6);
}

INSTANTIATE_TEST_SUITE_P(
    NumDevices, DeviceListTest,
    testing::Values(test_util::DeviceTestParam{/*num_devices=*/2,
                                               /*num_addressable_devices=*/1},
                    test_util::DeviceTestParam{/*num_devices=*/2,
                                               /*num_addressable_devices=*/2}));

using DeviceListSerDesTestParam =
    std::tuple<SerDesVersion, test_util::DeviceTestParam>;

class DeviceListSerDesTest
    : public testing::TestWithParam<DeviceListSerDesTestParam> {
 public:
  DeviceListSerDesTest()
      : version_(std::get<0>(GetParam())), fixture_(std::get<1>(GetParam())) {}

  SerDesVersion version() const { return version_; }

  Client* client() { return fixture_.client(); }
  DeviceListRef GetDevices(absl::Span<const int> device_indices) {
    return fixture_.GetDevices(device_indices);
  }

 private:
  SerDesVersion version_;
  test_util::DeviceTestFixture fixture_;
};

TEST_P(DeviceListSerDesTest, ToFromProto) {
  auto device_list = GetDevices({0, 1});
  DeviceListProto proto = device_list->ToProto(version());
  TF_ASSERT_OK_AND_ASSIGN(auto device_list_copy,
                          DeviceList::FromProto(client(), proto));
  EXPECT_EQ(*device_list_copy, *device_list);
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion_NumDevices, DeviceListSerDesTest,
    testing::Combine(testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
                     testing::Values(test_util::DeviceTestParam{
                                         /*num_devices=*/2,
                                         /*num_addressable_devices=*/1},
                                     test_util::DeviceTestParam{
                                         /*num_devices=*/2,
                                         /*num_addressable_devices=*/2})));

}  // namespace
}  // namespace ifrt
}  // namespace xla
