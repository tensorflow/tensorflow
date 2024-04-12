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

#include "xla/python/ifrt/device.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/synchronization/blocking_counter.h"
#include "xla/python/ifrt/device.pb.h"
#include "xla/python/ifrt/sharding_test_util.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace ifrt {
namespace {

class DeviceListTest : public test_util::ShardingTest {};

TEST_P(DeviceListTest, ToFromProto) {
  auto device_list = GetDevices({0, 1});
  DeviceListProto proto = device_list.ToProto();
  auto lookup_device_func = [&](int device_id) -> absl::StatusOr<Device*> {
    return client()->LookupDevice(device_id);
  };
  TF_ASSERT_OK_AND_ASSIGN(auto device_list_copy,
                          DeviceList::FromProto(lookup_device_func, proto));
  EXPECT_EQ(device_list_copy, device_list);
}

TEST_P(DeviceListTest, IdenticalHashFromConcurrentCalls) {
  auto device_list = GetDevices({0, 1});

  const int num_threads = 16;
  absl::BlockingCounter counter(num_threads);
  tsl::thread::ThreadPool thread_pool(
      tsl::Env::Default(), tsl::ThreadOptions(), "test_pool",
      std::min(num_threads, tsl::port::MaxParallelism()));
  std::vector<uint64_t> hashes(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    thread_pool.Schedule([&, i]() {
      hashes[i] = device_list.hash();
      counter.DecrementCount();
    });
  }

  counter.Wait();
  for (int i = 0; i < num_threads; ++i) {
    EXPECT_EQ(hashes[i], device_list.hash());
  }
  EXPECT_NE(device_list.hash(), 0);
}

TEST_P(DeviceListTest, EqualityTest) {
  auto device_list1 = GetDevices({0, 1});
  auto device_list2 = GetDevices({0, 1});
  EXPECT_EQ(device_list1, device_list2);

  auto device_list3 = device_list1;
  EXPECT_EQ(device_list1, device_list3);

  auto device_list4 = std::move(device_list2);
  EXPECT_EQ(device_list1, device_list4);

  auto device_list5 = GetDevices({0});
  EXPECT_NE(device_list1, device_list5);

  auto device_list6 = GetDevices({1, 0});
  EXPECT_NE(device_list1, device_list6);
}

INSTANTIATE_TEST_SUITE_P(NumDevices, DeviceListTest,
                         testing::Values(test_util::ShardingTestParam{
                             /*num_devices=*/2,
                             /*num_addressable_devices=*/2}));

}  // namespace
}  // namespace ifrt
}  // namespace xla
