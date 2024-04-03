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

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/python/ifrt/device.pb.h"
#include "xla/python/ifrt/sharding_test_util.h"
#include "tsl/platform/statusor.h"

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

INSTANTIATE_TEST_SUITE_P(NumDevices, DeviceListTest,
                         testing::Values(test_util::ShardingTestParam{
                             /*num_devices=*/2,
                             /*num_addressable_devices=*/2}));

}  // namespace
}  // namespace ifrt
}  // namespace xla
