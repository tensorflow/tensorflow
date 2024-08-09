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

#include "xla/python/ifrt/basic_device_allocation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_test_util.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::StartsWith;
using ::tsl::testing::IsOkAndHolds;

class BasicDeviceAllocationTest : public test_util::DeviceTest {};

TEST_P(BasicDeviceAllocationTest, BasicMethods) {
  auto* mock_client = llvm::dyn_cast<MockClient>(client());
  DeviceList devices(DeviceList::Devices(client()->devices().begin(),
                                         client()->devices().end()));
  TF_ASSERT_OK_AND_ASSIGN(auto allocation,
                          BasicDeviceAllocation::Create(devices));

  EXPECT_THAT(allocation->name(), StartsWith("BasicDeviceAllocation-"));
  EXPECT_THAT(allocation->GetDeviceList(),
              ElementsAreArray(client()->devices()));
  EXPECT_THAT(allocation->GetAddressableDeviceList(),
              ElementsAreArray(client()->addressable_devices()));
  EXPECT_EQ(allocation->GetDefaultMemoryKind(), MemoryKind("host"));
  EXPECT_THAT(allocation->GetAllMemoryKinds(), ElementsAre(MemoryKind("host")));

  EXPECT_CALL(*mock_client, GetTopologyForDevices(devices));
  EXPECT_THAT(allocation->GetTopology(), IsOkAndHolds(nullptr));
}

INSTANTIATE_TEST_SUITE_P(NumDevices, BasicDeviceAllocationTest,
                         testing::Values(test_util::DeviceTestParam{
                             /*num_devices=*/4,
                             /*num_addressable_devices=*/2}));

}  // namespace
}  // namespace ifrt
}  // namespace xla
