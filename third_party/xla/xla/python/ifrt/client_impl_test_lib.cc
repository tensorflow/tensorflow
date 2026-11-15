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

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::Contains;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::SizeIs;

TEST(ClientImplTest, RuntimeTypeAndPlatform) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  EXPECT_THAT(client->runtime_type(), Not(IsEmpty()));
  EXPECT_THAT(client->platform_name(), Not(IsEmpty()));
  EXPECT_THAT(client->platform_version(), Not(IsEmpty()));
  client->platform_id();
  absl::StatusOr<bool> supports_serialization =
      client->Attributes().Get<bool>("supports_executable_serialization");
  if (!absl::IsNotFound(supports_serialization.status())) {
    EXPECT_OK(supports_serialization.status());
  }
}

TEST(ClientImplTest, Devices) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  ASSERT_GE(client->device_count(), 0);
  EXPECT_THAT(client->devices(), SizeIs(client->device_count()));

  ASSERT_GE(client->addressable_device_count(), 0);
  EXPECT_THAT(client->addressable_devices(),
              SizeIs(client->addressable_device_count()));

  for (Device* device : client->devices()) {
    EXPECT_EQ(device->client(), client.get());
    EXPECT_THAT(device->ToString(), Not(IsEmpty()));
    EXPECT_THAT(device->DebugString(), Not(IsEmpty()));
    TF_ASSERT_OK_AND_ASSIGN(auto* looked_up_device,
                            client->LookupDevice(device->Id()));
    EXPECT_EQ(device, looked_up_device);
  }

  for (Device* device : client->addressable_devices()) {
    EXPECT_EQ(device->client(), client.get());
    EXPECT_TRUE(device->IsAddressable());
    EXPECT_EQ(device->ProcessIndex(), client->process_index());
    if (absl::StatusOr<Memory*> default_memory = device->DefaultMemory();
        default_memory.ok()) {
      ASSERT_NE(*default_memory, nullptr);
      EXPECT_THAT(device->Memories(), Contains(*default_memory));
      EXPECT_THAT((*default_memory)->Devices(), Contains(device));
    }
  }

  EXPECT_GE(client->process_index(), 0);
}

TEST(ClientImplTest, GetAllDevices) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  EXPECT_GE(client->GetAllDevices().size(), client->device_count());

  absl::flat_hash_set<DeviceId> seen_device_ids;
  for (Device* device : client->GetAllDevices()) {
    EXPECT_EQ(device->client(), client.get());
    TF_ASSERT_OK_AND_ASSIGN(auto* looked_up_device,
                            client->LookupDevice(device->Id()));
    EXPECT_EQ(device, looked_up_device);
    EXPECT_TRUE(seen_device_ids.insert(device->Id()).second)
        << "Duplicate device ID: " << device->Id();
  }
  for (Device* device : client->devices()) {
    EXPECT_THAT(seen_device_ids, Contains(device->Id()));
  }
}

TEST(ClientImplTest, LookupDeviceWithUnknownIdFails) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client, test_util::GetClient());

  EXPECT_THAT(client->LookupDevice(DeviceId(-1)), Not(absl_testing::IsOk()));
}

TEST(ClientImplTest, LookupAddressableDeviceWithNegativeIdFails) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Client> client, test_util::GetClient());

  absl::StatusOr<Device*> device = client->LookupAddressableDevice(-1);
  if (absl::IsUnimplemented(device.status())) {
    GTEST_SKIP() << "LookupAddressableDevice is not supported.";
  }
  EXPECT_THAT(device, Not(absl_testing::IsOk()));
}

TEST(ClientImplTest, DefaultCompiler) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  EXPECT_THAT(client->GetDefaultCompiler(), NotNull());
}

TEST(ClientImplTest, DefaultDeviceAssignment) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto device_assignment,
        client->GetDefaultDeviceAssignment(client->device_count(), 1));
    EXPECT_EQ(device_assignment.replica_count(), client->device_count());
    EXPECT_EQ(device_assignment.computation_count(), 1);
    for (int i = 0; i < device_assignment.replica_count(); ++i) {
      for (int j = 0; j < device_assignment.computation_count(); ++j) {
        EXPECT_THAT(client->LookupDevice(DeviceId(device_assignment(i, j))),
                    absl_testing::IsOk());
      }
    }
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto device_assignment,
        client->GetDefaultDeviceAssignment(1, client->device_count()));
    EXPECT_EQ(device_assignment.replica_count(), 1);
    EXPECT_EQ(device_assignment.computation_count(), client->device_count());
    for (int i = 0; i < device_assignment.replica_count(); ++i) {
      for (int j = 0; j < device_assignment.computation_count(); ++j) {
        EXPECT_THAT(client->LookupDevice(DeviceId(device_assignment(i, j))),
                    absl_testing::IsOk());
      }
    }
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
