/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {

// Return a fake device with the specified type and name.
static Device* CreateDevice(const char* type, const char* name) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
    Status Sync() override { return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  return new FakeDevice(attr);
}

TEST(DynamicDeviceMgrTest, AddDeviceToMgr) {
  std::unique_ptr<Device> d0(CreateDevice("CPU", "/device:CPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("CPU", "/device:CPU:1"));

  auto dm = MakeUnique<DynamicDeviceMgr>();
  EXPECT_EQ(dm->ListDevices().size(), 0);

  std::vector<std::unique_ptr<Device>> added_devices;
  added_devices.emplace_back(std::move(d0));
  added_devices.emplace_back(std::move(d1));
  TF_CHECK_OK(dm->AddDevices(std::move(added_devices)));
  EXPECT_EQ(dm->ListDevices().size(), 2);
}

TEST(DynamicDeviceMgrTest, RemoveDeviceFromMgr) {
  std::unique_ptr<Device> d0(CreateDevice("CPU", "/device:CPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("CPU", "/device:CPU:1"));
  Device* d1_ptr = d1.get();

  auto dm = MakeUnique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  devices.emplace_back(std::move(d1));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 2);

  std::vector<Device*> removed_devices{d1_ptr};
  TF_CHECK_OK(dm->RemoveDevices(removed_devices));
  EXPECT_EQ(dm->ListDevices().size(), 1);
}

TEST(DynamicDeviceMgrTest, RemoveDeviceByNameFromMgr) {
  std::unique_ptr<Device> d0(CreateDevice("CPU", "/device:CPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("CPU", "/device:CPU:1"));
  string d1_name = "/device:CPU:1";

  auto dm = MakeUnique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  devices.emplace_back(std::move(d1));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 2);

  std::vector<string> removed_devices{d1_name};
  TF_CHECK_OK(dm->RemoveDevicesByName(removed_devices));
  EXPECT_EQ(dm->ListDevices().size(), 1);
}

TEST(DynamicDeviceMgrTest, AddRepeatedDeviceToMgr) {
  std::unique_ptr<Device> d0(CreateDevice("CPU", "/device:CPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("CPU", "/device:CPU:0"));

  auto dm = MakeUnique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 1);

  std::vector<std::unique_ptr<Device>> added_devices;
  added_devices.emplace_back(std::move(d1));
  Status s = dm->AddDevices(std::move(added_devices));
  EXPECT_TRUE(absl::StrContains(s.error_message(),
                                "name conflicts with an existing deivce"));
}

TEST(DynamicDeviceMgrTest, RemoveNonExistingDeviceFromMgr) {
  std::unique_ptr<Device> d0(CreateDevice("CPU", "/device:CPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("CPU", "/device:CPU:1"));
  Device* d1_ptr = d1.get();

  auto dm = MakeUnique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 1);

  std::vector<Device*> removed_devices{d1_ptr};
  Status s = dm->RemoveDevices(removed_devices);
  EXPECT_TRUE(absl::StrContains(s.error_message(), "Unknown device"));
}

TEST(DynamicDeviceMgrTest, RemoveNonExistingDeviceByNameFromMgr) {
  std::unique_ptr<Device> d0(CreateDevice("CPU", "/device:CPU:0"));
  string d1_name = "/device:CPU:1";

  auto dm = MakeUnique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 1);

  std::vector<string> removed_devices{d1_name};
  Status s = dm->RemoveDevicesByName(removed_devices);
  EXPECT_TRUE(absl::StrContains(s.error_message(), "unknown device"));
}

}  // namespace
}  // namespace tensorflow
