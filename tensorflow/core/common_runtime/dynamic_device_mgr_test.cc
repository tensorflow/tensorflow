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
  std::unique_ptr<Device> d0(CreateDevice("GPU", "/device:GPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("CPU", "/device:CPU:1"));
  Device* d0_ptr = d0.get();
  Device* d1_ptr = d1.get();

  auto dm = MakeUnique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 1);

  std::vector<Device*> removed_devices{d0_ptr, d1_ptr};
  Status s = dm->RemoveDevices(removed_devices);
  EXPECT_TRUE(absl::StrContains(s.error_message(), "Unknown device"));
  EXPECT_EQ(dm->ListDevices().size(), 1);  // d0 *not* removed.
}

TEST(DynamicDeviceMgrTest, RemoveNonExistingDeviceByNameFromMgr) {
  std::unique_ptr<Device> d0(CreateDevice("GPU", "/device:GPU:0"));
  string d0_name = "/device:GPU:0";
  string d1_name = "/device:CPU:0";

  auto dm = MakeUnique<DynamicDeviceMgr>();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 1);

  std::vector<string> removed_devices{d0_name, d1_name};
  Status s = dm->RemoveDevicesByName(removed_devices);
  EXPECT_TRUE(absl::StrContains(s.error_message(), "unknown device"));
  EXPECT_EQ(dm->ListDevices().size(), 1);  // d0 *not* removed
}

TEST(DynamicDeviceMgrTest, HostCPU) {
  auto dm = MakeUnique<DynamicDeviceMgr>();

  // If there are no CPU devices, HostCPU() should return nullptr.
  std::unique_ptr<Device> gpu(CreateDevice("GPU", "/device:GPU:0"));
  Device* gpu_ptr = gpu.get();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(gpu));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 1);
  EXPECT_EQ(dm->HostCPU(), nullptr);

  // After adding a CPU device, it should return that device.
  std::unique_ptr<Device> cpu0(CreateDevice("CPU", "/device:CPU:0"));
  Device* cpu0_ptr = cpu0.get();
  devices.clear();
  devices.emplace_back(std::move(cpu0));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 2);
  EXPECT_EQ(dm->HostCPU(), cpu0_ptr);

  // If we add another CPU device, HostCPU() should remain the same.
  std::unique_ptr<Device> cpu1(CreateDevice("CPU", "/device:CPU:1"));
  Device* cpu1_ptr = cpu1.get();
  devices.clear();
  devices.emplace_back(std::move(cpu1));
  TF_CHECK_OK(dm->AddDevices(std::move(devices)));
  EXPECT_EQ(dm->ListDevices().size(), 3);
  EXPECT_EQ(dm->HostCPU(), cpu0_ptr);

  // Once we have a HostCPU() device, we can't remove it ...
  std::vector<Device*> removed{gpu_ptr, cpu0_ptr};
  EXPECT_TRUE(absl::StrContains(dm->RemoveDevices(removed).error_message(),
                                "Can not remove HostCPU device"));
  EXPECT_EQ(dm->ListDevices().size(), 3);
  EXPECT_EQ(dm->HostCPU(), cpu0_ptr);

  // ... but we should be able to remove another CPU device.
  removed = std::vector<Device*>{cpu1_ptr};
  TF_CHECK_OK(dm->RemoveDevices(removed));
  EXPECT_EQ(dm->ListDevices().size(), 2);
  EXPECT_EQ(dm->HostCPU(), cpu0_ptr);
}

}  // namespace
}  // namespace tensorflow
