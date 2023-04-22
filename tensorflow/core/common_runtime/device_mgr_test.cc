/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/device_mgr.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
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

TEST(StaticDeviceMgr, NoCPUDevice) {
  std::unique_ptr<Device> d0(CreateDevice("GPU", "/device:GPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("GPU", "/device:GPU:1"));
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  devices.emplace_back(std::move(d1));
  StaticDeviceMgr lm(std::move(devices));
  EXPECT_EQ(lm.HostCPU(), nullptr);
}

TEST(StaticDeviceMgr, SomeCPUDevice) {
  std::unique_ptr<Device> d0(CreateDevice("GPU", "/device:GPU:0"));
  std::unique_ptr<Device> d1(CreateDevice("GPU", "/device:GPU:1"));
  std::unique_ptr<Device> d2(CreateDevice("CPU", "/device:CPU:0"));
  Device* d2_ptr = d2.get();
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(std::move(d0));
  devices.emplace_back(std::move(d1));
  devices.emplace_back(std::move(d2));
  StaticDeviceMgr lm(std::move(devices));
  EXPECT_EQ(lm.HostCPU(), d2_ptr);
}

}  // namespace
}  // namespace tensorflow
