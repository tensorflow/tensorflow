/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/device_set.h"

#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// Return a fake device with the specified type and name.
static Device* Dev(const char* type, const char* name) {
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

class DeviceSetTest : public ::testing::Test {
 public:
  Device* AddDevice(const char* type, const char* name) {
    Device* d = Dev(type, name);
    owned_.emplace_back(d);
    devices_.AddDevice(d);
    return d;
  }

  const DeviceSet& device_set() const { return devices_; }

  std::vector<DeviceType> types() const {
    return devices_.PrioritizedDeviceTypeList();
  }

 private:
  DeviceSet devices_;
  std::vector<std::unique_ptr<Device>> owned_;
};

class DummyFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override {
    return Status::OK();
  }
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    return Status::OK();
  }
};

// Assumes the default priority is '50'.
REGISTER_LOCAL_DEVICE_FACTORY("d1", DummyFactory);
REGISTER_LOCAL_DEVICE_FACTORY("d2", DummyFactory, 51);
REGISTER_LOCAL_DEVICE_FACTORY("d3", DummyFactory, 49);

TEST_F(DeviceSetTest, PrioritizedDeviceTypeList) {
  EXPECT_EQ(50, DeviceSet::DeviceTypeOrder(DeviceType("d1")));
  EXPECT_EQ(51, DeviceSet::DeviceTypeOrder(DeviceType("d2")));
  EXPECT_EQ(49, DeviceSet::DeviceTypeOrder(DeviceType("d3")));

  EXPECT_EQ(std::vector<DeviceType>{}, types());

  AddDevice("d1", "/job:a/replica:0/task:0/device:d1:0");
  EXPECT_EQ(std::vector<DeviceType>{DeviceType("d1")}, types());

  AddDevice("d1", "/job:a/replica:0/task:0/device:d1:1");
  EXPECT_EQ(std::vector<DeviceType>{DeviceType("d1")}, types());

  // D2 is prioritized higher than D1.
  AddDevice("d2", "/job:a/replica:0/task:0/device:d2:0");
  EXPECT_EQ((std::vector<DeviceType>{DeviceType("d2"), DeviceType("d1")}),
            types());

  // D3 is prioritized below D1.
  AddDevice("d3", "/job:a/replica:0/task:0/device:d3:0");
  EXPECT_EQ((std::vector<DeviceType>{
                DeviceType("d2"),
                DeviceType("d1"),
                DeviceType("d3"),
            }),
            types());
}

TEST_F(DeviceSetTest, prioritized_devices) {
  Device* d1 = AddDevice("d1", "/job:a/replica:0/task:0/device:d1:0");
  Device* d2 = AddDevice("d2", "/job:a/replica:0/task:0/device:d2:0");
  EXPECT_EQ(device_set().prioritized_devices(),
            (PrioritizedDeviceVector{std::make_pair(d2, 51),
                                     std::make_pair(d1, 50)}));

  // Cache is rebuilt when a device is added.
  Device* d3 = AddDevice("d3", "/job:a/replica:0/task:0/device:d3:0");
  EXPECT_EQ(
      device_set().prioritized_devices(),
      (PrioritizedDeviceVector{std::make_pair(d2, 51), std::make_pair(d1, 50),
                               std::make_pair(d3, 49)}));
}

TEST_F(DeviceSetTest, prioritized_device_types) {
  AddDevice("d1", "/job:a/replica:0/task:0/device:d1:0");
  AddDevice("d2", "/job:a/replica:0/task:0/device:d2:0");
  EXPECT_EQ(
      device_set().prioritized_device_types(),
      (PrioritizedDeviceTypeVector{std::make_pair(DeviceType("d2"), 51),
                                   std::make_pair(DeviceType("d1"), 50)}));

  // Cache is rebuilt when a device is added.
  AddDevice("d3", "/job:a/replica:0/task:0/device:d3:0");
  EXPECT_EQ(
      device_set().prioritized_device_types(),
      (PrioritizedDeviceTypeVector{std::make_pair(DeviceType("d2"), 51),
                                   std::make_pair(DeviceType("d1"), 50),
                                   std::make_pair(DeviceType("d3"), 49)}));
}

TEST_F(DeviceSetTest, SortPrioritizedDeviceVector) {
  Device* d1_0 = AddDevice("d1", "/job:a/replica:0/task:0/device:d1:0");
  Device* d2_0 = AddDevice("d2", "/job:a/replica:0/task:0/device:d2:0");
  Device* d3_0 = AddDevice("d3", "/job:a/replica:0/task:0/device:d3:0");
  Device* d1_1 = AddDevice("d1", "/job:a/replica:0/task:0/device:d1:1");
  Device* d2_1 = AddDevice("d2", "/job:a/replica:0/task:0/device:d2:1");
  Device* d3_1 = AddDevice("d3", "/job:a/replica:0/task:0/device:d3:1");

  PrioritizedDeviceVector sorted{
      std::make_pair(d3_1, 30), std::make_pair(d1_0, 10),
      std::make_pair(d2_0, 20), std::make_pair(d3_0, 30),
      std::make_pair(d1_1, 20), std::make_pair(d2_1, 10)};

  device_set().SortPrioritizedDeviceVector(&sorted);

  EXPECT_EQ(sorted, (PrioritizedDeviceVector{
                        std::make_pair(d3_0, 30), std::make_pair(d3_1, 30),
                        std::make_pair(d2_0, 20), std::make_pair(d1_1, 20),
                        std::make_pair(d2_1, 10), std::make_pair(d1_0, 10)}));
}

TEST_F(DeviceSetTest, SortPrioritizedDeviceTypeVector) {
  PrioritizedDeviceTypeVector sorted{std::make_pair(DeviceType("d3"), 20),
                                     std::make_pair(DeviceType("d1"), 20),
                                     std::make_pair(DeviceType("d2"), 30)};

  device_set().SortPrioritizedDeviceTypeVector(&sorted);

  EXPECT_EQ(sorted, (PrioritizedDeviceTypeVector{
                        std::make_pair(DeviceType("d2"), 30),
                        std::make_pair(DeviceType("d1"), 20),
                        std::make_pair(DeviceType("d3"), 20)}));
}

}  // namespace
}  // namespace tensorflow
