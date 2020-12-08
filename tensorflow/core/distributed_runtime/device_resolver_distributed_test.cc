/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/test_utils.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

using ::testing::Property;
using ::testing::UnorderedElementsAre;

// Create a fake 'Device' whose only interesting attribute is a non-default
// DeviceLocality and incarnation.
std::unique_ptr<Device> NewDevice(const string& type, const string& name) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
    Status Sync() override { return Status::OK(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  attr.set_incarnation(random::New64());
  return absl::make_unique<FakeDevice>(attr);
}

class DeviceResDistTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::vector<std::unique_ptr<Device>> devices;
    devices.push_back(
        NewDevice("CPU", "/job:worker/replica:0/task:0/device:CPU:0"));
    devices.push_back(
        NewDevice("CPU", "/job:worker/replica:0/task:0/device:CPU:1"));
    dev_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    dev_resolver_ =
        absl::make_unique<DeviceResolverDistributed>(dev_mgr_.get());

    std::vector<DeviceAttributes> attributes;
    attributes.push_back(
        NewDevice("CPU", "/job:worker/replica:0/task:1/device:CPU:0")
            ->attributes());
    attributes.push_back(
        NewDevice("CPU", "/job:worker/replica:0/task:1/device:CPU:1")
            ->attributes());
    TF_ASSERT_OK(dev_resolver_->UpdateDeviceAttributes(attributes));
  }

  std::unique_ptr<DeviceMgr> dev_mgr_;
  std::unique_ptr<DeviceResolverDistributed> dev_resolver_;
};

TEST_F(DeviceResDistTest, GetDeviceAttributesLocal) {
  DeviceAttributes attributes;
  TF_ASSERT_OK(dev_resolver_->GetDeviceAttributes(
      "/job:worker/replica:0/task:0/device:CPU:0", &attributes));
  EXPECT_EQ(attributes.name(), "/job:worker/replica:0/task:0/device:CPU:0");
}

TEST_F(DeviceResDistTest, GetDeviceAttributesLocalUnknown) {
  DeviceAttributes attributes;
  EXPECT_TRUE(errors::IsNotFound(dev_resolver_->GetDeviceAttributes(
      "/job:worker/replica:0/task:0/device:CPU:9", &attributes)));
}

TEST_F(DeviceResDistTest, GetAllDeviceAttributes) {
  std::vector<DeviceAttributes> attributes;
  TF_ASSERT_OK(dev_resolver_->GetAllDeviceAttributes(
      "/job:worker/replica:0/task:0", &attributes));
  EXPECT_THAT(attributes,
              UnorderedElementsAre(
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:0/device:CPU:0"),
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:0/device:CPU:1")));
  TF_ASSERT_OK(dev_resolver_->GetAllDeviceAttributes(
      "/job:worker/replica:0/task:1", &attributes));
  EXPECT_THAT(attributes,
              UnorderedElementsAre(
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:1/device:CPU:0"),
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:1/device:CPU:1")));
}

TEST_F(DeviceResDistTest, GetAllDeviceAttributesUnknown) {
  std::vector<DeviceAttributes> attributes;
  EXPECT_TRUE(errors::IsNotFound(dev_resolver_->GetAllDeviceAttributes(
      "/job:worker/replica:0/task:3", &attributes)));
}

TEST_F(DeviceResDistTest, UpdateDeviceAttributes) {
  std::vector<DeviceAttributes> attributes;
  attributes.push_back(
      NewDevice("CPU", "/job:worker/replica:0/task:2/device:CPU:0")
          ->attributes());
  attributes.push_back(
      NewDevice("CPU", "/job:worker/replica:0/task:2/device:CPU:1")
          ->attributes());
  TF_ASSERT_OK(dev_resolver_->UpdateDeviceAttributes(attributes));
  // Get the new task.
  TF_ASSERT_OK(dev_resolver_->GetAllDeviceAttributes(
      "/job:worker/replica:0/task:2", &attributes));
  EXPECT_THAT(attributes,
              UnorderedElementsAre(
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:2/device:CPU:0"),
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:2/device:CPU:1")));
  // Get an existing task.
  TF_ASSERT_OK(dev_resolver_->GetAllDeviceAttributes(
      "/job:worker/replica:0/task:0", &attributes));
  EXPECT_THAT(attributes,
              UnorderedElementsAre(
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:0/device:CPU:0"),
                  Property(&DeviceAttributes::name,
                           "/job:worker/replica:0/task:0/device:CPU:1")));
}

TEST_F(DeviceResDistTest, UpdateDeviceAttributesExisting) {
  std::vector<DeviceAttributes> attributes;
  TF_ASSERT_OK(dev_resolver_->GetAllDeviceAttributes(
      "/job:worker/replica:0/task:0", &attributes));
  TF_ASSERT_OK(dev_resolver_->UpdateDeviceAttributes(attributes));
}

TEST_F(DeviceResDistTest, UpdateDeviceAttributesDifferentIncarnation) {
  std::vector<DeviceAttributes> attributes;
  attributes.push_back(
      NewDevice("CPU", "/job:worker/replica:0/task:0/device:CPU:0")
          ->attributes());
  attributes.push_back(
      NewDevice("CPU", "/job:worker/replica:0/task:0/device:CPU:1")
          ->attributes());
  EXPECT_TRUE(errors::IsFailedPrecondition(
      dev_resolver_->UpdateDeviceAttributes(attributes)));
}

}  // namespace
}  // namespace tensorflow
