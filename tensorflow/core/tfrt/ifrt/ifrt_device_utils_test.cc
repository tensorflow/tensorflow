/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_device_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/service/device_assignment.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {
using ::testing::ElementsAre;

static constexpr int kNumReplicas = 1;
static constexpr int kNumCoresPerReplica = 2;
// Intentionally have more devices than kNumReplicas * kNumCoresPerReplica for
// testing purposes.
static constexpr int kNumDevices = 2;

class IfrtDeviceUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(client_, xla::ifrt::test_util::GetClient());
    for (auto* device : client_->devices()) {
      ASSERT_OK_AND_ASSIGN(
          auto coords,
          device->Attributes().Get<std::vector<int64_t>>("coords"));
      ASSERT_OK_AND_ASSIGN(auto core,
                           device->Attributes().Get<int64_t>("core_on_chip"));
      ASSERT_EQ(coords.size(), 3);
      devices_.push_back(device);
      device_coords_.push_back(
          {static_cast<int>(coords[0]), static_cast<int>(coords[1]),
           static_cast<int>(coords[2]), static_cast<int>(core)});
    }
    ASSERT_GE(devices_.size(), kNumDevices);
  }

  std::shared_ptr<xla::ifrt::Client> client_;
  std::vector<xla::ifrt::Device*> devices_;
  std::vector<std::vector<int>> device_coords_;
};

TEST_F(IfrtDeviceUtilsTest, Basic) {
  std::vector<int> device_assignment_attr;
  device_assignment_attr.insert(device_assignment_attr.end(),
                                device_coords_[1].begin(),
                                device_coords_[1].end());
  device_assignment_attr.insert(device_assignment_attr.end(),
                                device_coords_[0].begin(),
                                device_coords_[0].end());
  TF_ASSERT_OK_AND_ASSIGN(
      auto devices_from_attribute,
      GetAssignedIfrtDevices(*client_, kNumReplicas, kNumCoresPerReplica,
                             device_assignment_attr));
  EXPECT_THAT(devices_from_attribute, ElementsAre(devices_[1], devices_[0]));
}

TEST_F(IfrtDeviceUtilsTest, InvertCoordinates) {
  std::vector<int> device_assignment_attr;
  device_assignment_attr.insert(device_assignment_attr.end(),
                                device_coords_[0].begin(),
                                device_coords_[0].end());
  device_assignment_attr.insert(device_assignment_attr.end(),
                                device_coords_[1].begin(),
                                device_coords_[1].end());
  TF_ASSERT_OK_AND_ASSIGN(
      auto devices_from_attribute,
      GetAssignedIfrtDevices(*client_, kNumReplicas, kNumCoresPerReplica,
                             device_assignment_attr));
  EXPECT_THAT(devices_from_attribute, ElementsAre(devices_[0], devices_[1]));
}

TEST_F(IfrtDeviceUtilsTest, EmptyDeviceAssignmentShallReturnDefault) {
  TF_ASSERT_OK_AND_ASSIGN(
      xla::DeviceAssignment default_assignment,
      client_->GetDefaultDeviceAssignment(kNumReplicas, kNumCoresPerReplica));
  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::Device * dev0,
      client_->LookupDevice(xla::ifrt::DeviceId(default_assignment(0, 0))));
  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::Device * dev1,
      client_->LookupDevice(xla::ifrt::DeviceId(default_assignment(0, 1))));

  TF_ASSERT_OK_AND_ASSIGN(
      auto devices_from_attribute,
      GetAssignedIfrtDevices(*client_, kNumReplicas, kNumCoresPerReplica,
                             std::nullopt));
  EXPECT_THAT(devices_from_attribute, ElementsAre(dev0, dev1));
}

TEST_F(IfrtDeviceUtilsTest, MismatchCoordinatesShallFail) {
  std::vector<int> device_assignment_attr = {999, 999, 999, 999,
                                             999, 999, 999, 998};
  auto status = GetAssignedIfrtDevices(*client_, 1, 2, device_assignment_attr);
  EXPECT_THAT(status,
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
