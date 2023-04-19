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
#include "tensorflow/tsl/framework/device_id_utils.h"

#include <string_view>
#include <vector>

#include "tensorflow/tsl/framework/device_id_manager.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace tsl {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

constexpr std::string_view kTestDeviceType = "CPU";

TEST(DeviceIdUtilsTest, CheckValidTfDeviceIdPass) {
  TfDeviceId tf_device_id(0);
  PlatformDeviceId platform_device_id(1);
  TF_EXPECT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(
      DeviceType(kTestDeviceType), tf_device_id, platform_device_id));
  tsl::CheckValidTfDeviceId("CPU", /*visible_device_count=*/2, tf_device_id);
  DeviceIdManager::TestOnlyReset();
}

TEST(DeviceIdUtilsTest, CheckValidTfDeviceIdNotFound) {
  TfDeviceId tf_device_id(0);
  EXPECT_DEATH(
      tsl::CheckValidTfDeviceId(DeviceType(kTestDeviceType),
                                /*visible_device_count=*/2, tf_device_id),
      "NOT_FOUND: TensorFlow device CPU:0 was not registered");
}

TEST(DeviceIdUtilsTest, CheckValidTfDeviceIdOutsideVisibleDeviceRange) {
  TfDeviceId tf_device_id(0);
  PlatformDeviceId platform_device_id(1);
  TF_EXPECT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(
      DeviceType(kTestDeviceType), tf_device_id, platform_device_id));
  EXPECT_DEATH(tsl::CheckValidTfDeviceId("CPU", /*visible_device_count=*/1,
                                         tf_device_id),
               "platform_device_id is outside discovered device range.");
  DeviceIdManager::TestOnlyReset();
}

TEST(DeviceIdUtilsTest, ParseEmptyVisibleDeviceList) {
  std::vector<PlatformDeviceId> visible_device_order;
  TF_EXPECT_OK(ParseVisibleDeviceList("", 2, &visible_device_order));
  PlatformDeviceId platform_device_id0(0), platform_device_id1(1);
  std::vector<PlatformDeviceId> expected = {platform_device_id0,
                                            platform_device_id1};
  EXPECT_EQ(visible_device_order, expected);
}

TEST(DeviceIdUtilsTest, ParseVisibleDeviceList) {
  std::vector<PlatformDeviceId> visible_device_order;
  TF_EXPECT_OK(ParseVisibleDeviceList("2,1", 3, &visible_device_order));
  PlatformDeviceId platform_device_id2(2), platform_device_id1(1);
  std::vector<PlatformDeviceId> expected = {platform_device_id2,
                                            platform_device_id1};
  EXPECT_EQ(visible_device_order, expected);
}

TEST(DeviceIdUtilsTest, ParseInvalidVisibleDeviceList) {
  std::vector<PlatformDeviceId> visible_device_order;
  EXPECT_THAT(
      ParseVisibleDeviceList("3,1", 3, &visible_device_order),
      StatusIs(tensorflow::error::INVALID_ARGUMENT,
               HasSubstr("'visible_device_list' listed an invalid Device id "
                         "'3' but visible device count is 3")));
}

TEST(DeviceIdUtilsTest, ParseDuplicateVisibleDeviceList) {
  std::vector<PlatformDeviceId> visible_device_order;
  EXPECT_THAT(
      ParseVisibleDeviceList("1,1", 3, &visible_device_order),
      StatusIs(
          tensorflow::error::INVALID_ARGUMENT,
          HasSubstr("visible_device_list contained a duplicate entry: 1,1")));
}

}  // namespace
}  // namespace tsl
