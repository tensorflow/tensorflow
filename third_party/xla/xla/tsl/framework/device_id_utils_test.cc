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
#include "xla/tsl/framework/device_id_utils.h"

#include <string_view>
#include <vector>

#include "xla/tsl/framework/device_id_manager.h"
#include "xla/tsl/util/device_name_utils.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"

namespace tsl {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

constexpr std::string_view kTestDeviceType = "CPU";

PlatformDeviceId TfToPlatformDeviceId(TfDeviceId tf_device_id) {
  PlatformDeviceId platform_device_id;
  TF_CHECK_OK(DeviceIdManager::TfToPlatformDeviceId(
      DeviceType(kTestDeviceType), tf_device_id, &platform_device_id));
  return platform_device_id;
}

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

TEST(DeviceIdUtilsTest, GetNumberTfDevicesDefault) {
  TF_ASSERT_OK_AND_ASSIGN(size_t num_tf_device,
                          GetNumberTfDevicesAndConfigurePlatformDeviceId(
                              {}, kTestDeviceType, "", 2));

  EXPECT_EQ(num_tf_device, 2);
  TfDeviceId tf_device_id_0(0);
  PlatformDeviceId expected_0(0);
  EXPECT_EQ(expected_0, TfToPlatformDeviceId(tf_device_id_0));
  TfDeviceId tf_device_id_1(1);
  PlatformDeviceId expected_1(1);
  EXPECT_EQ(expected_1, TfToPlatformDeviceId(tf_device_id_1));
  DeviceIdManager::TestOnlyReset();
}

TEST(DeviceIdUtilsTest, GetNumberTfDevicesWithVisibleDeviceList) {
  TF_ASSERT_OK_AND_ASSIGN(size_t num_tf_device,
                          GetNumberTfDevicesAndConfigurePlatformDeviceId(
                              {}, kTestDeviceType, "2,0", 3));

  EXPECT_EQ(num_tf_device, 2);
  TfDeviceId tf_device_id_0(0);
  PlatformDeviceId expected_2(2);
  EXPECT_EQ(expected_2, TfToPlatformDeviceId(tf_device_id_0));
  TfDeviceId tf_device_id_1(1);
  PlatformDeviceId expected_0(0);
  EXPECT_EQ(expected_0, TfToPlatformDeviceId(tf_device_id_1));
  DeviceIdManager::TestOnlyReset();
}

TEST(DeviceIdUtilsTest, GetNumberTfDevicesWithSessionOptionDeviceCount) {
  TF_ASSERT_OK_AND_ASSIGN(
      size_t num_tf_device,
      GetNumberTfDevicesAndConfigurePlatformDeviceId(
          {{std::string(kTestDeviceType), 2}}, kTestDeviceType, "1,0,2", 3));

  EXPECT_EQ(num_tf_device, 2);
  TfDeviceId tf_device_id_0(0);
  PlatformDeviceId expected_1(1);
  EXPECT_EQ(expected_1, TfToPlatformDeviceId(tf_device_id_0));
  TfDeviceId tf_device_id_1(1);
  PlatformDeviceId expected_0(0);
  EXPECT_EQ(expected_0, TfToPlatformDeviceId(tf_device_id_1));
  DeviceIdManager::TestOnlyReset();
}

TEST(DeviceIdUtilsTest, GetPlatformDeviceId) {
  TfDeviceId tf_device_id(0);
  PlatformDeviceId platform_device_id(1);
  TF_EXPECT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(
      DeviceType(kTestDeviceType), tf_device_id, platform_device_id));
  DeviceNameUtils::ParsedName device_name;
  device_name.id = 0;

  TF_ASSERT_OK_AND_ASSIGN(int device_id,
                          GetPlatformDeviceIdFromDeviceParsedName(
                              device_name, DeviceType(kTestDeviceType)));

  EXPECT_EQ(device_id, 1);
  DeviceIdManager::TestOnlyReset();
}

TEST(DeviceIdUtilsTest, GetPlatformDeviceIdNotFound) {
  DeviceNameUtils::ParsedName device_name;
  device_name.id = 0;

  EXPECT_THAT(
      GetPlatformDeviceIdFromDeviceParsedName(device_name,
                                              DeviceType(kTestDeviceType)),
      StatusIs(tensorflow::error::NOT_FOUND,
               HasSubstr("TensorFlow device CPU:0 was not registered")));
}

TEST(DeviceIdUtilsTest, GetDeviceIdWithPlatformDeviceId) {
  TfDeviceId tf_device_id(0);
  PlatformDeviceId platform_device_id(1);
  TF_EXPECT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(
      DeviceType(kTestDeviceType), tf_device_id, platform_device_id));
  DeviceNameUtils::ParsedName device_name;
  device_name.id = 0;

  TF_ASSERT_OK_AND_ASSIGN(int device_id,
                          GetDeviceIdFromDeviceParsedName(
                              device_name, DeviceType(kTestDeviceType)));

  EXPECT_EQ(device_id, 1);
  DeviceIdManager::TestOnlyReset();
}

TEST(DeviceIdUtilsTest, GetDeviceIdWithoutPlatformDeviceId) {
  DeviceNameUtils::ParsedName device_name;
  device_name.id = 0;

  TF_ASSERT_OK_AND_ASSIGN(int device_id,
                          GetDeviceIdFromDeviceParsedName(
                              device_name, DeviceType(kTestDeviceType)));

  EXPECT_EQ(device_id, 0);
}

}  // namespace
}  // namespace tsl
