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

#include "tensorflow/core/common_runtime/device/device_id_manager.h"

#include <vector>

#include <gmock/gmock.h>
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

PlatformDeviceId TfToPlatformDeviceId(const DeviceType& type, TfDeviceId tf) {
  PlatformDeviceId platform_device_id;
  TF_CHECK_OK(
      DeviceIdManager::TfToPlatformDeviceId(type, tf, &platform_device_id));
  return platform_device_id;
}

TEST(DeviceIdManagerTest, Basics) {
  DeviceType device_type("GPU");
  TfDeviceId key_0(0);
  PlatformDeviceId value_0(0);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type, key_0,
                                                             value_0));
  EXPECT_EQ(value_0, TfToPlatformDeviceId(device_type, key_0));

  // Multiple calls to map the same value is ok.
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type, key_0,
                                                             value_0));
  EXPECT_EQ(value_0, TfToPlatformDeviceId(device_type, key_0));

  // Map a different TfDeviceId to a different value.
  TfDeviceId key_1(3);
  PlatformDeviceId value_1(2);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type, key_1,
                                                             value_1));
  EXPECT_EQ(value_1, TfToPlatformDeviceId(device_type, key_1));

  // Mapping a different TfDeviceId to the same value is ok.
  TfDeviceId key_2(10);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type, key_2,
                                                             value_1));
  EXPECT_EQ(value_1, TfToPlatformDeviceId(device_type, key_2));

  // Mapping the same TfDeviceId to a different value.
  ASSERT_FALSE(
      DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type, key_2, value_0)
          .ok());

  // Getting a nonexistent mapping.
  ASSERT_FALSE(DeviceIdManager::TfToPlatformDeviceId(device_type,
                                                     TfDeviceId(100), &value_0)
                   .ok());
}

TEST(DeviceIdManagerTest, TwoDevices) {
  // Setup 0 --> 0 mapping for device GPU.
  DeviceType device_type0("GPU");
  TfDeviceId key_0(0);
  PlatformDeviceId value_0(0);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type0,
                                                             key_0, value_0));
  // Setup 2 --> 3 mapping for device XPU.
  DeviceType device_type1("XPU");
  TfDeviceId key_1(2);
  PlatformDeviceId value_1(3);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(device_type1,
                                                             key_1, value_1));

  // Key 0 is available for device GPU.
  EXPECT_EQ(value_0, TfToPlatformDeviceId(device_type0, key_0));
  // Key 2 is available for device XPU.
  EXPECT_EQ(value_1, TfToPlatformDeviceId(device_type1, key_1));
  // Key 2 is *not* available for device GPU
  ASSERT_FALSE(
      DeviceIdManager::TfToPlatformDeviceId(device_type0, key_1, &value_0)
          .ok());
  // Key 0 is not available for device XPU.
  ASSERT_FALSE(
      DeviceIdManager::TfToPlatformDeviceId(device_type1, key_0, &value_1)
          .ok());
  // Key 0 is not available for device FOO.
  ASSERT_FALSE(
      DeviceIdManager::TfToPlatformDeviceId("FOO", key_0, &value_0).ok());
}

TEST(DeviceIdManagerTest, GetTfDevicesOnSamePlatform) {
  // Setup 0 --> 0 and 1 --> 0 mapping for device GPU.
  DeviceType device_gpu("GPU");
  TfDeviceId tf_device_0(0);
  PlatformDeviceId platform_0(0);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(
      device_gpu, tf_device_0, platform_0));
  TfDeviceId tf_device_1(1);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(
      device_gpu, tf_device_1, platform_0));

  // Setup 2 --> 3 mapping for device XPU.
  DeviceType device_xpu("XPU");
  TfDeviceId tf_device_2(2);
  PlatformDeviceId platform_1(3);
  TF_ASSERT_OK(DeviceIdManager::InsertTfPlatformDeviceIdPair(
      device_xpu, tf_device_2, platform_1));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<TfDeviceId> tf_device_ids_gpu,
      DeviceIdManager::GetTfDevicesOnPlatform(device_gpu, platform_0));
  EXPECT_THAT(tf_device_ids_gpu,
              UnorderedElementsAre(tf_device_0, tf_device_1));

  TF_ASSERT_OK_AND_ASSIGN(
      tf_device_ids_gpu,
      DeviceIdManager::GetTfDevicesOnPlatform(device_gpu, platform_1));
  EXPECT_THAT(tf_device_ids_gpu, IsEmpty());

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<TfDeviceId> tf_device_ids_xpu,
      DeviceIdManager::GetTfDevicesOnPlatform(device_xpu, platform_1));
  EXPECT_THAT(tf_device_ids_xpu, UnorderedElementsAre(tf_device_2));
}

}  // namespace
}  // namespace tensorflow
