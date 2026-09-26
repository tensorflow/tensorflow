/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/compatibility/mldrift_compatibility.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/devicedb-sample.h"

namespace tflite {
namespace acceleration {
namespace {

AndroidInfo MakeAndroidInfo(absl::string_view manufacturer,
                            absl::string_view model, absl::string_view sdk) {
  AndroidInfo info;
  info.manufacturer = std::string(manufacturer);
  info.model = std::string(model);
  info.device = std::string(model);
  info.android_sdk_version = std::string(sdk);
  return info;
}

::tflite::gpu::GpuInfo MakeGpuInfo(absl::string_view renderer, int major,
                                   int minor) {
  ::tflite::gpu::GpuInfo info;
  info.opengl_info.renderer_name = std::string(renderer);
  info.opengl_info.major_version = major;
  info.opengl_info.minor_version = minor;
  return info;
}

TEST(MlDriftCompatibilityTest, SupportedInDatabaseReturnsTrue) {
  AndroidInfo android_info = MakeAndroidInfo("sample_mfr", "m712c", "24");
  ::tflite::gpu::GpuInfo gpu_info = MakeGpuInfo("Mali", 3, 1);

  EXPECT_TRUE(
      IsMlDriftGpuSupported(g_tflite_acceleration_devicedb_sample_binary,
                            g_tflite_acceleration_devicedb_sample_binary_len,
                            android_info, gpu_info));
}

TEST(MlDriftCompatibilityTest, UnsupportedInDatabaseReturnsFalse) {
  AndroidInfo android_info = MakeAndroidInfo("Samsung", "SM-G960F", "28");
  ::tflite::gpu::GpuInfo gpu_info = MakeGpuInfo("Mali-G72", 3, 2);

  EXPECT_FALSE(
      IsMlDriftGpuSupported(g_tflite_acceleration_devicedb_sample_binary,
                            g_tflite_acceleration_devicedb_sample_binary_len,
                            android_info, gpu_info));
}

TEST(MlDriftCompatibilityTest, UnknownDeviceInDatabaseReturnsFalse) {
  AndroidInfo android_info =
      MakeAndroidInfo("unknown_mfr", "unknown_model", "35");
  ::tflite::gpu::GpuInfo gpu_info = MakeGpuInfo("Unknown GPU", 3, 2);

  EXPECT_FALSE(
      IsMlDriftGpuSupported(g_tflite_acceleration_devicedb_sample_binary,
                            g_tflite_acceleration_devicedb_sample_binary_len,
                            android_info, gpu_info));
}

TEST(MlDriftCompatibilityTest, InvalidBinaryReturnsFalse) {
  AndroidInfo android_info = MakeAndroidInfo("sample_mfr", "m712c", "24");
  ::tflite::gpu::GpuInfo gpu_info = MakeGpuInfo("Mali", 3, 1);

  EXPECT_FALSE(IsMlDriftGpuSupported(nullptr, 0, android_info, gpu_info));
}

TEST(MlDriftCompatibilityTest,
     ProductionCompatibilityListRejectsUnknownDevice) {
  AndroidInfo android_info =
      MakeAndroidInfo("unknown_mfr", "unknown_model", "35");
  ::tflite::gpu::GpuInfo gpu_info = MakeGpuInfo("Unknown GPU", 3, 2);

  EXPECT_FALSE(IsMlDriftGpuSupported(android_info, gpu_info));
}

#if !defined(__ANDROID__)
TEST(MlDriftCompatibilityTest, NonAndroidReturnsTrue) {
  EXPECT_TRUE(IsMlDriftGpuSupportedOnThisDevice());
}
#endif

}  // namespace
}  // namespace acceleration
}  // namespace tflite
