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

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/gpu_compatibility.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/mldrift_compatibility_binary.h"

namespace tflite {
namespace acceleration {
namespace {

// Runs the same lookup as IsMlDriftGpuSupportedOnThisDevice() on Android, but
// with explicit device properties. Raw (non-canonical) spellings are used so
// canonicalization is covered too. Fixtures mirror entries of
// third_party/tensorflow/lite/experimental/acceleration/google/mldrift_compatibility.json.
bool IsSupported(absl::string_view manufacturer, absl::string_view model,
                 absl::string_view sdk, absl::string_view gl_renderer,
                 int gles_major, int gles_minor) {
  auto list = GPUCompatibilityList::Create(
      g_tflite_acceleration_mldrift_compatibility_binary,
      g_tflite_acceleration_mldrift_compatibility_binary_len);
  EXPECT_NE(list, nullptr);
  if (list == nullptr) return false;

  AndroidInfo android_info;
  android_info.manufacturer = std::string(manufacturer);
  android_info.model = std::string(model);
  android_info.device = std::string(model);
  android_info.android_sdk_version = std::string(sdk);
  ::tflite::gpu::GpuInfo gpu_info;
  gpu_info.opengl_info.renderer_name = std::string(gl_renderer);
  gpu_info.opengl_info.major_version = gles_major;
  gpu_info.opengl_info.minor_version = gles_minor;
  return list->Includes(android_info, gpu_info);
}

// GLES 3.2 / Adreno 720 / vivo / SDK >= 34 -> SUPPORTED (MINIMUM comparison).
TEST(MlDriftCompatibilityTest, AllowlistedDeviceIsSupported) {
  // Arrange
  constexpr absl::string_view kManufacturer = "vivo";
  constexpr absl::string_view kModel = "V2309A";
  constexpr absl::string_view kGlRenderer = "Adreno (TM) 720";
  constexpr int kGlesMajor = 3;
  constexpr int kGlesMinor = 2;

  // Act
  const bool supported_on_sdk_34 = IsSupported(
      kManufacturer, kModel, "34", kGlRenderer, kGlesMajor, kGlesMinor);
  const bool supported_on_sdk_35 = IsSupported(
      kManufacturer, kModel, "35", kGlRenderer, kGlesMajor, kGlesMinor);

  // Assert
  EXPECT_TRUE(supported_on_sdk_34);
  EXPECT_TRUE(supported_on_sdk_35);
}

// GLES 3.1 / Mali-T720 / SM-G532F / SDK <= 23 -> UNSUPPORTED.
TEST(MlDriftCompatibilityTest, DenylistedDeviceIsNotSupported) {
  // Arrange
  constexpr absl::string_view kManufacturer = "samsung";
  constexpr absl::string_view kModel = "SM-G532F";
  constexpr absl::string_view kSdk = "23";
  constexpr absl::string_view kGlRenderer = "Mali-T720";
  constexpr int kGlesMajor = 3;
  constexpr int kGlesMinor = 1;

  // Act
  const bool supported = IsSupported(kManufacturer, kModel, kSdk, kGlRenderer,
                                     kGlesMajor, kGlesMinor);

  // Assert
  EXPECT_FALSE(supported);
}

// Allowlist semantics: unknown devices must fall back to CPU.
TEST(MlDriftCompatibilityTest, UnknownDeviceIsNotSupported) {
  // Arrange
  constexpr absl::string_view kUnknownManufacturer = "acme";
  constexpr absl::string_view kUnknownModel = "phone-1";
  constexpr absl::string_view kSdk = "35";
  constexpr absl::string_view kUnknownRenderer = "Unknown GPU 9000";

  // Act
  const bool supported = IsSupported(kUnknownManufacturer, kUnknownModel, kSdk,
                                     kUnknownRenderer, 3, 2);

  // Assert
  EXPECT_FALSE(supported);
}

// Allowlist semantics: failed GPU probe (empty renderer) must fall back to CPU.
TEST(MlDriftCompatibilityTest, EmptyGpuRendererIsNotSupported) {
  // Arrange: Known device, but GPU probe failed to return renderer string.
  constexpr absl::string_view kManufacturer = "vivo";
  constexpr absl::string_view kModel = "V2309A";
  constexpr absl::string_view kSdk = "34";
  constexpr absl::string_view kEmptyRenderer = "";

  // Act
  const bool supported =
      IsSupported(kManufacturer, kModel, kSdk, kEmptyRenderer, 0, 0);

  // Assert
  EXPECT_FALSE(supported);
}

#if !defined(__ANDROID__)
TEST(MlDriftCompatibilityTest, NonAndroidReturnsTrue) {
  // Arrange: Non-Android host platform.

  // Act
  const bool supported = IsMlDriftGpuSupportedOnThisDevice();

  // Assert
  EXPECT_TRUE(supported);
}
#endif

}  // namespace
}  // namespace acceleration
}  // namespace tflite
