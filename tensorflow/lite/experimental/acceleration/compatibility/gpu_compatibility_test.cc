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
#include "tensorflow/lite/experimental/acceleration/compatibility/gpu_compatibility.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/acceleration/compatibility/devicedb-sample.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/variables.h"

namespace {

class GPUCompatibilityTest : public ::testing::Test {
 protected:
  GPUCompatibilityTest() {
    list_ = tflite::acceleration::GPUCompatibilityList::Create(
        g_tflite_acceleration_devicedb_sample_binary,
        g_tflite_acceleration_devicedb_sample_binary_len);
  }

  std::unique_ptr<tflite::acceleration::GPUCompatibilityList> list_;
};

TEST_F(GPUCompatibilityTest, ReturnsUnsupportedStatus) {
  ASSERT_TRUE(list_ != nullptr);

  std::map<std::string, std::string> variables = {
      {tflite::acceleration::kAndroidSdkVersion, "28"},
      {tflite::acceleration::kDeviceModel, "shiraz-ag-2011"},
  };

  EXPECT_EQ(list_->GetStatus(variables),
            tflite::acceleration::gpu::CompatibilityStatus::kUnsupported);
}

TEST_F(GPUCompatibilityTest, ReturnsSupportedStatus) {
  ASSERT_TRUE(list_ != nullptr);

  std::map<std::string, std::string> variables = {
      {tflite::acceleration::kAndroidSdkVersion, "24"},
      {tflite::acceleration::kDeviceModel, "M712C"},
      {tflite::acceleration::kOpenGLESVersion, "3.1"},
  };

  EXPECT_EQ(list_->GetStatus(variables),
            tflite::acceleration::gpu::CompatibilityStatus::kSupported);
}

TEST_F(GPUCompatibilityTest, ReturnsUnknownStatus) {
  ASSERT_TRUE(list_ != nullptr);

  std::map<std::string, std::string> variables = {
      {tflite::acceleration::kAndroidSdkVersion, "26"},
      {tflite::acceleration::kDeviceModel, "mag2016"},
      {tflite::acceleration::kOpenGLESVersion, "3.1"},
  };

  EXPECT_EQ(list_->GetStatus(variables),
            tflite::acceleration::gpu::CompatibilityStatus::kUnknown);
}

TEST_F(GPUCompatibilityTest, ReturnsSupportedForFullMatch) {
  ASSERT_TRUE(list_ != nullptr);

  tflite::acceleration::AndroidInfo android_info = {.android_sdk_version = "24",
                                                    .model = "m712c"};

  tflite::gpu::GpuInfo tflite_gpu_info;
  tflite_gpu_info.opengl_info.major_version = 3;
  tflite_gpu_info.opengl_info.minor_version = 1;

  EXPECT_TRUE(list_->Includes(android_info, tflite_gpu_info));
}

TEST_F(GPUCompatibilityTest, ReturnsUnsupportedForFullMatch) {
  ASSERT_TRUE(list_ != nullptr);

  tflite::acceleration::AndroidInfo android_info = {.android_sdk_version = "28",
                                                    .model = "SM-G960F",
                                                    .device = "starlte",
                                                    .manufacturer = "Samsung"};
  tflite::gpu::GpuInfo tflite_gpu_info;
  tflite_gpu_info.opengl_info.renderer_name = "Mali-G72";
  tflite_gpu_info.opengl_info.major_version = 3;
  tflite_gpu_info.opengl_info.minor_version = 2;
  EXPECT_FALSE(list_->Includes(android_info, tflite_gpu_info));
}

TEST_F(GPUCompatibilityTest, ReturnsDefaultOptions) {
  ASSERT_TRUE(list_ != nullptr);
  tflite::acceleration::AndroidInfo android_info;
  tflite::gpu::GpuInfo tflite_gpu_info;
  auto default_options = TfLiteGpuDelegateOptionsV2Default();
  auto best_options = list_->GetBestOptionsFor(android_info, tflite_gpu_info);
  EXPECT_EQ(best_options.is_precision_loss_allowed,
            default_options.is_precision_loss_allowed);
  EXPECT_EQ(best_options.inference_preference,
            default_options.inference_preference);
  EXPECT_EQ(best_options.inference_priority1,
            default_options.inference_priority1);
  EXPECT_EQ(best_options.inference_priority2,
            default_options.inference_priority2);
  EXPECT_EQ(best_options.inference_priority3,
            default_options.inference_priority3);
  EXPECT_EQ(best_options.experimental_flags,
            default_options.experimental_flags);
  EXPECT_EQ(best_options.max_delegated_partitions,
            default_options.max_delegated_partitions);
}

TEST(GPUCompatibility, RecogniseValidCompatibilityListFlatbuffer) {
  EXPECT_TRUE(tflite::acceleration::GPUCompatibilityList::IsValidFlatbuffer(
      g_tflite_acceleration_devicedb_sample_binary,
      g_tflite_acceleration_devicedb_sample_binary_len));
}

TEST(GPUCompatibility, RecogniseInvalidCompatibilityListFlatbuffer) {
  unsigned char invalid_buffer[100];
  std::fill(invalid_buffer, invalid_buffer + 100, ' ');
  EXPECT_FALSE(tflite::acceleration::GPUCompatibilityList::IsValidFlatbuffer(
      invalid_buffer, 100));
}

TEST(GPUCompatibility, CreationWithInvalidCompatibilityListFlatbuffer) {
  unsigned char invalid_buffer[10];
  std::fill(invalid_buffer, invalid_buffer + 10, ' ');
  std::unique_ptr<tflite::acceleration::GPUCompatibilityList> list =
      tflite::acceleration::GPUCompatibilityList::Create(invalid_buffer, 10);
  EXPECT_EQ(list, nullptr);
}

TEST(GPUCompatibility, CreationWithNullCompatibilityListFlatbuffer) {
  std::unique_ptr<tflite::acceleration::GPUCompatibilityList> list =
      tflite::acceleration::GPUCompatibilityList::Create(nullptr, 0);
  EXPECT_EQ(list, nullptr);
}

TEST(GPUCompatibility, ConvertCompatibilityStatusToStringCorrectly) {
  EXPECT_EQ(
      tflite::acceleration::GPUCompatibilityList::CompatibilityStatusToString(
          tflite::acceleration::gpu::CompatibilityStatus::kSupported),
      tflite::acceleration::gpu::kStatusSupported);
  EXPECT_EQ(
      tflite::acceleration::GPUCompatibilityList::CompatibilityStatusToString(
          tflite::acceleration::gpu::CompatibilityStatus::kUnsupported),
      tflite::acceleration::gpu::kStatusUnsupported);
  EXPECT_EQ(
      tflite::acceleration::GPUCompatibilityList::CompatibilityStatusToString(
          tflite::acceleration::gpu::CompatibilityStatus::kUnknown),
      tflite::acceleration::gpu::kStatusUnknown);
}

TEST(GPUCompatibility, ConvertStringToCompatibilityStatusCorrectly) {
  EXPECT_EQ(
      tflite::acceleration::GPUCompatibilityList::StringToCompatibilityStatus(
          tflite::acceleration::gpu::kStatusSupported),
      tflite::acceleration::gpu::CompatibilityStatus::kSupported);
  EXPECT_EQ(
      tflite::acceleration::GPUCompatibilityList::StringToCompatibilityStatus(
          tflite::acceleration::gpu::kStatusUnsupported),
      tflite::acceleration::gpu::CompatibilityStatus::kUnsupported);
  EXPECT_EQ(
      tflite::acceleration::GPUCompatibilityList::StringToCompatibilityStatus(
          tflite::acceleration::gpu::kStatusUnknown),
      tflite::acceleration::gpu::CompatibilityStatus::kUnknown);
}

}  // namespace
