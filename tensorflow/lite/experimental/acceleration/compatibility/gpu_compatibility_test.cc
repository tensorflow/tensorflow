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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

class GPUCompatibilityTest : public ::testing::Test {
 protected:
  GPUCompatibilityTest() {
    list_ = absl::make_unique<tflite::acceleration::GPUCompatibilityList>();
  }

  std::unique_ptr<tflite::acceleration::GPUCompatibilityList> list_;
};

TEST_F(GPUCompatibilityTest, Load) { EXPECT_TRUE(list_->IsDatabaseLoaded()); }

TEST_F(GPUCompatibilityTest, ReturnsSupportedForFullMatch) {
  tflite::acceleration::AndroidInfo android_info = {.android_sdk_version = "27",
                                                    .model = "cph1803",
                                                    .device = "cph1803",
                                                    .manufacturer = "Oppo"};
  tflite::gpu::GpuInfo tflite_gpu_info = {
      .renderer_name = "Adreno (TM) 506",
      .major_version = 3,
      .minor_version = 2,
  };
  EXPECT_TRUE(list_->Includes(android_info, tflite_gpu_info));
}

TEST_F(GPUCompatibilityTest, ReturnsUnsupportedForFullMatch) {
  tflite::acceleration::AndroidInfo android_info = {.android_sdk_version = "28",
                                                    .model = "SM-G960F",
                                                    .device = "starlte",
                                                    .manufacturer = "Samsung"};
  tflite::gpu::GpuInfo tflite_gpu_info = {
      .renderer_name = "Mali-G72",
      .major_version = 3,
      .minor_version = 2,
  };
  EXPECT_FALSE(list_->Includes(android_info, tflite_gpu_info));
}

}  // namespace
