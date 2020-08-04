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
#include "tensorflow/lite/experimental/acceleration/compatibility/gpu_compatibility_recommender.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

class GPUCompatibilityRecommenderTest : public ::testing::Test {
 protected:
  GPUCompatibilityRecommenderTest() {
    recommender_ =
        absl::make_unique<tflite::acceleration::GPUCompatibilityRecommender>();
  }

  std::unique_ptr<tflite::acceleration::GPUCompatibilityRecommender>
      recommender_;
};

TEST_F(GPUCompatibilityRecommenderTest, Load) {
  EXPECT_TRUE(recommender_->IsDatabaseLoaded());
}

TEST_F(GPUCompatibilityRecommenderTest, ReturnsSupportedForFullMatch) {
  tflite::acceleration::AndroidInfo android_info = {
      .android_sdk_version = "28",
      .model = "redmi_note_7G960F",
      .device = "lavender",
      .manufacturer = "xiaomi"};
  tflite::gpu::GpuInfo tflite_gpu_info = {
      .renderer_name = "adreno_(tm)_512",
      .major_version = 3,
      .minor_version = 2,
  };
  EXPECT_TRUE(recommender_->Includes(android_info, tflite_gpu_info));
}

TEST_F(GPUCompatibilityRecommenderTest, ReturnsUnsupported) {
  tflite::acceleration::AndroidInfo android_info = {.android_sdk_version = "28",
                                                    .model = "sm_g960f",
                                                    .device = "starlte",
                                                    .manufacturer = "samsung"};
  tflite::gpu::GpuInfo tflite_gpu_info = {
      .renderer_name = "mali_g72",
      .major_version = 3,
      .minor_version = 2,
  };

  EXPECT_FALSE(recommender_->Includes(android_info, tflite_gpu_info));
}

TEST_F(GPUCompatibilityRecommenderTest, MissingInfoReturnsUnsupported) {
  tflite::acceleration::AndroidInfo android_info = {.android_sdk_version = "23",
                                                    .model = "sm_g532f",
                                                    .device = "grandpplte",
                                                    .manufacturer = "samsung"};
  tflite::gpu::GpuInfo tflite_gpu_info = {
      .renderer_name = "mali_t720",
      .major_version = 3,
      .minor_version = 1,
  };
  EXPECT_FALSE(recommender_->Includes(android_info, tflite_gpu_info));
}

TEST_F(GPUCompatibilityRecommenderTest, ReturnsDefaultOptions) {
  tflite::acceleration::AndroidInfo android_info;
  tflite::gpu::GpuInfo tflite_gpu_info;
  auto default_options = TfLiteGpuDelegateOptionsV2Default();
  auto best_options =
      recommender_->GetBestOptionsFor(android_info, tflite_gpu_info);
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

}  // namespace
