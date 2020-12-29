/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/texture2d.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLTest, Texture2DTestFloat) {
  const std::vector<float> data = {1.0, 2.0, 3.0, -4.0, 5.1, 6.7, 4.1, 6.17};
  Texture2D texture;
  ASSERT_OK(CreateTexture2DRGBA32F(1, 2, &env_.context(), &texture));
  ASSERT_OK(texture.WriteData(env_.queue(),
                              absl::MakeConstSpan(data.data(), data.size())));
  std::vector<float> gpu_data;
  ASSERT_OK(texture.ReadData<float>(env_.queue(), &gpu_data));

  EXPECT_THAT(gpu_data, Pointwise(FloatNear(0.0f), data));
}

TEST_F(OpenCLTest, Texture2DTestHalf) {
  const std::vector<half> data = {half(1.4),  half(2.1),  half(2.2), half(1.34),
                                  half(20.1), half(2.24), half(0.1), half(0.2)};
  Texture2D texture;
  ASSERT_OK(CreateTexture2DRGBA16F(2, 1, &env_.context(), &texture));
  ASSERT_OK(texture.WriteData(env_.queue(),
                              absl::MakeConstSpan(data.data(), data.size())));
  std::vector<half> gpu_data;
  ASSERT_OK(texture.ReadData<half>(env_.queue(), &gpu_data));

  EXPECT_THAT(gpu_data, Pointwise(FloatNear(0.0f), data));
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
