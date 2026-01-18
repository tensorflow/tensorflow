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

#include "tensorflow/lite/delegates/gpu/metal/buffer.h"

#include <vector>

#import <Metal/Metal.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/types.h"

using tflite::gpu::half;

namespace tflite {
namespace gpu {
namespace metal {

TEST(BufferTest, TestBufferF32) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ASSERT_TRUE(device != nil);

  const std::vector<float> data = {1.0f, 2.0f, 3.0f, -4.0f, 5.1f};
  Buffer buffer;
  ASSERT_TRUE(CreateBuffer(sizeof(float) * 5, nullptr, device, &buffer).ok());
  ASSERT_TRUE(buffer.WriteData(absl::MakeConstSpan(data.data(), data.size())).ok());
  std::vector<float> gpu_data;
  ASSERT_TRUE(buffer.ReadData<float>(&gpu_data).ok());

  ASSERT_EQ(gpu_data.size(), data.size());
  for (int i = 0; i < gpu_data.size(); ++i) {
    EXPECT_EQ(gpu_data[i], data[i]);
  }
}

TEST(BufferTest, TestBufferF16) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  ASSERT_TRUE(device != nil);

  const std::vector<half> data = {half(1.0f), half(2.0f), half(3.0f), half(-4.0f), half(5.1f)};
  Buffer buffer;
  ASSERT_TRUE(CreateBuffer(
      sizeof(half) * 5, nullptr, device, &buffer).ok());
  ASSERT_TRUE(buffer.WriteData(absl::MakeConstSpan(data.data(), data.size())).ok());
  std::vector<half> gpu_data;
  ASSERT_TRUE(buffer.ReadData<half>(&gpu_data).ok());

  ASSERT_EQ(gpu_data.size(), data.size());
  for (int i = 0; i < gpu_data.size(); ++i) {
    EXPECT_EQ(gpu_data[i], data[i]);
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
