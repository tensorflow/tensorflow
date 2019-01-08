/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/symmetric_per_channel_params.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // TF:flatbuffers
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace internal {
namespace {

TEST(SymmetricPerChannelParamsTest, TestReadWrite) {
  TensorT tensor;
  std::unique_ptr<SymmetricPerChannelParams> read_back_params;
  const std::vector<float> scales = {3.0, 4.0, 5.0};
  const int channel_dim_index = 42;

  SymmetricPerChannelParams params(scales, channel_dim_index);
  params.AddToTensor(&tensor);
  auto status =
      SymmetricPerChannelParams::ReadFromTensor(tensor, &read_back_params);
  EXPECT_EQ(kTfLiteOk, status);
  ASSERT_TRUE(read_back_params);

  EXPECT_EQ(channel_dim_index, read_back_params->channel_dim_index());
  ASSERT_EQ(read_back_params->scales(), scales);
}

}  // namespace
}  // namespace internal
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
