/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/kernels/resampler.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/test_util.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace gl {
namespace {

absl::Status ResamplerIdentityTest(const BHWC& shape) {
  TensorRef<BHWC> src_tensor;
  src_tensor.type = DataType::FLOAT32;
  src_tensor.ref = 0;
  src_tensor.shape = shape;

  TensorRef<BHWC> warp_tensor;
  warp_tensor.type = DataType::FLOAT32;
  warp_tensor.ref = 1;
  warp_tensor.shape = BHWC(1, shape.h, shape.w, 2);

  TensorRef<BHWC> dst_tensor;
  dst_tensor.type = DataType::FLOAT32;
  dst_tensor.ref = 2;
  dst_tensor.shape = shape;

  SingleOpModel model({ToString(OperationType::RESAMPLER)},
                      {src_tensor, warp_tensor}, {dst_tensor});
  std::vector<float> src_data(src_tensor.shape.DimensionsProduct());
  std::vector<float> warp_data(warp_tensor.shape.DimensionsProduct());
  std::vector<float> dst_data(dst_tensor.shape.DimensionsProduct());
  for (int i = 0; i < src_data.size(); ++i) {
    src_data[i] = std::sin(i);
    dst_data[i] = src_data[i];
  }
  for (int y = 0; y < shape.h; ++y) {
    for (int x = 0; x < shape.w; ++x) {
      warp_data[(y * shape.w + x) * 2 + 0] = x;
      warp_data[(y * shape.w + x) * 2 + 1] = y;
    }
  }
  if (!model.PopulateTensor(0, std::move(src_data))) {
    return absl::InternalError("failed loading data");
  }
  if (!model.PopulateTensor(1, std::move(warp_data))) {
    return absl::InternalError("failed loading data");
  }
  RETURN_IF_ERROR(model.Invoke(*NewResamplerNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), dst_data));
  return absl::OkStatus();
}

TEST(ResamplerTest, Identity_2_2_1) {
  auto status = ResamplerIdentityTest(BHWC(1, 2, 2, 1));
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST(ResamplerTest, Identity_3_5_3) {
  auto status = ResamplerIdentityTest(BHWC(1, 3, 5, 3));
  ASSERT_TRUE(status.ok()) << status.message();
}

TEST(ResamplerTest, Identity_6_1_7) {
  auto status = ResamplerIdentityTest(BHWC(1, 6, 1, 7));
  ASSERT_TRUE(status.ok()) << status.message();
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
