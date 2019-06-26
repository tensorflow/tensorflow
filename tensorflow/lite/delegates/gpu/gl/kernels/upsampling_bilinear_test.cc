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

#include "tensorflow/lite/delegates/gpu/gl/kernels/upsampling_bilinear.h"

#include <vector>

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

TEST(UpsamplingBilinearTest, 1x1x2To2x2x2) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 2);

  Upsample2DAttributes attr;
  attr.align_corners = true;
  attr.new_shape = HW(2, 2);
  attr.type = UpsamplingType::BILINEAR;

  SingleOpModel model({ToString(OperationType::UPSAMPLE_2D), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 2.0}));
  ASSERT_OK(model.Invoke(*NewUpsamplingNodeShader()));
  EXPECT_THAT(
      model.GetOutput(0),
      Pointwise(FloatNear(1e-6), {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0}));
}

TEST(UpsamplingBilinearTest, 1x2x1To1x4x1) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 1, 4, 1);

  Upsample2DAttributes attr;
  attr.align_corners = false;
  attr.new_shape = HW(1, 4);
  attr.type = UpsamplingType::BILINEAR;

  SingleOpModel model({ToString(OperationType::UPSAMPLE_2D), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 4.0}));
  ASSERT_OK(model.Invoke(*NewUpsamplingNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0, 2.5, 4.0, 4.0}));
}

TEST(UpsamplingBilinearTest, 2x2x1To4x4x1) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 4, 4, 1);

  Upsample2DAttributes attr;
  attr.align_corners = false;
  attr.new_shape = HW(4, 4);
  attr.type = UpsamplingType::BILINEAR;

  SingleOpModel model({ToString(OperationType::UPSAMPLE_2D), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 4.0, 6.0, 8.0}));
  ASSERT_OK(model.Invoke(*NewUpsamplingNodeShader()));
  EXPECT_THAT(
      model.GetOutput(0),
      Pointwise(FloatNear(1e-6), {1.0, 2.5, 4.0, 4.0, 3.5, 4.75, 6.0, 6.0, 6.0,
                                  7.0, 8.0, 8.0, 6.0, 7.0, 8.0, 8.0}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
