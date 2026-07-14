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

#include "tensorflow/lite/delegates/gpu/gl/kernels/pooling.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/test_util.h"

MATCHER(IntEq, "") {
  return *reinterpret_cast<const int*>(&std::get<0>(arg)) == std::get<1>(arg);
}

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace gl {
namespace {

TEST(PoolingTest, MaxKernel2x2Stride2x2WithIndices) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 4, 4, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> indices;
  indices.type = DataType::INT32;
  indices.ref = 2;
  indices.shape = BHWC(1, 2, 2, 1);

  Pooling2DAttributes attr;
  attr.kernel = HW(2, 2);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.type = PoolingType::MAX;
  attr.output_indices = true;

  SingleOpModel model({ToString(OperationType::POOLING_2D), attr}, {input},
                      {output, indices});
  ASSERT_TRUE(model.PopulateTensor(
      0, {1, 2, 1, 2, 3, 4, 3, 4, 7, 8, 7, 8, 5, 6, 5, 6}));
  ASSERT_OK(model.Invoke(*NewPoolingNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {4, 4, 8, 8}));
  // Indices tensor is a vector<float>, but these float values should be treated
  // as integers, that's why special matcher IntNear() is used.
  EXPECT_THAT(model.GetOutput(1), Pointwise(IntEq(), {3, 3, 1, 1}));
}

TEST(PoolingTest, MaxKernel2x2Stride2x2WithoutIndices) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 4, 4, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  Pooling2DAttributes attr;
  attr.kernel = HW(2, 2);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.type = PoolingType::MAX;

  SingleOpModel model({ToString(OperationType::POOLING_2D), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(
      0, {1, 2, 1, 2, 3, 4, 3, 4, 7, 8, 7, 8, 5, 6, 5, 6}));
  ASSERT_OK(model.Invoke(*NewPoolingNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {4, 4, 8, 8}));
}

TEST(PoolingTest, AverageKernel2x2Stride2x2) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 4, 4, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  Pooling2DAttributes attr;
  attr.kernel = HW(2, 2);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);
  attr.type = PoolingType::AVERAGE;

  SingleOpModel model({ToString(OperationType::POOLING_2D), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(
      0, {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}));
  ASSERT_OK(model.Invoke(*NewPoolingNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {1, 2, 3, 4}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
