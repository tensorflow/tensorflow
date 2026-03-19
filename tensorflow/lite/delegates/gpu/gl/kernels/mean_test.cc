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

#include "tensorflow/lite/delegates/gpu/gl/kernels/mean.h"

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

TEST(MeanTest, TestTrivialImpl) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 1, 1, 1);

  MeanAttributes attr;
  attr.dims = {Axis::HEIGHT, Axis::WIDTH};

  SingleOpModel model({ToString(OperationType::MEAN), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 2.0, 3.0, 4.0}));
  ASSERT_OK(model.Invoke(*NewMeanNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {2.5}));
}

TEST(MeanTest, TestTiledImpl) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 16, 16, 8);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 1, 1, 8);

  MeanAttributes attr;
  attr.dims = {Axis::HEIGHT, Axis::WIDTH};

  SingleOpModel model({ToString(OperationType::MEAN), attr}, {input}, {output});
  std::vector<float> input_data;
  input_data.reserve(1 * 16 * 16 * 8);
  for (int i = 0; i < 1 * 16 * 16 * 8; ++i) input_data.push_back(i % 8);
  ASSERT_TRUE(model.PopulateTensor(0, std::move(input_data)));
  ASSERT_OK(model.Invoke(*NewMeanNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0, 1, 2, 3, 4, 5, 6, 7}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
