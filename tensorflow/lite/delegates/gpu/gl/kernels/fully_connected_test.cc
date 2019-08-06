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

#include "tensorflow/lite/delegates/gpu/gl/kernels/fully_connected.h"

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

TEST(FullyConnectedTest, MatrixByVectorMultiplication) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 2);

  FullyConnectedAttributes attr;

  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 4;
  bias.id = 1;
  bias.data = {1, 2, 3, 4};
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(4, 1, 1, 2);
  weights.id = 2;
  weights.data = {1, 2, 3, 4, 5, 6, 7, 8};
  attr.weights = std::move(weights);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 1, 1, 4);

  SingleOpModel model({ToString(OperationType::FULLY_CONNECTED), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2}));
  ASSERT_OK(model.Invoke(*NewFullyConnectedNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {6, 13, 20, 27}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
