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

#include "tensorflow/lite/delegates/gpu/gl/kernels/lstm.h"

#include <cmath>
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

TEST(LstmTest, BaseTest) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 16);

  TensorRef<BHWC> prev_state;
  prev_state.type = DataType::FLOAT32;
  prev_state.ref = 1;
  prev_state.shape = BHWC(1, 1, 1, 4);

  TensorRef<BHWC> output_state;
  output_state.type = DataType::FLOAT32;
  output_state.ref = 2;
  output_state.shape = BHWC(1, 1, 1, 4);

  TensorRef<BHWC> output_activation;
  output_activation.type = DataType::FLOAT32;
  output_activation.ref = 3;
  output_activation.shape = BHWC(1, 1, 1, 4);

  LstmAttributes attr;
  attr.kernel_type = LstmKernelType::BASIC;

  SingleOpModel model({ToString(OperationType::LSTM), attr},
                      {input, prev_state}, {output_state, output_activation});
  std::vector input_data = {
      -std::log(2.0f), -std::log(2.0f), -std::log(2.0f), -std::log(2.0f),
      std::log(3.0f),  std::log(3.0f),  std::log(3.0f),  std::log(3.0f),
      -std::log(4.0f), -std::log(4.0f), -std::log(4.0f), -std::log(4.0f),
      -std::log(5.0f), -std::log(5.0f), -std::log(5.0f), -std::log(5.0f)};
  ASSERT_TRUE(model.PopulateTensor(0, std::move(input_data)));
  ASSERT_TRUE(model.PopulateTensor(1, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewLstmNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6),
                        {7.0 / 15.0, 10.0 / 15.0, 13.0 / 15.0, 16.0 / 15.0}));
  EXPECT_THAT(
      model.GetOutput(1),
      Pointwise(FloatNear(1e-6), {(1.f / 6.f) * std::tanh(7.f / 15.f),
                                  (1.f / 6.f) * std::tanh(10.f / 15.f),
                                  (1.f / 6.f) * std::tanh(13.f / 15.f),
                                  (1.f / 6.f) * std::tanh(16.f / 15.f)}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
