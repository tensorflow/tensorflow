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

TEST(LstmTest, Input2x2x1) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> prev_state;
  prev_state.type = DataType::FLOAT32;
  prev_state.ref = 1;
  prev_state.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> output_state;
  output_state.type = DataType::FLOAT32;
  output_state.ref = 2;
  output_state.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> output_activation;
  output_activation.type = DataType::FLOAT32;
  output_activation.ref = 3;
  output_activation.shape = BHWC(1, 2, 2, 1);

  LstmAttributes attr;
  attr.kernel_type = LstmKernelType::BASIC;

  SingleOpModel model({ToString(OperationType::LSTM), attr},
                      {input, prev_state}, {output_state, output_activation});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_TRUE(model.PopulateTensor(1, {5, 6, 7, 8}));
  ASSERT_OK(model.Invoke(*NewLstmNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {2.5, 3.0, 3.5, 4.0}));
  EXPECT_THAT(
      model.GetOutput(1),
      Pointwise(FloatNear(1e-6), {0.493307, 0.497527, 0.499089, 0.499665}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
