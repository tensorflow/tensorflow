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

#include "tensorflow/lite/delegates/gpu/cl/kernels/lstm.h"

#include <cmath>
#include <cstdlib>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/cl/kernels/cl_test.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace cl {
namespace {

TEST_F(OpenCLOperationTest, LSTM) {
  TensorFloat32 src_tensor;
  src_tensor.shape = BHWC(1, 1, 1, 16);
  src_tensor.data = {
      -std::log(2.0f), -std::log(2.0f), -std::log(2.0f), -std::log(2.0f),
      std::log(3.0f),  std::log(3.0f),  std::log(3.0f),  std::log(3.0f),
      -std::log(4.0f), -std::log(4.0f), -std::log(4.0f), -std::log(4.0f),
      -std::log(5.0f), -std::log(5.0f), -std::log(5.0f), -std::log(5.0f)};
  // input_gate = 1.0 / (1.0 + exp(log(2.0f))) = 1.0 / 3.0;
  // new_input = tanh(log(3.0f)) = (exp(2 * log(3.0f)) - 1) / exp(2 * log(3.0f))
  // + 1 = (9 - 1) / (9 + 1) = 0.8;
  // forget_gate = 1.0 / (1.0 + exp(log(4.0f)))
  //  = 1.0 / 5.0;
  // output_gate = 1.0 / (1.0 + exp(log(5.0f))) = 1.0 / 6.0;
  // new_st = input_gate * new_input + forget_gate * prev_st
  //   = 1.0 / 3.0 * 0.8 + 1.0 / 5.0 * prev_st
  //   = 4.0 / 15.0 + 3.0 / 15.0 = 7.0 / 15.0
  // activation = output_gate * tanh(new_st)
  TensorFloat32 prev_state;
  prev_state.shape = BHWC(1, 1, 1, 4);
  prev_state.data = {1.0f, 2.0f, 3.0f, 4.0f};

  for (auto storage : env_.GetSupportedStorages()) {
    for (auto precision : env_.GetSupportedPrecisions()) {
      const float eps = precision == CalculationsPrecision::F32 ? 1e-6f : 1e-3f;
      OperationDef op_def;
      op_def.precision = precision;
      auto data_type = DeduceDataTypeFromPrecision(precision);
      op_def.src_tensors.push_back({data_type, storage});
      op_def.src_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      op_def.dst_tensors.push_back({data_type, storage});
      TensorFloat32 new_state;
      TensorFloat32 new_activ;
      LSTM operation = CreateLSTM(op_def);
      ASSERT_OK(ExecuteGPUOperation(
          {src_tensor, prev_state}, creation_context_, &operation,
          {BHWC(1, 1, 1, 4), BHWC(1, 1, 1, 4)}, {&new_state, &new_activ}));
      EXPECT_THAT(new_state.data,
                  Pointwise(FloatNear(eps), {7.0 / 15.0, 10.0 / 15.0,
                                             13.0 / 15.0, 16.0 / 15.0}));
      EXPECT_THAT(
          new_activ.data,
          Pointwise(FloatNear(eps), {(1.0 / 6.0) * std::tanh(7.0 / 15.0),
                                     (1.0 / 6.0) * std::tanh(10.0 / 15.0),
                                     (1.0 / 6.0) * std::tanh(13.0 / 15.0),
                                     (1.0 / 6.0) * std::tanh(16.0 / 15.0)}));
    }
  }
}

}  // namespace
}  // namespace cl
}  // namespace gpu
}  // namespace tflite
