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
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {

// Forward declaration for op kernels.
namespace ops {
namespace custom {

TfLiteRegistration* Register_ASSIGN_VARIABLE();
TfLiteRegistration* Register_READ_VARIABLE();

}  // namespace custom
}  // namespace ops

namespace {

class VariableOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    assign_registration_ = ::tflite::ops::custom::Register_ASSIGN_VARIABLE();
    ASSERT_NE(assign_registration_, nullptr);
    read_registration_ = ::tflite::ops::custom::Register_READ_VARIABLE();
    ASSERT_NE(read_registration_, nullptr);

    ConstructGraph();
  }

  void ConstructGraph() {
    // Construct a graph like ths:
    //   Input: %0, %1, %2
    //   Output: %3
    //   variable_assign(%0, %2)
    //   %3 = read(%1)

    int first_new_tensor_index;
    ASSERT_EQ(interpreter_.AddTensors(4, &first_new_tensor_index), kTfLiteOk);
    ASSERT_EQ(interpreter_.SetInputs({0, 1, 2}), kTfLiteOk);
    ASSERT_EQ(interpreter_.SetOutputs({3}), kTfLiteOk);
    interpreter_.SetTensorParametersReadWrite(0, kTfLiteInt32, "", 0, nullptr,
                                              {}, false);
    interpreter_.SetTensorParametersReadWrite(1, kTfLiteInt32, "", 0, nullptr,
                                              {}, false);
    interpreter_.SetTensorParametersReadWrite(2, kTfLiteFloat32, "", 0, nullptr,
                                              {}, false);
    interpreter_.SetTensorParametersReadWrite(3, kTfLiteFloat32, "", 0, nullptr,
                                              {}, false);
    int node_index;
    interpreter_.AddNodeWithParameters({0, 2}, {}, nullptr, 0, nullptr,
                                       assign_registration_, &node_index);
    interpreter_.AddNodeWithParameters({1}, {3}, nullptr, 0, nullptr,
                                       read_registration_, &node_index);
  }
  TfLiteRegistration* assign_registration_;
  TfLiteRegistration* read_registration_;
  Interpreter interpreter_;
};

TEST_F(VariableOpsTest, TestAssignThenReadVariable) {
  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);
  TfLiteTensor* input_assign_index = interpreter_.tensor(0);
  input_assign_index->data.i32[0] = 1;
  TfLiteTensor* input_read_index = interpreter_.tensor(1);
  input_read_index->data.i32[0] = 1;
  TfLiteTensor* input_data_index = interpreter_.tensor(2);
  GetTensorData<float>(input_data_index)[0] = 1717;
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

  // Verify output.
  TfLiteTensor* output = interpreter_.tensor(3);
  ASSERT_EQ(output->dims->size, 0);
  EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
}

TEST_F(VariableOpsTest, TestReadVariableBeforeAssign) {
  ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);
  TfLiteTensor* input_assign_index = interpreter_.tensor(0);
  input_assign_index->data.i32[0] = 1;
  TfLiteTensor* input_read_index = interpreter_.tensor(1);
  input_read_index->data.i32[0] = 2;
  TfLiteTensor* input_data_index = interpreter_.tensor(2);
  GetTensorData<float>(input_data_index)[0] = 1717;

  // Error because variable 2 is never initialized.
  ASSERT_EQ(interpreter_.Invoke(), kTfLiteError);
}

TEST_F(VariableOpsTest, TestReassignToDifferentSize) {
  // 1st invocation. The variable is assigned as a scalar.
  {
    ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

    TfLiteTensor* input_assign_index = interpreter_.tensor(0);
    input_assign_index->data.i32[0] = 1;
    TfLiteTensor* input_read_index = interpreter_.tensor(1);
    input_read_index->data.i32[0] = 1;
    TfLiteTensor* input_data_index = interpreter_.tensor(2);
    GetTensorData<float>(input_data_index)[0] = 1717;
    ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

    // Verify output.
    TfLiteTensor* output = interpreter_.tensor(3);
    ASSERT_EQ(output->dims->size, 0);
    EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
  }

  // 2nd invocation. The variable is assigned as a 1D vector with 2 elements.
  {
    interpreter_.ResizeInputTensor(2, {2});
    ASSERT_EQ(interpreter_.AllocateTensors(), kTfLiteOk);

    TfLiteTensor* input_assign_index = interpreter_.tensor(0);
    input_assign_index->data.i32[0] = 1;
    TfLiteTensor* input_read_index = interpreter_.tensor(1);
    input_read_index->data.i32[0] = 1;
    TfLiteTensor* input_data_index = interpreter_.tensor(2);
    GetTensorData<float>(input_data_index)[0] = 1717;
    GetTensorData<float>(input_data_index)[1] = 2121;
    ASSERT_EQ(interpreter_.Invoke(), kTfLiteOk);

    // Verify output.
    TfLiteTensor* output = interpreter_.tensor(3);
    ASSERT_EQ(output->dims->size, 1);
    ASSERT_EQ(output->dims->data[0], 2);
    EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
    EXPECT_EQ(GetTensorData<float>(output)[1], 2121);
  }
}

}  // namespace
}  // namespace tflite
