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
#include <stdint.h>

#include <memory>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

namespace tflite {
namespace {
const char kContainer[] = "c";
const char kSharedName[] = "a";

class VariableOpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    assign_registration_ = ::tflite::ops::builtin::Register_ASSIGN_VARIABLE();
    ASSERT_NE(assign_registration_, nullptr);
    read_registration_ = ::tflite::ops::builtin::Register_READ_VARIABLE();
    ASSERT_NE(read_registration_, nullptr);
    var_handle_registration_ = ::tflite::ops::builtin::Register_VAR_HANDLE();
    ASSERT_NE(var_handle_registration_, nullptr);

    ConstructGraph();
  }

  void ConstructInvalidGraph() {
    interpreter_ = std::make_unique<Interpreter>();
    // Invalid graph, variable is read before it is assigned a value.

    // Construct a graph like this:
    //   Input: %0
    //   Output: %2
    //   %1 = var_handle()
    //   %2 = read(%1)

    TfLiteVarHandleParams* var_handle_params = GetVarHandleParams();

    int first_new_tensor_index;
    ASSERT_EQ(interpreter_->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetOutputs({2}), kTfLiteOk);
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteResource, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", 0,
                                               nullptr, {}, false);
    int node_index;
    interpreter_->AddNodeWithParameters({}, {1}, nullptr, 0, var_handle_params,
                                        var_handle_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr,
                                        read_registration_, &node_index);
  }

  TfLiteVarHandleParams* GetVarHandleParams() {
    TfLiteVarHandleParams* var_handle_params =
        reinterpret_cast<TfLiteVarHandleParams*>(
            malloc(sizeof(TfLiteVarHandleParams)));
    var_handle_params->container = kContainer;
    var_handle_params->shared_name = kSharedName;
    return var_handle_params;
  }

  void ConstructGraph() {
    interpreter_ = std::make_unique<Interpreter>();
    // Construct a graph like this:
    //   Input: %0
    //   Output: %2
    //   %1 = var_handle()
    //   variable_assign(%1, %0)
    //   %2 = read(%1)

    int first_new_tensor_index;
    ASSERT_EQ(interpreter_->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetOutputs({2}), kTfLiteOk);
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteResource, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", 0,
                                               nullptr, {}, false);
    int node_index;

    TfLiteVarHandleParams* var_handle_params = GetVarHandleParams();
    interpreter_->AddNodeWithParameters({}, {1}, nullptr, 0, var_handle_params,
                                        var_handle_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1, 0}, {}, nullptr, 0, nullptr,
                                        assign_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr,
                                        read_registration_, &node_index);
  }

  // Similar with `ConstructGraph`, but with static tensor shapes.
  void ConstructGraphWithKnownShape() {
    interpreter_ = std::make_unique<Interpreter>();
    // Construct a graph like this:
    //   Input: %0
    //   Output: %2
    //   %1 = var_handle()
    //   variable_assign(%1, %0)
    //   %2 = read(%1)

    int first_new_tensor_index;
    ASSERT_EQ(interpreter_->AddTensors(3, &first_new_tensor_index), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetInputs({0}), kTfLiteOk);
    ASSERT_EQ(interpreter_->SetOutputs({2}), kTfLiteOk);
    interpreter_->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {2, 2},
                                               TfLiteQuantization());
    interpreter_->SetTensorParametersReadWrite(1, kTfLiteResource, "", 0,
                                               nullptr, {}, false);
    interpreter_->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {2, 2},
                                               TfLiteQuantization());
    int node_index;

    TfLiteVarHandleParams* var_handle_params = GetVarHandleParams();
    interpreter_->AddNodeWithParameters({}, {1}, nullptr, 0, var_handle_params,
                                        var_handle_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1, 0}, {}, nullptr, 0, nullptr,
                                        assign_registration_, &node_index);
    interpreter_->AddNodeWithParameters({1}, {2}, nullptr, 0, nullptr,
                                        read_registration_, &node_index);
  }

  TfLiteRegistration* assign_registration_;
  TfLiteRegistration* read_registration_;
  TfLiteRegistration* var_handle_registration_;
  std::unique_ptr<Interpreter> interpreter_;
};

TEST_F(VariableOpsTest, TestAssignThenReadVariable) {
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  TfLiteTensor* input_data_index = interpreter_->tensor(0);
  GetTensorData<float>(input_data_index)[0] = 1717;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  // Verify output.
  TfLiteTensor* output = interpreter_->tensor(2);
  ASSERT_EQ(output->dims->size, 0);
  EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
}

TEST_F(VariableOpsTest, TestAssignThenReadVariableWithKnownShape) {
  ConstructGraphWithKnownShape();
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  TfLiteTensor* input_data_index = interpreter_->tensor(0);
  GetTensorData<float>(input_data_index)[0] = 1.0;
  GetTensorData<float>(input_data_index)[1] = 2.0;
  GetTensorData<float>(input_data_index)[2] = 3.0;
  GetTensorData<float>(input_data_index)[3] = 4.0;
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

  // Verify output.
  TfLiteTensor* output = interpreter_->tensor(2);
  ASSERT_EQ(output->dims->size, 2);
  EXPECT_EQ(GetTensorData<float>(output)[0], 1.0);
  EXPECT_EQ(GetTensorData<float>(output)[1], 2.0);
  EXPECT_EQ(GetTensorData<float>(output)[2], 3.0);
  EXPECT_EQ(GetTensorData<float>(output)[3], 4.0);
}

TEST_F(VariableOpsTest, TestReadVariableBeforeAssign) {
  ConstructInvalidGraph();
  ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);
  TfLiteTensor* input_data_index = interpreter_->tensor(0);
  GetTensorData<float>(input_data_index)[0] = 1717;

  // Error because variable 2 is never initialized.
  ASSERT_EQ(interpreter_->Invoke(), kTfLiteError);
}

TEST_F(VariableOpsTest, TestReassignToDifferentSize) {
  // 1st invocation. The variable is assigned as a scalar.
  {
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    TfLiteTensor* input_data_index = interpreter_->tensor(0);
    GetTensorData<float>(input_data_index)[0] = 1717;
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

    // Verify output.
    TfLiteTensor* output = interpreter_->tensor(2);
    ASSERT_EQ(output->dims->size, 0);
    EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
  }

  // 2nd invocation. The variable is assigned as a 1D vector with 2 elements.
  {
    interpreter_->ResizeInputTensor(0, {2});
    ASSERT_EQ(interpreter_->AllocateTensors(), kTfLiteOk);

    TfLiteTensor* input_data_index = interpreter_->tensor(0);
    GetTensorData<float>(input_data_index)[0] = 1717;
    GetTensorData<float>(input_data_index)[1] = 2121;
    ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk);

    // Verify output.
    TfLiteTensor* output = interpreter_->tensor(2);
    ASSERT_EQ(output->dims->size, 1);
    ASSERT_EQ(output->dims->data[0], 2);
    EXPECT_EQ(GetTensorData<float>(output)[0], 1717);
    EXPECT_EQ(GetTensorData<float>(output)[1], 2121);
  }
}

}  // namespace
}  // namespace tflite
