/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <array>

#include "tensorflow/contrib/lite/experimental/c/c_api.h"

#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/allocation.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/testing/util.h"

namespace {

TEST(CApiSimple, Smoke) {
  tflite::FileCopyAllocation model_file(
      "tensorflow/contrib/lite/testdata/add.bin",
      tflite::DefaultErrorReporter());

  TFL_Interpreter* interpreter =
      TFL_NewInterpreter(model_file.base(), model_file.bytes());
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TFL_InterpreterAllocateTensors(interpreter), kTfLiteOk);

  ASSERT_EQ(TFL_InterpreterGetInputTensorCount(interpreter), 1);
  ASSERT_EQ(TFL_InterpreterGetOutputTensorCount(interpreter), 1);

  std::array<int, 1> input_dims = {2};
  ASSERT_EQ(TFL_InterpreterResizeInputTensor(interpreter, 0, input_dims.data(),
                                             input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TFL_InterpreterAllocateTensors(interpreter), kTfLiteOk);

  TFL_Tensor* input_tensor = TFL_InterpreterGetInputTensor(interpreter, 0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TFL_TensorType(input_tensor), kTfLiteFloat32);
  EXPECT_EQ(TFL_TensorNumDims(input_tensor), 1);
  EXPECT_EQ(TFL_TensorDim(input_tensor, 0), 2);
  EXPECT_EQ(TFL_TensorByteSize(input_tensor), sizeof(float) * 2);

  std::array<float, 2> input = {1.f, 3.f};
  ASSERT_EQ(TFL_TensorCopyFromBuffer(input_tensor, input.data(),
                                     input.size() * sizeof(float)),
            kTfLiteOk);

  ASSERT_EQ(TFL_InterpreterInvoke(interpreter), kTfLiteOk);

  const TFL_Tensor* output_tensor =
      TFL_InterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(TFL_TensorType(output_tensor), kTfLiteFloat32);
  EXPECT_EQ(TFL_TensorNumDims(output_tensor), 1);
  EXPECT_EQ(TFL_TensorDim(output_tensor, 0), 2);
  EXPECT_EQ(TFL_TensorByteSize(output_tensor), sizeof(float) * 2);

  std::array<float, 2> output;
  ASSERT_EQ(TFL_TensorCopyToBuffer(output_tensor, output.data(),
                                   output.size() * sizeof(float)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3.f);
  EXPECT_EQ(output[1], 9.f);

  TFL_DeleteInterpreter(interpreter);
}

}  // namespace

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
