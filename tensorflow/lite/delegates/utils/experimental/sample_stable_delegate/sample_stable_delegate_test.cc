/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace {

TEST(SampleStableDelegate, StaticallyLinkedDelegateAndModelWithAdd) {
  // Create an instance of the sample stable delegate that implements the ADD
  // operation.
  tflite::TfLiteOpaqueDelegateUniquePtr opaque_delegate =
      tflite::TfLiteOpaqueDelegateFactory::Create(
          std::make_unique<tflite::example::SampleStableDelegate>());
  ASSERT_NE(opaque_delegate, nullptr);

  //
  // Create the model and the interpreter
  //
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate.get());
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);
  // The options can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);

  //
  // Allocate the tensors and fill the input tensor.
  //
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  TfLiteTensor* input_tensor =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/0);
  ASSERT_NE(input_tensor, nullptr);
  const float kTensorCellValue = 3.f;
  int64_t n = tflite::NumElements(input_tensor);
  std::vector<float> input(n, kTensorCellValue);
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(float)),
            kTfLiteOk);

  //
  // Run the interpreter and read the output tensor.
  //
  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);
  std::vector<float> output(n, 0);
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(float)),
            kTfLiteOk);

  // The 'add.bin' model does the following operation ('t_output' denotes the
  // single output tensor, and 't_input' denotes the single input tensor):
  //
  // t_output = t_input + t_input + t_input = t_input * 3
  for (int i = 0; i < output.size(); ++i) {
    EXPECT_EQ(output[i], kTensorCellValue * 3);
  }

  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
}

TEST(SampleStableDelegate, StaticallyLinkedDelegateAndModelWithSub) {
  // Create an instance of the sample stable delegate that implements the SUB
  // operation.
  tflite::TfLiteOpaqueDelegateUniquePtr opaque_delegate =
      tflite::TfLiteOpaqueDelegateFactory::Create(
          std::make_unique<tflite::example::SampleStableDelegate>());
  ASSERT_NE(opaque_delegate, nullptr);

  //
  // Create the model and the interpreter
  //
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/sub.bin");
  ASSERT_NE(model, nullptr);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate.get());
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);
  // The options can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);

  //
  // Allocate the tensors and fill the input tensor.
  //
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  TfLiteTensor* input_tensor_0 =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/0);
  ASSERT_NE(input_tensor_0, nullptr);
  const float kTensor0CellValue = 3.f;
  int64_t n = tflite::NumElements(input_tensor_0);
  std::vector<float> input_0(n, kTensor0CellValue);
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor_0, input_0.data(),
                                       input_0.size() * sizeof(float)),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  TfLiteTensor* input_tensor_1 =
      TfLiteInterpreterGetInputTensor(interpreter, /*input_index=*/1);
  ASSERT_NE(input_tensor_1, nullptr);
  n = tflite::NumElements(input_tensor_1);
  const float kTensor1CellValue = 2.f;
  std::vector<float> input_1(n, kTensor1CellValue);
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor_1, input_1.data(),
                                       input_1.size() * sizeof(float)),
            kTfLiteOk);

  //
  // Run the interpreter and read the output tensor.
  //
  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);
  std::vector<float> output(n, 0);
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(float)),
            kTfLiteOk);

  // The 'sub.bin' model does the following operation ('t_output' denotes the
  // single output tensor, and 't_input_0' and 't_input_1' denote the two input
  // tensors):
  //
  // t_output = t_input_0 - t_input_1
  for (int i = 0; i < output.size(); ++i) {
    EXPECT_EQ(output[i], kTensor0CellValue - kTensor1CellValue);
  }

  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
}

}  // namespace
