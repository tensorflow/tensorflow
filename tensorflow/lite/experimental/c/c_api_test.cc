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

#include "tensorflow/lite/experimental/c/c_api.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/testing/util.h"

namespace {

TEST(CAPI, Version) { EXPECT_STRNE("", TFL_Version()); }

TEST(CApiSimple, Smoke) {
  TFL_Model* model = TFL_NewModelFromFile(
      "tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TFL_InterpreterOptions* options = TFL_NewInterpreterOptions();
  ASSERT_NE(options, nullptr);
  TFL_InterpreterOptionsSetNumThreads(options, 2);

  TFL_Interpreter* interpreter = TFL_NewInterpreter(model, options);
  ASSERT_NE(interpreter, nullptr);

  // The options/model can be deleted immediately after interpreter creation.
  TFL_DeleteInterpreterOptions(options);
  TFL_DeleteModel(model);

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
  EXPECT_NE(TFL_TensorData(input_tensor), nullptr);
  EXPECT_STREQ(TFL_TensorName(input_tensor), "input");

  TFL_QuantizationParams input_params =
      TFL_TensorQuantizationParams(input_tensor);
  EXPECT_EQ(input_params.scale, 0.f);
  EXPECT_EQ(input_params.zero_point, 0);

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
  EXPECT_NE(TFL_TensorData(output_tensor), nullptr);
  EXPECT_STREQ(TFL_TensorName(output_tensor), "output");

  TFL_QuantizationParams output_params =
      TFL_TensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.f);
  EXPECT_EQ(output_params.zero_point, 0);

  std::array<float, 2> output;
  ASSERT_EQ(TFL_TensorCopyToBuffer(output_tensor, output.data(),
                                   output.size() * sizeof(float)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3.f);
  EXPECT_EQ(output[1], 9.f);

  TFL_DeleteInterpreter(interpreter);
}

TEST(CApiSimple, QuantizationParams) {
  TFL_Model* model = TFL_NewModelFromFile(
      "tensorflow/lite/testdata/add_quantized.bin");
  ASSERT_NE(model, nullptr);

  TFL_Interpreter* interpreter = TFL_NewInterpreter(model, nullptr);
  ASSERT_NE(interpreter, nullptr);

  TFL_DeleteModel(model);

  const std::array<int, 1> input_dims = {2};
  ASSERT_EQ(TFL_InterpreterResizeInputTensor(interpreter, 0, input_dims.data(),
                                             input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TFL_InterpreterAllocateTensors(interpreter), kTfLiteOk);

  TFL_Tensor* input_tensor = TFL_InterpreterGetInputTensor(interpreter, 0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TFL_TensorType(input_tensor), kTfLiteUInt8);
  EXPECT_EQ(TFL_TensorNumDims(input_tensor), 1);
  EXPECT_EQ(TFL_TensorDim(input_tensor, 0), 2);

  TFL_QuantizationParams input_params =
      TFL_TensorQuantizationParams(input_tensor);
  EXPECT_EQ(input_params.scale, 0.003922f);
  EXPECT_EQ(input_params.zero_point, 0);

  const std::array<uint8_t, 2> input = {1, 3};
  ASSERT_EQ(TFL_TensorCopyFromBuffer(input_tensor, input.data(),
                                     input.size() * sizeof(uint8_t)),
            kTfLiteOk);

  ASSERT_EQ(TFL_InterpreterInvoke(interpreter), kTfLiteOk);

  const TFL_Tensor* output_tensor =
      TFL_InterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);

  TFL_QuantizationParams output_params =
      TFL_TensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.003922f);
  EXPECT_EQ(output_params.zero_point, 0);

  std::array<uint8_t, 2> output;
  ASSERT_EQ(TFL_TensorCopyToBuffer(output_tensor, output.data(),
                                   output.size() * sizeof(uint8_t)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3);
  EXPECT_EQ(output[1], 9);

  const float dequantizedOutput0 =
      output_params.scale * (output[0] - output_params.zero_point);
  const float dequantizedOutput1 =
      output_params.scale * (output[1] - output_params.zero_point);
  EXPECT_EQ(dequantizedOutput0, 0.011766f);
  EXPECT_EQ(dequantizedOutput1, 0.035298f);

  TFL_DeleteInterpreter(interpreter);
}

TEST(CApiSimple, ErrorReporter) {
  TFL_Model* model = TFL_NewModelFromFile(
      "tensorflow/lite/testdata/add.bin");
  TFL_InterpreterOptions* options = TFL_NewInterpreterOptions();

  // Install a custom error reporter into the interpreter by way of options.
  tflite::TestErrorReporter reporter;
  TFL_InterpreterOptionsSetErrorReporter(
      options,
      [](void* user_data, const char* format, va_list args) {
        reinterpret_cast<tflite::TestErrorReporter*>(user_data)->Report(format,
                                                                        args);
      },
      &reporter);
  TFL_Interpreter* interpreter = TFL_NewInterpreter(model, options);

  // The options/model can be deleted immediately after interpreter creation.
  TFL_DeleteInterpreterOptions(options);
  TFL_DeleteModel(model);

  // Invoke the interpreter before tensor allocation.
  EXPECT_EQ(TFL_InterpreterInvoke(interpreter), kTfLiteError);

  // The error should propagate to the custom error reporter.
  EXPECT_EQ(reporter.error_messages(),
            "Invoke called on model that is not ready.");
  EXPECT_EQ(reporter.num_calls(), 1);

  TFL_DeleteInterpreter(interpreter);
}

}  // namespace

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
