/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/c_api_experimental.h"

namespace tflite {
namespace {

TEST(SignatureRunnerTest, TestNoSignatures) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/no_signatures.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreate(model, /*optional_options=*/nullptr);
  ASSERT_NE(interpreter, nullptr);

  int nun_signatures = TfLiteInterpreterGetSignatureCount(interpreter);
  ASSERT_EQ(nun_signatures, 0);

  ASSERT_EQ(TfLiteInterpreterGetSignatureRunner(interpreter, "foo"), nullptr);

  TfLiteSignatureRunner* runner =
      TfLiteInterpreterGetSignatureRunner(interpreter, nullptr);
  ASSERT_NE(runner, nullptr);

  int num_interpreter_inputs =
      TfLiteInterpreterGetInputTensorCount(interpreter);
  int num_runner_inputs = TfLiteSignatureRunnerGetInputCount(runner);
  ASSERT_EQ(num_runner_inputs, num_interpreter_inputs);

  for (int i = 0; i < num_interpreter_inputs; ++i) {
    auto* interpreter_input_tensor =
        TfLiteInterpreterGetInputTensor(interpreter, i);
    ASSERT_NE(interpreter_input_tensor, nullptr);
    auto* interpreter_input_name = TfLiteTensorName(interpreter_input_tensor);
    ASSERT_NE(interpreter_input_name, nullptr);
    auto* runner_input_name = TfLiteSignatureRunnerGetInputName(runner, i);
    ASSERT_NE(runner_input_name, nullptr);
    EXPECT_STREQ(runner_input_name, interpreter_input_name);
    auto* runner_input_tensor =
        TfLiteSignatureRunnerGetInputTensor(runner, interpreter_input_name);
    ASSERT_NE(runner_input_tensor, nullptr);
    ASSERT_EQ(runner_input_tensor, interpreter_input_tensor);
  }

  int num_interpreter_outputs =
      TfLiteInterpreterGetOutputTensorCount(interpreter);
  int num_runner_outputs = TfLiteSignatureRunnerGetOutputCount(runner);
  ASSERT_EQ(num_runner_outputs, num_interpreter_outputs);

  for (int i = 0; i < num_interpreter_outputs; ++i) {
    auto* interpreter_output_tensor =
        TfLiteInterpreterGetOutputTensor(interpreter, i);
    ASSERT_NE(interpreter_output_tensor, nullptr);
    auto* interpreter_output_name = TfLiteTensorName(interpreter_output_tensor);
    ASSERT_NE(interpreter_output_name, nullptr);
    auto* runner_output_name = TfLiteSignatureRunnerGetOutputName(runner, i);
    ASSERT_NE(runner_output_name, nullptr);
    EXPECT_STREQ(runner_output_name, interpreter_output_name);
    auto* runner_output_tensor =
        TfLiteSignatureRunnerGetOutputTensor(runner, interpreter_output_name);
    ASSERT_NE(runner_output_tensor, nullptr);
    ASSERT_EQ(runner_output_tensor, interpreter_output_tensor);
  }

  std::array<int, 1> input_dims{2};
  ASSERT_EQ(TfLiteSignatureRunnerResizeInputTensor(
                runner, "x1", input_dims.data(), input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteSignatureRunnerResizeInputTensor(
                runner, "x2", input_dims.data(), input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteSignatureRunnerAllocateTensors(runner), kTfLiteOk);
  TfLiteTensor* input1 = TfLiteSignatureRunnerGetInputTensor(runner, "x1");
  ASSERT_NE(input1, nullptr);
  TfLiteTensor* input2 = TfLiteSignatureRunnerGetInputTensor(runner, "x2");
  ASSERT_NE(input2, nullptr);
  ASSERT_EQ(TfLiteSignatureRunnerGetInputTensor(runner, "foo"), nullptr);
  const TfLiteTensor* output =
      TfLiteSignatureRunnerGetOutputTensor(runner, "Identity");
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(TfLiteSignatureRunnerGetOutputTensor(runner, "foo"), nullptr);
  input1->data.f[0] = -8;
  input1->data.f[1] = 0.5;
  input2->data.f[0] = -1;
  input2->data.f[1] = 1.5;
  ASSERT_EQ(TfLiteSignatureRunnerInvoke(runner), kTfLiteOk);
  ASSERT_EQ(output->data.f[0], 0);
  ASSERT_EQ(output->data.f[1], 2);

  TfLiteSignatureRunnerDelete(runner);
  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
}

TEST(SignatureRunnerTest, TestMultiSignatures) {
  TfLiteModel* model = TfLiteModelCreateFromFile(
      "tensorflow/lite/testdata/multi_signatures.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  // The options can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);

  std::vector<std::string> signature_defs;
  for (int i = 0; i < TfLiteInterpreterGetSignatureCount(interpreter); i++) {
    signature_defs.push_back(TfLiteInterpreterGetSignatureKey(interpreter, i));
  }
  ASSERT_EQ(signature_defs.size(), 2);
  ASSERT_EQ(signature_defs[0], "add");
  ASSERT_EQ(signature_defs[1], "sub");
  ASSERT_EQ(TfLiteInterpreterGetSignatureRunner(interpreter, "foo"), nullptr);

  TfLiteSignatureRunner* add_runner = TfLiteInterpreterGetSignatureRunner(
      interpreter, signature_defs[0].c_str());
  ASSERT_NE(add_runner, nullptr);
  std::vector<const char*> input_names;
  for (int i = 0; i < TfLiteSignatureRunnerGetInputCount(add_runner); i++) {
    input_names.push_back(TfLiteSignatureRunnerGetInputName(add_runner, i));
  }
  std::vector<const char*> output_names;
  for (int i = 0; i < TfLiteSignatureRunnerGetOutputCount(add_runner); i++) {
    output_names.push_back(TfLiteSignatureRunnerGetOutputName(add_runner, i));
  }
  ASSERT_EQ(input_names.size(), 1);
  ASSERT_EQ(std::string(input_names[0]), "x");
  ASSERT_EQ(output_names.size(), 1);
  ASSERT_EQ(std::string(output_names[0]), "output_0");
  std::array<int, 1> add_runner_input_dims{2};
  ASSERT_EQ(TfLiteSignatureRunnerResizeInputTensor(
                add_runner, "x", add_runner_input_dims.data(),
                add_runner_input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteSignatureRunnerAllocateTensors(add_runner), kTfLiteOk);
  TfLiteTensor* add_input =
      TfLiteSignatureRunnerGetInputTensor(add_runner, "x");
  ASSERT_EQ(TfLiteSignatureRunnerGetInputTensor(add_runner, "foo"), nullptr);
  const TfLiteTensor* add_output =
      TfLiteSignatureRunnerGetOutputTensor(add_runner, "output_0");
  ASSERT_EQ(TfLiteSignatureRunnerGetOutputTensor(add_runner, "foo"), nullptr);
  ASSERT_NE(add_input, nullptr);
  ASSERT_NE(add_output, nullptr);
  add_input->data.f[0] = 2;
  add_input->data.f[1] = 4;
  ASSERT_EQ(TfLiteSignatureRunnerInvoke(add_runner), kTfLiteOk);
  ASSERT_EQ(add_output->data.f[0], 4);
  ASSERT_EQ(add_output->data.f[1], 6);
  TfLiteSignatureRunnerDelete(add_runner);

  TfLiteSignatureRunner* sub_runner =
      TfLiteInterpreterGetSignatureRunner(interpreter, "sub");
  ASSERT_NE(sub_runner, nullptr);
  std::vector<const char*> input_names2;
  for (int i = 0; i < TfLiteSignatureRunnerGetInputCount(sub_runner); i++) {
    input_names2.push_back(TfLiteSignatureRunnerGetInputName(sub_runner, i));
  }
  std::vector<const char*> output_names2;
  for (int i = 0; i < TfLiteSignatureRunnerGetOutputCount(sub_runner); i++) {
    output_names2.push_back(TfLiteSignatureRunnerGetOutputName(sub_runner, i));
  }
  ASSERT_EQ(input_names2.size(), 1);
  ASSERT_EQ(std::string(input_names2[0]), "x");
  ASSERT_EQ(output_names2.size(), 1);
  ASSERT_EQ(std::string(output_names2[0]), "output_0");
  std::array<int, 1> sub_runner_input_dims{3};
  ASSERT_EQ(TfLiteSignatureRunnerResizeInputTensor(
                sub_runner, "x", sub_runner_input_dims.data(),
                sub_runner_input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteSignatureRunnerAllocateTensors(sub_runner), kTfLiteOk);
  TfLiteTensor* sub_input =
      TfLiteSignatureRunnerGetInputTensor(sub_runner, "x");
  ASSERT_EQ(TfLiteSignatureRunnerGetInputTensor(sub_runner, "foo"), nullptr);
  const TfLiteTensor* sub_output =
      TfLiteSignatureRunnerGetOutputTensor(sub_runner, "output_0");
  ASSERT_EQ(TfLiteSignatureRunnerGetOutputTensor(sub_runner, "foo"), nullptr);
  ASSERT_NE(sub_input, nullptr);
  ASSERT_NE(sub_output, nullptr);
  sub_input->data.f[0] = 2;
  sub_input->data.f[1] = 4;
  sub_input->data.f[2] = 6;
  ASSERT_EQ(TfLiteSignatureRunnerInvoke(sub_runner), kTfLiteOk);
  ASSERT_EQ(sub_output->data.f[0], -1);
  ASSERT_EQ(sub_output->data.f[1], 1);
  ASSERT_EQ(sub_output->data.f[2], 3);
  TfLiteSignatureRunnerDelete(sub_runner);

  TfLiteInterpreterDelete(interpreter);
  TfLiteModelDelete(model);
}

}  // namespace
}  // namespace tflite
