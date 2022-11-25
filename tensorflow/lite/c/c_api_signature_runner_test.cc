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
