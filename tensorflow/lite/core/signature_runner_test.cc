/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/core/signature_runner.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace impl {
namespace {

TEST(SignatureRunnerTest, TestMultiSignatures) {
  TestErrorReporter reporter;
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/multi_signatures.bin", &reporter);
  ASSERT_TRUE(model);
  ops::builtin::BuiltinOpResolver resolver;
  InterpreterBuilder builder(*model, resolver);

  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(builder(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);

  std::vector<const std::string*> signature_defs =
      interpreter->signature_keys();
  ASSERT_EQ(signature_defs.size(), 2);
  ASSERT_EQ(*(signature_defs[0]), "add");
  ASSERT_EQ(*(signature_defs[1]), "sub");
  ASSERT_EQ(interpreter->GetSignatureRunner("dummy"), nullptr);

  SignatureRunner* add_runner =
      interpreter->GetSignatureRunner(signature_defs[0]->c_str());
  ASSERT_NE(add_runner, nullptr);
  ASSERT_EQ(add_runner->signature_key(), "add");
  const std::vector<const char*>& input_names = add_runner->input_names();
  const std::vector<const char*>& output_names = add_runner->output_names();
  ASSERT_EQ(input_names.size(), 1);
  ASSERT_EQ(std::string(input_names[0]), "x");
  ASSERT_EQ(output_names.size(), 1);
  ASSERT_EQ(std::string(output_names[0]), "output_0");
  ASSERT_EQ(add_runner->ResizeInputTensor("x", {2}), kTfLiteOk);
  ASSERT_EQ(add_runner->AllocateTensors(), kTfLiteOk);
  TfLiteTensor* add_input = add_runner->input_tensor("x");
  ASSERT_EQ(add_runner->input_tensor("dummy"), nullptr);
  const TfLiteTensor* add_output = add_runner->output_tensor("output_0");
  ASSERT_EQ(add_runner->output_tensor("dummy"), nullptr);
  ASSERT_NE(add_input, nullptr);
  ASSERT_NE(add_output, nullptr);
  add_input->data.f[0] = 2;
  add_input->data.f[1] = 4;
  ASSERT_EQ(add_runner->Invoke(), kTfLiteOk);
  ASSERT_EQ(add_output->data.f[0], 4);
  ASSERT_EQ(add_output->data.f[1], 6);

  SignatureRunner* sub_runner = interpreter->GetSignatureRunner("sub");
  ASSERT_NE(sub_runner, nullptr);
  ASSERT_EQ(sub_runner->signature_key(), "sub");
  const std::vector<const char*>& input_names2 = sub_runner->input_names();
  const std::vector<const char*>& output_names2 = sub_runner->output_names();
  ASSERT_EQ(input_names2.size(), 1);
  ASSERT_EQ(std::string(input_names2[0]), "x");
  ASSERT_EQ(output_names2.size(), 1);
  ASSERT_EQ(std::string(output_names2[0]), "output_0");
  ASSERT_EQ(sub_runner->ResizeInputTensor("x", {3}), kTfLiteOk);
  ASSERT_EQ(sub_runner->AllocateTensors(), kTfLiteOk);
  TfLiteTensor* sub_input = sub_runner->input_tensor("x");
  const TfLiteTensor* sub_output = sub_runner->output_tensor("output_0");
  ASSERT_NE(sub_input, nullptr);
  ASSERT_NE(sub_output, nullptr);
  sub_input->data.f[0] = 2;
  sub_input->data.f[1] = 4;
  sub_input->data.f[2] = 6;
  ASSERT_EQ(sub_runner->Invoke(), kTfLiteOk);
  ASSERT_EQ(sub_output->data.f[0], -1);
  ASSERT_EQ(sub_output->data.f[1], 1);
  ASSERT_EQ(sub_output->data.f[2], 3);
}

TEST(SignatureRunnerTest, ReverseSignatureModel) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/reverse_signature_model.bin");
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*model,
                               ops::builtin::BuiltinOpResolver{})(&interpreter),
            kTfLiteOk);
  ASSERT_TRUE(interpreter);

  auto signature_runner = interpreter->GetSignatureRunner("serving_default");
  ASSERT_NE(signature_runner, nullptr);

  // Check the legacy the input and output names order.
  auto& input_names = signature_runner->input_names();
  ASSERT_EQ(input_names.size(), 2);
  EXPECT_STREQ(input_names[0], "x");
  EXPECT_STREQ(input_names[1], "y");

  auto& output_names = signature_runner->output_names();
  ASSERT_EQ(output_names.size(), 2);
  EXPECT_STREQ(output_names[0], "prod");
  EXPECT_STREQ(output_names[1], "sum");

  // Check if the input and output names are in the order of the subgraph
  // inputs and outputs instead of the signature appearance order.
  auto& subgraph_input_names = signature_runner->subgraph_input_names();
  ASSERT_EQ(subgraph_input_names.size(), 2);
  EXPECT_STREQ(subgraph_input_names[0], "y");
  EXPECT_STREQ(subgraph_input_names[1], "x");

  auto& subgraph_output_names = signature_runner->subgraph_output_names();
  ASSERT_EQ(subgraph_output_names.size(), 2);
  EXPECT_STREQ(subgraph_output_names[0], "sum");
  EXPECT_STREQ(subgraph_output_names[1], "prod");
}

}  // namespace
}  // namespace impl
}  // namespace tflite
