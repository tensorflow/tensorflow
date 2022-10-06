/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/optional_debug_tools.h"

#include <algorithm>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

namespace tflite {
namespace {
// This is specific to the testdata/add.bin model used in the test.
void InitInputTensorData(Interpreter* interpreter) {
  ASSERT_EQ(interpreter->inputs().size(), 1);
  TfLiteTensor* t = interpreter->input_tensor(0);
  ASSERT_EQ(t->type, kTfLiteFloat32);
  float* data = static_cast<float*>(t->data.data);
  int num_elements = t->bytes / sizeof(float);
  std::fill(data, data + num_elements, 1.0f);
}
}  // namespace

TEST(OptionalDebugTools, PrintInterpreterState) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/add.bin");
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          *model, ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &interpreter),
      kTfLiteOk);

  // Ensure printing the interpreter state doesn't crash:
  //   * Before allocation
  //   * After allocation
  //   * After invocation
  PrintInterpreterState(interpreter.get());

  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  PrintInterpreterState(interpreter.get());

  InitInputTensorData(interpreter.get());
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  PrintInterpreterState(interpreter.get());
}

TEST(OptionalDebugTools, PrintInterpreterStateWithDelegate) {
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/add.bin");
  ASSERT_TRUE(model);

  // Create and instantiate an interpreter with a delegate.
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          *model, ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &interpreter),
      kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(xnnpack_delegate.get()),
            kTfLiteOk);

  InitInputTensorData(interpreter.get());
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  // Ensure printing the interpreter state doesn't crash.
  PrintInterpreterState(interpreter.get());
}

}  // namespace tflite
