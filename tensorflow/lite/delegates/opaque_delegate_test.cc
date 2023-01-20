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

#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
namespace delegates {

namespace {

TEST(TestOpaqueDelegate, AddDelegate) {
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(
          "third_party/tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate,
                                       void* data) -> TfLiteStatus {
    return kTfLiteOk;
  };
  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  builder.AddDelegate(opaque_delegate);
  std::unique_ptr<tflite::Interpreter> interpreter;
  EXPECT_EQ(kTfLiteOk, builder(&interpreter));
  ASSERT_NE(interpreter, nullptr);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
}

}  // anonymous namespace
}  // namespace delegates
}  // namespace tflite
