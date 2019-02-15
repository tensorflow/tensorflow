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

#include "tensorflow/lite/experimental/c/c_api_experimental.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/experimental/c/c_api.h"
#include "tensorflow/lite/testing/util.h"

namespace {

TfLiteRegistration* GetDummyRegistration() {
  static TfLiteRegistration registration = {
      .init = nullptr,
      .free = nullptr,
      .prepare = nullptr,
      .invoke = [](TfLiteContext*, TfLiteNode*) { return kTfLiteOk; },
  };
  return &registration;
}

TEST(CApiExperimentalSimple, Smoke) {
  TFL_Model* model = TFL_NewModelFromFile(
      "tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TFL_InterpreterOptions* options = TFL_NewInterpreterOptions();
  TFL_InterpreterOptionsAddBuiltinOp(options, kTfLiteBuiltinAdd,
                                     GetDummyRegistration(), 1, 1);

  TFL_Interpreter* interpreter = TFL_NewInterpreter(model, options);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TFL_InterpreterAllocateTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TFL_InterpreterResetVariableTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TFL_InterpreterInvoke(interpreter), kTfLiteOk);

  TFL_DeleteInterpreter(interpreter);
  TFL_DeleteInterpreterOptions(options);
  TFL_DeleteModel(model);
}

}  // namespace

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
