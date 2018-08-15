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

#include "tensorflow/contrib/lite/experimental/c/c_api_experimental.h"

#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/experimental/c/c_api.h"
#include "tensorflow/contrib/lite/testing/util.h"

namespace {

TEST(CApiExperimentalSimple, Smoke) {
  TFL_Model* model = TFL_NewModelFromFile(
      "tensorflow/contrib/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TFL_Interpreter* interpreter =
      TFL_NewInterpreter(model, /*optional_options=*/nullptr);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TFL_InterpreterAllocateTensors(interpreter), kTfLiteOk);

  EXPECT_EQ(TFL_InterpreterResetVariableTensorsToZero(interpreter), kTfLiteOk);

  TFL_DeleteModel(model);
  TFL_DeleteInterpreter(interpreter);
}

}  // namespace

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
