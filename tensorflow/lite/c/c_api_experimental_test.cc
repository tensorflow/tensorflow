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

#include "tensorflow/lite/c/c_api_experimental.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/testing/util.h"

namespace {

const TfLiteRegistration* GetDummyRegistration() {
  static const TfLiteRegistration registration = {
      /*init=*/nullptr,
      /*free=*/nullptr,
      /*prepare=*/nullptr,
      /*invoke=*/[](TfLiteContext*, TfLiteNode*) { return kTfLiteOk; }};
  return &registration;
}

TEST(CApiExperimentalTest, Smoke) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddBuiltinOp(options, kTfLiteBuiltinAdd,
                                       GetDummyRegistration(), 1, 1);
  TfLiteInterpreterOptionsSetUseNNAPI(options, true);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterResetVariableTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

// Test using TfLiteInterpreterCreateWithSelectedOps.
TEST(CApiExperimentalTest, SelectedBuiltins) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddBuiltinOp(options, kTfLiteBuiltinAdd,
                                       GetDummyRegistration(), 1, 1);

  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreateWithSelectedOps(model, options);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterResetVariableTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

// Test that when using TfLiteInterpreterCreateWithSelectedOps,
// we do NOT get the standard builtin operators by default.
TEST(CApiExperimentalTest, MissingBuiltin) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  // Install a custom error reporter into the interpreter by way of options.
  tflite::TestErrorReporter reporter;
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetErrorReporter(
      options,
      [](void* user_data, const char* format, va_list args) {
        reinterpret_cast<tflite::TestErrorReporter*>(user_data)->Report(format,
                                                                        args);
      },
      &reporter);

  // Create an interpreter with no builtins at all.
  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreateWithSelectedOps(model, options);

  // Check that interpreter creation failed, because the model contain a buitin
  // op that wasn't supported, and that we got the expected error messages.
  ASSERT_EQ(interpreter, nullptr);
  EXPECT_EQ(reporter.error_messages(),
            "Didn't find op for builtin opcode 'ADD' version '1'\n"
            "Registration failed.\n");
  EXPECT_EQ(reporter.num_calls(), 2);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

struct OpResolverData {
  bool called_for_add = false;
};

const TfLiteRegistration* MyFindBuiltinOp(void* user_data,
                                          TfLiteBuiltinOperator op,
                                          int version) {
  OpResolverData* my_data = static_cast<OpResolverData*>(user_data);
  if (op == kTfLiteBuiltinAdd && version == 1) {
    my_data->called_for_add = true;
    return GetDummyRegistration();
  }
  return nullptr;
}

const TfLiteRegistration* MyFindCustomOp(void*, const char* custom_op,
                                         int version) {
  if (absl::string_view(custom_op) == "foo" && version == 1) {
    return GetDummyRegistration();
  }
  return nullptr;
}

// Test using TfLiteInterpreterCreateWithSelectedOps.
TEST(CApiExperimentalTest, SetOpResolver) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

  OpResolverData my_data;
  TfLiteInterpreterOptionsSetOpResolver(options, MyFindBuiltinOp,
                                        MyFindCustomOp, &my_data);
  EXPECT_FALSE(my_data.called_for_add);

  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreateWithSelectedOps(model, options);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterResetVariableTensors(interpreter), kTfLiteOk);
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);
  EXPECT_TRUE(my_data.called_for_add);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
}

}  // namespace

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
