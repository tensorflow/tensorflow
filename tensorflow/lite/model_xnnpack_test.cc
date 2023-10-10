/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/macros.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/util.h"

namespace tflite {

TEST(FloatModel, WithXnnpackDelegate) {
  // Note: this graph will be fully delegated by the XNNPACK delegate.
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/multi_add.bin");
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*model,
                               ops::builtin::BuiltinOpResolver())(&interpreter),
            kTfLiteOk);
  ASSERT_TRUE(interpreter);

  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

#if TFLITE_HAS_ATTRIBUTE_WEAK || defined(TFLITE_BUILD_WITH_XNNPACK_DELEGATE)
  // As the graph is fully delegated by XNNPACK delegate, we will expect the
  // following:
  EXPECT_EQ(1, interpreter->execution_plan().size());
  int first_node_id = interpreter->execution_plan()[0];
  const auto& first_node_reg =
      interpreter->node_and_registration(first_node_id)->second;
  const std::string op_name = GetOpNameByRegistration(first_node_reg);
  EXPECT_EQ("DELEGATE TfLiteXNNPackDelegate", op_name);
#endif
}

TEST(FloatModel, DefaultXnnpackDelegateNotAllowed) {
  // Note: this graph will be fully delegated by the XNNPACK delegate.
  auto model = FlatBufferModel::BuildFromFile(
      "tensorflow/lite/testdata/multi_add.bin");
  ASSERT_TRUE(model);

  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          *model, ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &interpreter),
      kTfLiteOk);
  ASSERT_TRUE(interpreter);

  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

#if TFLITE_HAS_ATTRIBUTE_WEAK || defined(TFLITE_BUILD_WITH_XNNPACK_DELEGATE)
  // As we don't allow applying xnnpack delegate by default, we will expect the
  // following:
  EXPECT_LT(1, interpreter->execution_plan().size());
  int first_node_id = interpreter->execution_plan()[0];
  const auto& first_node_reg =
      interpreter->node_and_registration(first_node_id)->second;
  const std::string op_name = GetOpNameByRegistration(first_node_reg);
  EXPECT_EQ("ADD", op_name);
#endif
}

}  // namespace tflite
