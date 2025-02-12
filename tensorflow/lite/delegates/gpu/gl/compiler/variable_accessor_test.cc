/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

TEST(PreprocessorTest, CornerCases) {
  VariableAccessor variable_accessor(/*inline_values=*/true);
  std::string result;
  EXPECT_EQ(variable_accessor.Rewrite("unknown", &result),
            RewriteStatus::NOT_RECOGNIZED);
}

TEST(PreprocessorTest, Value) {
  VariableAccessor variable_accessor(/*inline_values=*/true);
  ASSERT_TRUE(variable_accessor.AddUniformParameter({"var", int32_t(1)}));
  std::string result;
  ASSERT_EQ(variable_accessor.Rewrite("var", &result), RewriteStatus::SUCCESS);
  EXPECT_EQ(result, "1");
}

TEST(PreprocessorTest, ValueVec) {
  VariableAccessor variable_accessor(/*inline_values=*/true);
  ASSERT_TRUE(variable_accessor.AddUniformParameter({"var", int2(1, 2)}));
  std::string result;
  ASSERT_EQ(variable_accessor.Rewrite("var", &result), RewriteStatus::SUCCESS);
  EXPECT_EQ(result, "ivec2(1,2)");
}

TEST(PreprocessorTest, Field) {
  VariableAccessor variable_accessor(/*inline_values=*/true);
  ASSERT_TRUE(
      variable_accessor.AddUniformParameter({"var", float2(1.0, 2.1234567)}));
  std::string result;
  ASSERT_EQ(variable_accessor.Rewrite("var.y", &result),
            RewriteStatus::SUCCESS);
  EXPECT_EQ(result, "2.123456717f");
}

TEST(PreprocessorTest, FieldFail) {
  VariableAccessor variable_accessor(/*inline_values=*/true);
  ASSERT_TRUE(variable_accessor.AddUniformParameter({"var", 1.0f}));
  ASSERT_TRUE(variable_accessor.AddUniformParameter({"vec", float2(1.0, 1.0)}));
  std::string result;
  ASSERT_EQ(variable_accessor.Rewrite("var.y", &result), RewriteStatus::ERROR);
  EXPECT_EQ(result, "INVALID_ACCESS_BY_FIELD");

  result.clear();
  ASSERT_EQ(variable_accessor.Rewrite("vec.z", &result), RewriteStatus::ERROR);
  EXPECT_EQ(result, "INVALID_ACCESS_BY_FIELD");
}

TEST(PreprocessorTest, Variable) {
  VariableAccessor variable_accessor(/*inline_values=*/true);
  std::vector<int2> v;
  v.push_back(int2(1, 2));
  ASSERT_TRUE(variable_accessor.AddUniformParameter({"var", v}));
  std::string result;
  ASSERT_EQ(variable_accessor.Rewrite("var[i].y", &result),
            RewriteStatus::SUCCESS);
  ASSERT_EQ(result, "var[i].y");
  EXPECT_EQ(variable_accessor.GetConstDeclarations(),
            "const ivec2 var[] = ivec2[1](ivec2(1,2));\n");
}

TEST(PreprocessorTest, InlineVariableFail) {
  VariableAccessor variable_accessor(/*inline_values=*/true);
  ASSERT_TRUE(variable_accessor.AddUniformParameter({"var", 1}));
  std::string result;
  ASSERT_EQ(variable_accessor.Rewrite("var[i]", &result), RewriteStatus::ERROR);
  EXPECT_EQ(result, "INVALID_ACCESS_BY_INDEX");
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
