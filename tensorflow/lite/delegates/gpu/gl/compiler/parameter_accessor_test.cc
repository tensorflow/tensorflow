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

#include "tensorflow/lite/delegates/gpu/gl/compiler/parameter_accessor.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

TEST(Preprocessor, CornerCases) {
  ParameterAccessor accessor(true);
  std::string result;
  ASSERT_EQ(accessor.Rewrite("unknown", &result),
            RewriteStatus::NOT_RECOGNIZED);
}

TEST(Preprocessor, Value) {
  ParameterAccessor accessor(true);
  ASSERT_TRUE(accessor.AddParameter(UniformParameter{"var", int32_t(1)}));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("var", &result), RewriteStatus::SUCCESS);
  ASSERT_EQ(result, "1");
}

TEST(Preprocessor, ValueVec) {
  ParameterAccessor accessor(true);
  ASSERT_TRUE(accessor.AddParameter(UniformParameter{"var", int2(1, 2)}));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("var", &result), RewriteStatus::SUCCESS);
  ASSERT_EQ(result, "ivec2(1,2)");
}

TEST(Preprocessor, Field) {
  ParameterAccessor accessor(true);
  ASSERT_TRUE(
      accessor.AddParameter(UniformParameter{"var", float2(1.0, 2.1234567)}));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("var.y", &result), RewriteStatus::SUCCESS);
  ASSERT_EQ(result, "2.123456717f");
}

TEST(Preprocessor, FieldFail) {
  ParameterAccessor accessor(true);
  ASSERT_TRUE(accessor.AddParameter(UniformParameter{"var", 1.0f}));
  ASSERT_TRUE(accessor.AddParameter(UniformParameter{"vec", float2(1.0, 1.0)}));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("var.y", &result), RewriteStatus::ERROR);
  ASSERT_EQ(result, "INVALID_ACCESS_BY_FIELD");

  result.clear();
  EXPECT_EQ(accessor.Rewrite("vec.z", &result), RewriteStatus::ERROR);
  ASSERT_EQ(result, "INVALID_ACCESS_BY_FIELD");
}

TEST(Preprocessor, Variable) {
  ParameterAccessor accessor(true);
  std::vector<int2> v;
  v.push_back(int2(1, 2));
  ASSERT_TRUE(accessor.AddParameter(UniformParameter{"var", v}));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("var[i].y", &result), RewriteStatus::SUCCESS);
  ASSERT_EQ(result, "var[i].y");
  ASSERT_EQ(accessor.GetConstDeclarations(),
            "const ivec2 var[] = ivec2[1](ivec2(1,2));\n");
}

TEST(Preprocessor, InlineVariableFail) {
  ParameterAccessor accessor(true);
  ASSERT_TRUE(accessor.AddParameter(UniformParameter{"var", 1}));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("var[i]", &result), RewriteStatus::ERROR);
  ASSERT_EQ(result, "INVALID_ACCESS_BY_INDEX");
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
