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

#include "tensorflow/lite/delegates/gpu/gl/compiler/object_accessor.h"

#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/variable_accessor.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {

struct ParameterComparator {
  template <typename T>
  bool operator()(const T& t) const {
    const T* v = std::get_if<T>(&p.value);
    return v && t == *v;
  }
  const Variable& p;
};

// partially equal
bool operator==(const Variable& l, const Variable& r) {
  return l.name == r.name && std::visit(ParameterComparator{l}, r.value);
}

namespace {

TEST(Preprocessor, CornerCases) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  std::string result;
  ASSERT_EQ(accessor.Rewrite("", &result), RewriteStatus::NOT_RECOGNIZED);
  ASSERT_EQ(accessor.Rewrite("=", &result), RewriteStatus::NOT_RECOGNIZED);
}

TEST(Preprocessor, ReadFromBuffer) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  ASSERT_TRUE(
      accessor.AddObject("obj", MakeReadonlyBuffer(std::vector<float>{1.0})));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("obj[i]", &result), RewriteStatus::SUCCESS);
  EXPECT_TRUE(variable_accessor.GetUniformParameters().empty());
  ASSERT_EQ(result, "obj.data[i]");
}

TEST(Preprocessor, ReadFromBufferLinear) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  ASSERT_TRUE(accessor.AddObject(
      "obj", MakeReadonlyBuffer(uint3(1, 2, 3), std::vector<float>{1.0})));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("obj[i]", &result), RewriteStatus::SUCCESS);
  EXPECT_TRUE(variable_accessor.GetUniformParameters().empty());
  ASSERT_EQ(result, "obj.data[i]");
}

TEST(Preprocessor, ReadFromBufferByIndex) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  ASSERT_TRUE(accessor.AddObject(
      "obj", MakeReadonlyBuffer(uint3(1, 2, 3), std::vector<float>{1.0})));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("obj[x,y + 5,z]", &result),
            RewriteStatus::SUCCESS);
  EXPECT_THAT(variable_accessor.GetUniformParameters(),
              testing::UnorderedElementsAre(Variable{"obj_w", 1},
                                            Variable{"obj_h", 2}));
  ASSERT_EQ(result, "obj.data[x + $obj_w$ * (y + 5 + $obj_h$ * (z))]");
}

TEST(Preprocessor, ReadFromTexture) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  ASSERT_TRUE(accessor.AddObject(
      "obj", MakeReadonlyTexture(uint3(1, 2, 3), {1.0, 2.0, 3.0, 4.0})));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("obj[i,j,k]", &result), RewriteStatus::SUCCESS);
  // textures don't need extra variables to be stored for indexed access
  EXPECT_TRUE(variable_accessor.GetUniformParameters().empty());
  ASSERT_EQ(result, "imageLoad(obj, ivec3(i, j, k))");
}

TEST(Preprocessor, ReadFromTexture1D) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  ASSERT_TRUE(
      accessor.AddObject("obj", MakeReadonlyTexture({1.0, 2.0, 3.0, 4.0})));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("obj[i]", &result), RewriteStatus::SUCCESS);
  EXPECT_TRUE(variable_accessor.GetUniformParameters().empty());
  ASSERT_EQ(result, "imageLoad(obj, ivec2(i, 0))");
}

TEST(Preprocessor, WriteToBuffer) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  ASSERT_TRUE(
      accessor.AddObject("obj", MakeReadonlyBuffer(std::vector<float>{1.0})));
  std::string result;
  EXPECT_EQ(accessor.Rewrite(" obj[i]  =value", &result),
            RewriteStatus::SUCCESS);
  EXPECT_TRUE(variable_accessor.GetUniformParameters().empty());
  ASSERT_EQ(result, "obj.data[i] = value");
}

TEST(Preprocessor, WriteToBufferByIndex) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  ASSERT_TRUE(accessor.AddObject(
      "obj", MakeReadonlyBuffer(uint3(1, 2, 3), {1.0, 2.0, 3.0, 4.0})));
  std::string result;
  EXPECT_EQ(accessor.Rewrite(" obj[i,j,k]  =value", &result),
            RewriteStatus::SUCCESS);
  EXPECT_THAT(variable_accessor.GetUniformParameters(),
              testing::UnorderedElementsAre(Variable{"obj_w", 1},
                                            Variable{"obj_h", 2}));
  ASSERT_EQ(result, "obj.data[i + $obj_w$ * (j + $obj_h$ * (k))] = value");
}

TEST(Preprocessor, WriteToTexture) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  ASSERT_TRUE(accessor.AddObject(
      "obj", MakeReadonlyTexture(uint3(1, 1, 1), {1.0, 2.0, 3.0, 4.0})));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("obj[i,j,k]= value ", &result),
            RewriteStatus::SUCCESS);
  ASSERT_EQ(result, "imageStore(obj, ivec3(i, j, k), value)");
}

TEST(Preprocessor, WriteToTexture1D) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  ASSERT_TRUE(
      accessor.AddObject("obj", MakeReadonlyTexture({1.0, 2.0, 3.0, 4.0})));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("obj[i]= value ", &result),
            RewriteStatus::SUCCESS);
  EXPECT_TRUE(variable_accessor.GetUniformParameters().empty());
  ASSERT_EQ(result, "imageStore(obj, ivec2(i, 0), value)");
}

TEST(Preprocessor, FailedWriteToBuffer) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  ASSERT_TRUE(
      accessor.AddObject("obj", MakeReadonlyBuffer(std::vector<float>{1.0})));
  std::string result;
  EXPECT_EQ(accessor.Rewrite(" obj[i,j]  =value", &result),
            RewriteStatus::ERROR);
  ASSERT_EQ(result, "WRONG_NUMBER_OF_INDICES");
}

TEST(Preprocessor, FailedWriteToTexture) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  ASSERT_TRUE(accessor.AddObject(
      "obj", MakeReadonlyTexture(uint3(1, 1, 1), {1.0, 2.0, 3.0, 4.0})));
  std::string result;
  EXPECT_EQ(accessor.Rewrite("obj[i]= value ", &result), RewriteStatus::ERROR);
  ASSERT_EQ(result, "WRONG_NUMBER_OF_INDICES");
}

TEST(Preprocessor, DeclareTexture) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(false, &variable_accessor);
  ASSERT_TRUE(accessor.AddObject(
      "obj", MakeReadonlyTexture(uint3(1, 1, 1), {1.0, 2.0, 3.0, 4.0})));
  ASSERT_EQ(accessor.GetObjectDeclarations(),
            "layout(rgba32f, binding = 0) readonly uniform highp image2DArray "
            "obj;\n");
}

TEST(Preprocessor, DeclareBuffer) {
  VariableAccessor variable_accessor(/*inline_values=*/false);
  ObjectAccessor accessor(true, &variable_accessor);
  ASSERT_TRUE(
      accessor.AddObject("obj", MakeReadonlyBuffer(std::vector<float>{1.0})));
  ASSERT_EQ(accessor.GetObjectDeclarations(),
            "layout(binding = 0) buffer B0 { vec4 data[]; } obj;\n");
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
