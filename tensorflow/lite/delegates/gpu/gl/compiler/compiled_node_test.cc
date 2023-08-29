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

#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

bool VariableDuplicates(std::vector<Variable> variables) {
  std::sort(
      std::begin(variables), std::end(variables),
      [](const auto& lhs, const auto& rhs) { return lhs.name < rhs.name; });
  for (int i = 0; i < variables.size() - 1; ++i) {
    if (variables[i].name == variables[i + 1].name) return true;
  }
  return false;
}

TEST(CompiledNodeTest, NoDuplicates) {
  Variable scalar;
  scalar.name = "scalar";
  Variable scalar1;
  scalar1.name = "scalar1";
  CompiledNodeAttributes attr;
  CompiledNodeAttributes merged_attr;
  attr.code.parameters = {scalar, scalar1};
  merged_attr.code.parameters = {scalar};
  ASSERT_OK(MergeCode(&attr, &merged_attr));

  // Check for duplicates
  EXPECT_FALSE(VariableDuplicates(merged_attr.code.parameters));
}

TEST(CompiledNodeTest, NameConvergenceConflict) {
  Variable scalar;
  scalar.name = "scalar";
  Variable scalar1;
  scalar1.name = "scalar1";
  CompiledNodeAttributes attr;
  CompiledNodeAttributes merged_attr;
  attr.code.parameters = {scalar1, scalar};
  merged_attr.code.parameters = {scalar};
  ASSERT_OK(MergeCode(&attr, &merged_attr));

  // Check for duplicates
  EXPECT_FALSE(VariableDuplicates(merged_attr.code.parameters));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
