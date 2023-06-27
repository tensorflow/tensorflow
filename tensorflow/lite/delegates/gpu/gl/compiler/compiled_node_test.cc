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

  // Count instances of "scalar1". Expect only 1.
  int count = 0;
  for (const Variable& var : merged_attr.code.parameters) {
    if (var.name == "scalar1") {
      ++count;
    }
  }
  EXPECT_EQ(count, 1);
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
