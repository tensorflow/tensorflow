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
#include "tensorflow/compiler/mlir/lite/offset_buffer.h"

#include "tensorflow/core/platform/test.h"

namespace tflite {
namespace {

TEST(OffsetBufferTest, IsValidBufferOffsetTrueGreaterThan1) {
  EXPECT_TRUE(IsValidBufferOffset(/*offset=*/2));
}

TEST(OffsetBufferTest, IsValidBufferOffsetFalseForLessThanOrEqualTo1) {
  EXPECT_FALSE(IsValidBufferOffset(/*offset=*/1));
  EXPECT_FALSE(IsValidBufferOffset(/*offset=*/0));
}

}  // namespace
}  // namespace tflite
