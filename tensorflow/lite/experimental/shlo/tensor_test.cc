/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/shlo/tensor.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/data_type.h"

namespace shlo_ref {
namespace {

TEST(TensorTest, BaselineTypeWorks) {
  EXPECT_EQ(BaselineType(DataType::kI1), DataType::kI1);
  EXPECT_EQ(BaselineType(DataType::kSI4), DataType::kSI4);
  EXPECT_EQ(BaselineType(DataType::kSI8), DataType::kSI8);
  EXPECT_EQ(BaselineType(DataType::kSI16), DataType::kSI16);
  EXPECT_EQ(BaselineType(DataType::kSI32), DataType::kSI32);
  EXPECT_EQ(BaselineType(DataType::kBF16), DataType::kBF16);
  EXPECT_EQ(BaselineType(DataType::kF16), DataType::kF16);
  EXPECT_EQ(BaselineType(DataType::kF32), DataType::kF32);
}

}  // namespace

}  // namespace shlo_ref
