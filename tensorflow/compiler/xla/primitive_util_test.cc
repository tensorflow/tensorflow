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

#include "tensorflow/compiler/xla/primitive_util.h"

#include <numeric>
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(PrimitiveUtilTest, StringToPrimitiveType) {
  auto expect_ok_and_equal = [](const string& str, PrimitiveType expected) {
    TF_ASSERT_OK_AND_ASSIGN(PrimitiveType actual,
                            primitive_util::StringToPrimitiveType(str));
    EXPECT_EQ(expected, actual);
  };
  expect_ok_and_equal("f32", F32);
  expect_ok_and_equal("tuple", TUPLE);
  expect_ok_and_equal("pred", PRED);
  expect_ok_and_equal("s32", S32);

  EXPECT_IS_NOT_OK(primitive_util::StringToPrimitiveType("F32").status());
  EXPECT_IS_NOT_OK(primitive_util::StringToPrimitiveType("Pred").status());
  EXPECT_IS_NOT_OK(primitive_util::StringToPrimitiveType("preD").status());
}

TEST(PrimitiveUtilTest, FloatTypes) {
  EXPECT_EQ(primitive_util::SignificandWidth(F32), 24);
  EXPECT_EQ(primitive_util::SignificandWidth(BF16), 8);
  EXPECT_EQ(primitive_util::ExponentWidth(F32), 8);
  EXPECT_EQ(primitive_util::ExponentWidth(BF16), 8);
}

}  // namespace
}  // namespace xla
