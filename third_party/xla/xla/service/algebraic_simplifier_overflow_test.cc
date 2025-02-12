/* Copyright 2017 The OpenXLA Authors.

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

#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

class AlgebraicSimplifierOverflowTest : public HloTestBase {};

// Test that the algebraic simplifier does not generate integer overflows
// by moving the subtraction to the other side of the comparison
TEST_F(AlgebraicSimplifierOverflowTest, CompareOptOverflow) {
  const std::string hlo_text = R"(
    HloModule m
    ENTRY test {
      a = s32[2] parameter(0)
      b = s32[2] constant({1, 1})
      diff = s32[2] subtract(a, b)
      c = s32[2] constant({2147483647, 2147483647})
      ROOT ret = pred[2] compare(diff, c), direction=GT
    }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0}));
}

}  // namespace
}  // namespace xla
