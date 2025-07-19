/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>
#include <utility>

#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class DynamicReshapeTest : public HloTestBase {};

TEST_F(DynamicReshapeTest, SingleDynamicDimension) {
  constexpr const char* kModuleStr = R"(
    HloModule DynamicReshapeTest.SingleDynamicDimension

    ENTRY main {
      param = s32[2, 3, 3] parameter(0)
      two = s32[] parameter(1)
      param_padded = s32[2, <=3, 3] set-dimension-size(param, two),
        dimensions={1}
      nine = s32[] parameter(2)
      ROOT reshaped = s32[<=18] dynamic-reshape(param_padded, nine)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR3<int32_t>(
      {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
       {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}}});
  Literal arg1 = LiteralUtil::CreateR0<int32_t>(2);
  Literal arg2 = LiteralUtil::CreateR0<int32_t>(9);

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          Execute(std::move(module), {&arg0, &arg1, &arg2}));

  Literal expected =
      LiteralUtil::CreateR1<int32_t>({0, 1, 2, 3, 4, 5, 9, 10, 11});
  EXPECT_EQ(result, expected);
}

TEST_F(DynamicReshapeTest, DoubleDynamicDimensions) {
  constexpr const char* kModuleStr = R"(
    HloModule DynamicReshapeTest.DoubleDynamicDimensions

    ENTRY main {
      param = s32[2, 3, 3] parameter(0)
      two = s32[] parameter(1)
      param_padded_partial = s32[2, <=3, 3] set-dimension-size(param, two),
        dimensions={1}
      param_padded = s32[2, <=3, <=3] set-dimension-size(param_padded_partial,
        two), dimensions={2}
      eight = s32[] parameter(2)
      ROOT reshaped = s32[<=18] dynamic-reshape(param_padded, eight)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR3<int32_t>(
      {{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}},
       {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}}});
  Literal arg1 = LiteralUtil::CreateR0<int32_t>(2);
  Literal arg2 = LiteralUtil::CreateR0<int32_t>(8);

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          Execute(std::move(module), {&arg0, &arg1, &arg2}));

  Literal expected =
      LiteralUtil::CreateR1<int32_t>({0, 1, 3, 4, 9, 10, 12, 13});
  EXPECT_EQ(result, expected);
}

TEST_F(DynamicReshapeTest, OutputDoubleDynamicDimensions) {
  constexpr const char* kModuleStr = R"(
    HloModule DynamicReshapeTest.OutputDoubleDynamicDimensions

    ENTRY main {
      param = s32[18] parameter(0)
      eight = s32[] parameter(1)
      param_dynamic = s32[<=18] set-dimension-size(param, eight), dimensions={0}
      two = s32[] parameter(2)
      ROOT reshaped = s32[2, <=3, <=3] dynamic-reshape(param_dynamic, two, two,
        two)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR1<int32_t>(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Literal arg1 = LiteralUtil::CreateR0<int32_t>(8);
  Literal arg2 = LiteralUtil::CreateR0<int32_t>(2);

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          Execute(std::move(module), {&arg0, &arg1, &arg2}));

  Literal expected =
      LiteralUtil::CreateR3<int32_t>({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}});
  EXPECT_EQ(result, expected);
}

TEST_F(DynamicReshapeTest, Complicated) {
  constexpr const char* kModuleStr = R"(
    HloModule DynamicReshapeTest.Complicated

    ENTRY main {
      param = s32[3, 4, 4] parameter(0)
      two = s32[] parameter(1)
      param_dynamic = s32[<=3, 4, 4] set-dimension-size(param, two),
        dimensions={0}
      three = s32[] parameter(2)
      param_dynamic1 = s32[<=3, <=4, 4] set-dimension-size(
        param_dynamic, three), dimensions={1}
      param_dynamic2 = s32[<=3, <=4, <=4] set-dimension-size(
        param_dynamic1, three), dimensions={2}
      six = s32[] parameter(3)

      // Static reshape is from [3, 4, 4] to [6, 8].
      // Dynamic reshape is from [2, 3, 3] to [3, 6].
      ROOT reshaped = s32[<=6, <=8] dynamic-reshape(param_dynamic2, three, six)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR3<int32_t>(
      {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}},
       {{16, 17, 18, 19}, {20, 21, 22, 23}, {24, 25, 26, 27}, {28, 29, 30, 31}},
       {{32, 33, 34, 35},
        {36, 37, 38, 39},
        {40, 41, 42, 43},
        {44, 45, 46, 47}}});
  Literal arg1 = LiteralUtil::CreateR0<int32_t>(2);
  Literal arg2 = LiteralUtil::CreateR0<int32_t>(3);
  Literal arg3 = LiteralUtil::CreateR0<int32_t>(6);

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, Execute(std::move(module), {&arg0, &arg1, &arg2, &arg3}));

  Literal expected = LiteralUtil::CreateR2<int32_t>(
      {{0, 1, 2, 4, 5, 6}, {8, 9, 10, 16, 17, 18}, {20, 21, 22, 24, 25, 26}});
  EXPECT_EQ(result, expected);
}

}  // namespace
}  // namespace xla
