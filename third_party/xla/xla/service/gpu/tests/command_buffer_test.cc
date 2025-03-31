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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class CommandBufferTest : public HloTestBase {};

TEST_F(CommandBufferTest, Fusions) {
  constexpr absl::string_view hlo_text = R"(
  HloModule m, is_scheduled=true

  double {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] add(p0, p0)
  }

  square {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] multiply(p0, p0)
  }

  sum {
    p0 = f32[2,2] parameter(0)
    p1 = f32[2,2] parameter(1)
    ROOT sum = f32[2,2] add(p0, p1)
  }

  command_buffer {
    p0 = f32[2,2] parameter(0)
    f0 = f32[2,2] fusion(p0), kind=kLoop, calls=double
    f1 = f32[2,2] fusion(p0), kind=kLoop, calls=square
    ROOT f3 = f32[2,2] fusion(f0, f1), kind=kLoop, calls=sum
  }

  ENTRY main {
    p0 = f32[2,2] parameter(0)
    ROOT call = f32[2,2] call(p0), to_apply=command_buffer
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  Literal argument = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  Literal expected = LiteralUtil::CreateR2<float>({{3.0, 8.0}, {15.0, 24.0}});

  Literal result = ExecuteNoHloPasses(std::move(module), {&argument});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

TEST_F(CommandBufferTest, TrueFalseConditional) {
  constexpr absl::string_view hlo_text = R"(
  HloModule m, is_scheduled=true

  double {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] add(p0, p0)
  }

  square {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] multiply(p0, p0)
  }

  double_computation {
    p0 = f32[2,2] parameter(0)
    ROOT double = f32[2,2] fusion(p0), kind=kLoop, calls=double
  }

  square_computation {
    p0 = f32[2,2] parameter(0)
    ROOT square = f32[2,2] fusion(p0), kind=kLoop, calls=square
  }

  command_buffer {
    p0 = pred[] parameter(0)
    p1 = f32[2,2] parameter(1)
    ROOT conditional = f32[2,2] conditional(p0, p1, p1),
                                true_computation=double_computation,
                                false_computation=square_computation
  }

  ENTRY main {
    p0 = pred[] parameter(0)
    p1 = f32[2,2] parameter(1)
    ROOT call = f32[2,2] call(p0, p1), to_apply=command_buffer
  })";

  Literal p1 = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});

  {  // Execute `true` branch.
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_text));

    Literal pred = LiteralUtil::CreateR0<bool>(true);
    Literal expected = LiteralUtil::CreateR2<float>({{2.0, 4.0}, {6.0, 8.0}});
    Literal result = ExecuteNoHloPasses(std::move(m), {&pred, &p1});
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  {  // Execute `false` branch.
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_text));

    Literal pred = LiteralUtil::CreateR0<bool>(false);
    Literal expected = LiteralUtil::CreateR2<float>({{1.0, 4.0}, {9.0, 16.0}});
    Literal result = ExecuteNoHloPasses(std::move(m), {&pred, &p1});
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }
}

TEST_F(CommandBufferTest, IndexConditional) {
  constexpr absl::string_view hlo_text = R"(
  HloModule m, is_scheduled=true

  double {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] add(p0, p0)
  }

  square {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] multiply(p0, p0)
  }

  double_computation {
    p0 = f32[2,2] parameter(0)
    ROOT double = f32[2,2] fusion(p0), kind=kLoop, calls=double
  }

  square_computation {
    p0 = f32[2,2] parameter(0)
    ROOT square = f32[2,2] fusion(p0), kind=kLoop, calls=square
  }

  command_buffer {
    p0 = s32[] parameter(0)
    p1 = f32[2,2] parameter(1)
    ROOT conditional = f32[2,2] conditional(p0, p1, p1),
      branch_computations={double_computation, square_computation}
  }

  ENTRY main {
    p0 = s32[] parameter(0)
    p1 = f32[2,2] parameter(1)
    ROOT call = f32[2,2] call(p0, p1), to_apply=command_buffer
  })";

  Literal p1 = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});

  {  // Execute `0` branch.
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_text));

    Literal index = LiteralUtil::CreateR0<int32_t>(0);
    Literal expected = LiteralUtil::CreateR2<float>({{2.0, 4.0}, {6.0, 8.0}});
    Literal result = ExecuteNoHloPasses(std::move(m), {&index, &p1});
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  {  // Execute `1` branch.
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_text));

    Literal index = LiteralUtil::CreateR0<int32_t>(1);
    Literal expected = LiteralUtil::CreateR2<float>({{1.0, 4.0}, {9.0, 16.0}});
    Literal result = ExecuteNoHloPasses(std::move(m), {&index, &p1});
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }

  {  // Execute `1024` branch (our of bound index executes N-1 branch).
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_text));

    Literal index = LiteralUtil::CreateR0<int32_t>(1024);
    Literal expected = LiteralUtil::CreateR2<float>({{1.0, 4.0}, {9.0, 16.0}});
    Literal result = ExecuteNoHloPasses(std::move(m), {&index, &p1});
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }
}

TEST_F(CommandBufferTest, WhileLoop) {
  constexpr absl::string_view hlo_text = R"(
  HloModule m, is_scheduled=true

  compare_fusion {
    p0 = s32[] parameter(0)
    ten = s32[] constant(10)
    ROOT compare = compare(p0, ten), direction=LT
  }

  add_one {
    p0 = s32[] parameter(0)
    one = s32[] constant(1)
    ROOT add = add(p0, one)
  }

  add_two {
    p0 = f32[] parameter(0)
    two = f32[] constant(2.0)
    ROOT add = add(p0, two)
  }

  body {
    p0 = (s32[], f32[]) parameter(0)
    cnt = get-tuple-element(p0), index=0
    val = get-tuple-element(p0), index=1
    add_cnt = s32[] fusion(cnt), kind=kLoop, calls=add_one
    add_val = f32[] fusion(val), kind=kLoop, calls=add_two
    ROOT tuple = (s32[], f32[]) tuple(add_cnt, add_val)
  }

  cond {
    p0 = (s32[], f32[]) parameter(0)
    cnt = get-tuple-element(p0), index=0
    ROOT compare = pred[] fusion(cnt), kind=kLoop, calls=compare_fusion
  }

  command_buffer {
    p0 = (s32[], f32[]) parameter(0)
    ROOT while = while(p0), condition=cond, body=body
  }

  ENTRY main {
    p0 = (s32[], f32[]) parameter(0)
    ROOT call = (s32[], f32[]) call(p0), to_apply=command_buffer
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  Literal cnt = LiteralUtil::CreateR0<int32_t>(0);
  Literal value = LiteralUtil::CreateR0<float>(0.0);
  Literal argument = LiteralUtil::MakeTuple({&cnt, &value});

  Literal expected_cnt = LiteralUtil::CreateR0<int32_t>(10);
  Literal expected_value = LiteralUtil::CreateR0<float>(20.0);
  Literal expected = LiteralUtil::MakeTuple({&expected_cnt, &expected_value});

  Literal result = ExecuteNoHloPasses(std::move(module), {&argument});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

}  // namespace
}  // namespace xla::gpu
