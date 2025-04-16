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

}  // namespace
}  // namespace xla::gpu
