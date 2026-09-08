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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class ReduceWindowRewriterExecutionTest
    : public HloPjRtInterpreterReferenceMixin<HloTestBase> {};

TEST_F(ReduceWindowRewriterExecutionTest, RewriterTest) {
  absl::string_view hlo_string = R"(
HloModule module

addition {
  x.6 = f32[] parameter(0)
  y.7 = f32[] parameter(1)
  ROOT add.8 = f32[] add(x.6, y.7)
}

ENTRY entry {
  arg0.1 = f32[1000]{0} parameter(0), parameter_replication={false}
  reshape.2 = f32[1000]{0} reshape(arg0.1)
  convert.4 = f32[1000]{0} convert(reshape.2)
  constant.3 = f32[] constant(0)
  reduce-window.9 = f32[1000]{0} reduce-window(convert.4, constant.3), window={size=1000 pad=999_0}, to_apply=addition
  convert.10 = f32[1000]{0} convert(reduce-window.9)
  reshape.11 = f32[1000]{0} reshape(convert.10)
  tuple.12 = (f32[1000]{0}) tuple(reshape.11)
  ROOT get-tuple-element.13 = f32[1000]{0} get-tuple-element(tuple.12), index=0
}
)";

  // Verify correctness of the rewrite.
  EXPECT_TRUE(RunAndCompare(hlo_string, ErrorSpec{1e-5, 1e-5}));

  // Check that the tree reduction decomposition did happen.
  MatchOptimizedHlo(hlo_string, R"(
// CHECK: reduce-window
// CHECK: reduce-window
  )");
}

TEST_F(ReduceWindowRewriterExecutionTest, DynamicDimensionScan) {
  // Regression test for https://github.com/openxla/xla/issues/44944: a
  // cumsum scan over a dynamic dimension through the full pass pipeline.
  absl::string_view hlo_string = R"(
HloModule module

addition {
  x = s32[] parameter(0)
  y = s32[] parameter(1)
  ROOT add = s32[] add(x, y)
}

ENTRY entry {
  arg = s32[64]{0} parameter(0)
  size = s32[] constant(48)
  arg_dynamic = s32[<=64]{0} set-dimension-size(arg, size), dimensions={0}
  constant = s32[] constant(0)
  ROOT reduce-window = s32[<=64]{0} reduce-window(arg_dynamic, constant), window={size=64 pad=63_0}, to_apply=addition
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  std::vector<int32_t> ones(64, 1);
  Literal arg = LiteralUtil::CreateR1<int32_t>(ones);
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&arg}));

  // Cumulative sums of 48 ones: 1, 2, ..., 48.
  std::vector<int32_t> prefix_sums(48);
  absl::c_iota(prefix_sums, 1);
  Literal expected = LiteralUtil::CreateR1<int32_t>(prefix_sums);

  EXPECT_EQ(result.ToStatic(), expected);
}

}  // namespace
}  // namespace xla
