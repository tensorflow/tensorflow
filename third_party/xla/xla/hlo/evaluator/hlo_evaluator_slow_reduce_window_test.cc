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
#include <memory>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

TEST_F(HloHardwareIndependentTestBase, SlowReduceWindow) {
  constexpr absl::string_view kHloModule = R"(
    HloModule SlowReduceWindow
    %add {
      %lhs = s32[] parameter(0)
      %rhs = s32[] parameter(1)
      ROOT %sum = s32[] add(%lhs, %rhs)
    }
    ENTRY slow_reduce_window {
      %input = s32[8192] parameter(0)
      %zero = s32[] constant(0)
      ROOT %scan = s32[8192] reduce-window(%input, %zero), window={size=8192 pad=8191_0}, to_apply=%add
    }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  std::vector<int32_t> data(8192, 1);
  auto input = LiteralUtil::CreateR1<int32_t>(data);
  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(
      Literal actual_literal,
      evaluator.Evaluate(*hlo_module->entry_computation(), {&input}));
  std::vector<int32_t> expected(8192);
  std::iota(expected.begin(), expected.end(), 1);
  EXPECT_THAT(actual_literal.data<int32_t>(),
              ::testing::ElementsAreArray(expected));
}

}  // namespace
}  // namespace xla
