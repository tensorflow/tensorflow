/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/conv_operand_swapper.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/window_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class ConvOperandSwapperTest : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module, int64_t distance_threshold = 100) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(
                                         hlo_module, GetModuleConfigForTest()));
    TF_RETURN_IF_ERROR(ConvOperandSwapper().Run(module.get()).status());
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }
};

TEST_F(ConvOperandSwapperTest, SwapConvOperands) {
  constexpr absl::string_view hlo_string = R"(
  HloModule m
  test {
    a = f32[3,3,160,160] parameter(0)
    b = f32[128,32,32,160] parameter(1)
    ROOT c = f32[128,32,32,160] convolution(a,b),
     window={size=32x32 pad=30_30x30_30 rhs_reversal=1x1},
     dim_labels=01bf_o01i->f01b
  }
  )";
  auto module_status = RunPass(hlo_string);
  EXPECT_TRUE(module_status.status().ok());
  auto module = std::move(module_status).value();
  HloInstruction* conv = module->entry_computation()->root_instruction();
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_THAT(conv, op::Convolution(op::Parameter(1), op::Parameter(0)));
  EXPECT_EQ(window_util::ToString(conv->window()),
            "size=3x3 pad=1_1x1_1 rhs_reversal=1x1");
}

}  // namespace
}  // namespace xla
