// Copyright 2025 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "xla/hlo/transforms/expanders/convolution_type_canonicalizer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class ConvolutionTypeCanonicalizerTest : public HloHardwareIndependentTestBase {
};

TEST_F(ConvolutionTypeCanonicalizerTest, DotBf16ToS32) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  p0 = bf16[10,10]{1,0} parameter(0)
  p1 = bf16[10,10]{1,0} parameter(1)
  ROOT dot = s32[10,10]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ConvolutionTypeCanonicalizer pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Convert(op::Dot(op::Parameter(0), op::Parameter(1))));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              xla::testing::opcode_matchers::Shape("s32[10,10]{1,0}"));
}

TEST_F(ConvolutionTypeCanonicalizerTest, ConvolutionBf16ToS32) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  p0 = bf16[1,1024,1024,1]{3,2,1,0} parameter(0)
  p1 = bf16[1,1,1,1]{3,2,1,0} parameter(1)
  ROOT conv = s32[1,1024,1024,1]{3,2,1,0} convolution(p0, p1), window={size=1x1}, dim_labels=b01f_01io->b01f
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ConvolutionTypeCanonicalizer pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Convert(op::Convolution(op::Parameter(0), op::Parameter(1))));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      xla::testing::opcode_matchers::Shape("s32[1,1024,1024,1]{3,2,1,0}"));
}

TEST_F(ConvolutionTypeCanonicalizerTest, NoChangeNeeded) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY main {
  p0 = f32[10,10]{1,0} parameter(0)
  p1 = f32[10,10]{1,0} parameter(1)
  ROOT dot = f32[10,10]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ConvolutionTypeCanonicalizer pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
