/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/real_imag_expander.h"

#include <memory>
#include <utility>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/types.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace m = match;

class RealImagExpanderTest : public HloTestBase {};

TEST_F(RealImagExpanderTest, RealWithNonComplexInput) {
  const char* kModuleStr = R"(
    HloModule real_float
    ENTRY main {
      input = f32[4] parameter(0)
      ROOT real = real(input)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  RealImagExpander expander;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&expander, module.get()));
  EXPECT_TRUE(result);

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter(0)));
}

TEST_F(RealImagExpanderTest, ImagWithNonComplexInput) {
  const char* kModuleStr = R"(
    HloModule imag_float
    ENTRY main {
      input = f32[4,2,8] parameter(0)
      ROOT imag = imag(input)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  RealImagExpander expander;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&expander, module.get()));
  EXPECT_TRUE(result);

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast()));

  XLA_VLOG_LINES(1, module->ToString());
}

TEST_F(RealImagExpanderTest, RealImagWithComplexInput) {
  const char* kModuleStr = R"(
    HloModule real_float
    ENTRY main {
      input = c64[4] parameter(0)
      real = real(input)
      imag = imag(input)
      ROOT t = tuple(real, imag)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  RealImagExpander expander;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&expander, module.get()));

  // If inputs are complex, the pass should not change anything.
  EXPECT_FALSE(result);
}

TEST_F(RealImagExpanderTest, MultipleImagWithNonComplexInput) {
  const char* kModuleStr = R"(
    HloModule imag_float
    ENTRY main {
      input = f32[4,2,8] parameter(0)
      imag1 = imag(input)
      ROOT imag2 = imag(imag1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  // Replace imag1 with an identical instruction, changing the iteration order
  // of computation->instructions(). Previously OpExpanderPass could crash if
  // the instructions were not in post-order, and this tests that the crash does
  // not reoccur.
  auto param = module->entry_computation()->parameter_instruction(0);
  HloInstruction* imag1 =
      module->entry_computation()->root_instruction()->mutable_operand(0);
  TF_ASSERT_OK_AND_ASSIGN(HloInstruction * new_imag,
                          MakeUnaryHlo(HloOpcode::kImag, param));
  TF_ASSERT_OK(
      module->entry_computation()->ReplaceInstruction(imag1, new_imag));

  RealImagExpander expander;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&expander, module.get()));
  EXPECT_TRUE(result);

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast()));

  XLA_VLOG_LINES(1, module->ToString());
}

}  // namespace
}  // namespace xla
