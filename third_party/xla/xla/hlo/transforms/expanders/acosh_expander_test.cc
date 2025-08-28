/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/acosh_expander.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/pattern_matcher.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = match;

class AcoshExpanderTest : public HloHardwareIndependentTestBase {};

TEST_F(AcoshExpanderTest, ExpandWith) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = f32[2,3] parameter(0)
      ROOT r = f32[2,3] acosh(p)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));

  auto computation = m->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAcosh);
  AcoshExpander acosh_expander;
  ASSERT_TRUE(acosh_expander.Run(m.get()).value());
  root = computation->root_instruction();
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Select()));
}

}  // namespace
}  // namespace xla
