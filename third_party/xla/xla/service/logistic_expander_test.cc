/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/logistic_expander.h"

#include <memory>
#include <string_view>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/dynamic_padder.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = match;

class LogisticExpanderTest : public HloTestBase {};


// option is enabled.
TEST_F(LogisticExpanderTest, ExpandWith) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p = f32[2,3] parameter(0)
      ROOT r = f32[2,3] logistic(p)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));

  auto computation = m->entry_computation();
  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kLogistic);
  LogisticExpander logistic_expander;
  ASSERT_TRUE(logistic_expander.Run(m.get()).value());
  root = computation->root_instruction();
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Divide(
                  m::Broadcast(m::ConstantScalar(1.0)),
                  m::AddAnyOrder(m::Broadcast(m::ConstantScalar(1.0)),
                                 m::Exp(m::Negate(m::Parameter(0)))))));
}

TEST_F(LogisticExpanderTest, DynamicDimensions) {
  constexpr std::string_view hlo = R"(
HloModule DynamicDimensions

ENTRY main {
  p = f32[<=10] parameter(0)
  ROOT root = f32[<=10] logistic(p)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  LogisticExpander logistic_expander;
  ASSERT_TRUE(logistic_expander.Run(module.get()).value());
  DynamicPadder dynamic_padder;
  EXPECT_TRUE(dynamic_padder.Run(module.get()).value());
}
}  // namespace
}  // namespace xla
