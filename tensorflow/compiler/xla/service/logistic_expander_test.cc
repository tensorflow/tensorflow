/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/logistic_expander.h"

#include <memory>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace m = match;

class LogisticExpanderTest : public HloTestBase {};

// Test that we expand kLogistic with 0.5 + 0.5 * tanh(0.5*x) when the proper
// option is enabled.
TEST_F(LogisticExpanderTest, ExpandWithTanh) {
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
  LogisticExpander logistic_expander(LogisticExpansionType::kTanh);
  ASSERT_TRUE(logistic_expander.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::AddAnyOrder(
                  m::MultiplyAnyOrder(m::Broadcast(m::ConstantScalar(0.5)),
                                      m::Tanh(m::MultiplyAnyOrder(
                                          m::Broadcast(m::ConstantScalar(0.5)),
                                          m::Parameter(0)))),
                  m::Broadcast(m::ConstantScalar(0.5)))));
}

// Test that we expand kLogistic with 1.0 / (1.0 + exp(-x)) when the proper
// option is enabled.
TEST_F(LogisticExpanderTest, ExpandWithEXP) {
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
  LogisticExpander logistic_expander(LogisticExpansionType::kExp);
  ASSERT_TRUE(logistic_expander.Run(m.get()).ValueOrDie());
  root = computation->root_instruction();
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Divide(
                  m::Broadcast(m::ConstantScalar(1.0)),
                  m::AddAnyOrder(m::Broadcast(m::ConstantScalar(1.0)),
                                 m::Exp(m::Negate(m::Parameter(0)))))));
}

}  // namespace
}  // namespace xla
