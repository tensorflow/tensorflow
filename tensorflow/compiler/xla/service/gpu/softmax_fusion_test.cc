/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/softmax_fusion.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

using SoftmaxFusionTest = HloTestBase;

TEST_F(SoftmaxFusionTest, SingleSoftmaxPattern) {
  const std::string& hlo_string = R"(

HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[128,128]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[128]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[128,128]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[128,128]{1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxFusion fusion;
  EXPECT_TRUE(fusion.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  auto* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::CustomCall(m::Parameter(0))));
  ASSERT_TRUE(root->has_to_apply());
  // Assert that the softmax computation has exactly the softmax pattern.
  ASSERT_THAT(root->to_apply()->root_instruction(),
              GmockMatch(m::Subtract(
                  m::Parameter(0),
                  m::Broadcast(m::Reduce(m::Parameter(0), m::Constant())))));
}

TEST_F(SoftmaxFusionTest, SoftmaxPatternWithExtraStuff) {
  const std::string& hlo_string = R"(

HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[128,128]{1,0} parameter(0)
  exponential = f32[128,128]{1,0} exponential(param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[128]{0} reduce(exponential, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[128,128]{1,0} broadcast(reduce), dimensions={0}
  negate = f32[128,128]{1,0} negate(exponential)
  subtract = f32[128,128]{1,0} subtract(negate, broadcast)
  ROOT log = f32[128,128]{1,0} log(subtract)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxFusion fusion;
  EXPECT_TRUE(fusion.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  auto* root = module->entry_computation()->root_instruction();
  HloInstruction* custom_call;
  ASSERT_THAT(
      root,
      GmockMatch(m::Log(m::CustomCall(&custom_call, m::Exp(m::Parameter(0))))));
  ASSERT_TRUE(custom_call->has_to_apply());
  // Assert that the softmax computation has the softmax pattern with the extra
  // Negate, but without the Exponential.
  ASSERT_THAT(custom_call->to_apply()->root_instruction(),
              GmockMatch(m::Subtract(
                  m::Negate(m::Parameter(0)),
                  m::Broadcast(m::Reduce(m::Parameter(0), m::Constant())))));
}

TEST_F(SoftmaxFusionTest, DoubleSoftmaxPattern) {
  const std::string& hlo_string = R"(

HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT maximum = f32[] add(arg_0.1, arg_1.1)
}

ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
  exponential = f32[127,125]{1,0} exponential(subtract)
  constant_zero = f32[] constant(0)
  second_reduce = f32[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f32[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = f32[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxFusion fusion;
  EXPECT_TRUE(fusion.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  auto* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::CustomCall(m::Parameter(0))));
  ASSERT_TRUE(root->has_to_apply());
  // Assert that we have matched both softmax patterns.
  HloInstruction* exp1;
  HloInstruction* exp2;
  ASSERT_THAT(
      root->to_apply()->root_instruction(),
      GmockMatch(m::Divide(m::Exp(&exp1, m::Subtract()),
                           m::Broadcast(m::Reduce(m::Exp(&exp2, m::Subtract()),
                                                  m::Constant())))));
  EXPECT_EQ(exp1, exp2);
  ASSERT_THAT(exp1,
              GmockMatch(m::Exp(m::Subtract(
                  m::Parameter(0),
                  m::Broadcast(m::Reduce(m::Parameter(0), m::Constant()))))));
}

TEST_F(SoftmaxFusionTest, DoubleSoftmaxPatternWithExtraStuff) {
  const std::string& hlo_string = R"(

HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT maximum = f32[] add(arg_0.1, arg_1.1)
}

ENTRY main {
  param_0 = f32[64,128]{1,0} parameter(0)
  log = f32[64,128]{1,0} log(param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[64]{0} reduce(log, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[64,128]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[64,128]{1,0} subtract(log, broadcast)
  exponential = f32[64,128]{1,0} exponential(subtract)
  negate = f32[64,128]{1,0} negate(exponential)
  constant_zero = f32[] constant(0)
  second_reduce = f32[64]{0} reduce(negate, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f32[64,128]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = f32[64,128]{1,0} divide(negate, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxFusion fusion;
  EXPECT_TRUE(fusion.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  auto* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::CustomCall(m::Log(m::Parameter(0)))));
  // Assert that we have matched both softmax patterns.
  ASSERT_TRUE(root->has_to_apply());
  HloInstruction* neg1;
  HloInstruction* neg2;
  ASSERT_THAT(
      root->to_apply()->root_instruction(),
      GmockMatch(m::Divide(
          m::Negate(&neg1, m::Exp()),
          m::Broadcast(m::Reduce(m::Negate(&neg2, m::Exp()), m::Constant())))));
  EXPECT_EQ(neg1, neg2);
  ASSERT_THAT(neg1,
              GmockMatch(m::Negate(m::Exp(m::Subtract(
                  m::Parameter(0),
                  m::Broadcast(m::Reduce(m::Parameter(0), m::Constant())))))));
}

TEST_F(SoftmaxFusionTest, TripleSoftmaxPattern) {
  const std::string& hlo_string = R"(

HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT maximum = f32[] add(arg_0.1, arg_1.1)
}

ENTRY main {
  param_0 = f32[64,128]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[64]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[64,128]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[64,128]{1,0} subtract(param_0, broadcast)
  constant_zero = f32[] constant(0)
  second_reduce = f32[64]{0} reduce(subtract, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f32[64,128]{1,0} broadcast(second_reduce), dimensions={0}
  divide = f32[64,128]{1,0} divide(subtract, second_broadcast)
  third_reduce = f32[64]{0} reduce(divide, constant_zero), dimensions={1}, to_apply=add_computation
  third_broadcast = f32[64,128]{1,0} broadcast(third_reduce), dimensions={0}
  ROOT add = f32[64,128]{1,0} add(divide, third_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxFusion fusion;
  EXPECT_TRUE(fusion.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  auto* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::CustomCall(m::Parameter(0))));
  ASSERT_TRUE(root->has_to_apply());
  // Assert that we have matched all three softmax patterns.
  HloInstruction* div1;
  HloInstruction* div2;
  ASSERT_THAT(
      root->to_apply()->root_instruction(),
      GmockMatch(m::Add(m::Divide(&div1, m::Subtract(), m::Broadcast()),
                        m::Broadcast(m::Reduce(
                            m::Divide(&div2, m::Subtract(), m::Broadcast()),
                            m::Constant())))));
  EXPECT_EQ(div1, div2);
  HloInstruction* sub1;
  HloInstruction* sub2;
  ASSERT_THAT(div1, GmockMatch(m::Divide(
                        m::Subtract(&sub1, m::Parameter(0), m::Broadcast()),
                        m::Broadcast(m::Reduce(
                            m::Subtract(&sub2, m::Parameter(0), m::Broadcast()),
                            m::Constant())))));
  EXPECT_EQ(sub1, sub2);
  ASSERT_THAT(sub1,
              GmockMatch(m::Subtract(
                  m::Parameter(0),
                  m::Broadcast(m::Reduce(m::Parameter(0), m::Constant())))));
}

TEST_F(SoftmaxFusionTest, SingleSoftmaxPatternWrongLayout) {
  const std::string& hlo_string = R"(

HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[32,8,128,128]{0,1,2,3} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[32,8,128]{0,1,2} reduce(param_0, constant_neg_inf), dimensions={3}, to_apply=max_computation
  broadcast = f32[32,8,128,128]{0,1,2,3} broadcast(reduce), dimensions={0,1,2}
  ROOT subtract = f32[32,8,128,128]{0,1,2,3} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxFusion fusion;
  EXPECT_FALSE(fusion.Run(module.get()).value());
}

TEST_F(SoftmaxFusionTest, SingleSoftmaxPatternWrongReduceDimension) {
  const std::string& hlo_string = R"(

HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[32,8,128,128]{3,2,1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[32,8,128]{2,1,0} reduce(param_0, constant_neg_inf), dimensions={2}, to_apply=max_computation
  broadcast = f32[32,8,128,128]{3,2,1,0} broadcast(reduce), dimensions={0,1,2}
  ROOT subtract = f32[32,8,128,128]{3,2,1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxFusion fusion;
  EXPECT_FALSE(fusion.Run(module.get()).value());
}

TEST_F(SoftmaxFusionTest, SingleSoftmaxPatternWrongBroadcastDimension) {
  const std::string& hlo_string = R"(

HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[32,8,128,128]{3,2,1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[32,8,128]{2,1,0} reduce(param_0, constant_neg_inf), dimensions={3}, to_apply=max_computation
  broadcast = f32[32,8,128,128]{3,2,1,0} broadcast(reduce), dimensions={0,1,3}
  ROOT subtract = f32[32,8,128,128]{3,2,1,0} subtract(param_0, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxFusion fusion;
  EXPECT_FALSE(fusion.Run(module.get()).value());
}

TEST_F(SoftmaxFusionTest, SingleSoftmaxPatternExtraUsage) {
  const std::string& hlo_string = R"(

HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[32,8,128,128]{3,2,1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[32,8,128]{2,1,0} reduce(param_0, constant_neg_inf), dimensions={3}, to_apply=max_computation
  broadcast = f32[32,8,128,128]{3,2,1,0} broadcast(reduce), dimensions={0,1,2}
  subtract = f32[32,8,128,128]{3,2,1,0} subtract(param_0, broadcast)
  ROOT mul = f32[32,8,128,128]{3,2,1,0} multiply(subtract, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxFusion fusion;
  EXPECT_FALSE(fusion.Run(module.get()).value());
}

TEST_F(SoftmaxFusionTest, SingleSoftmaxPatternWithBinaryElementwiseSeparator) {
  const std::string& hlo_string = R"(

HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[32,8,128,128]{3,2,1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[32,8,128]{2,1,0} reduce(param_0, constant_neg_inf), dimensions={3}, to_apply=max_computation
  broadcast = f32[32,8,128,128]{3,2,1,0} broadcast(reduce), dimensions={0,1,2}
  add = f32[32,8,128,128]{3,2,1,0} add(param_0, param_0)
  ROOT subtract = f32[32,8,128,128]{3,2,1,0} subtract(add, broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxFusion fusion;
  EXPECT_FALSE(fusion.Run(module.get()).value());
}

TEST_F(SoftmaxFusionTest, DoubleSoftmaxPatternWithBinaryElementwiseSeparator) {
  const std::string& hlo_string = R"(

HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT maximum = f32[] add(arg_0.1, arg_1.1)
}

ENTRY main {
  param_0 = f32[64,128]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[64]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[64,128]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[64,128]{1,0} subtract(param_0, broadcast)
  exponential = f32[64,128]{1,0} exponential(subtract)
  add = f32[64,128]{1,0} add(exponential, exponential)
  constant_zero = f32[] constant(0)
  second_reduce = f32[64]{0} reduce(add, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f32[64,128]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = f32[64,128]{1,0} divide(add, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxFusion fusion;
  EXPECT_TRUE(fusion.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  auto* root = module->entry_computation()->root_instruction();
  HloInstruction* exp1;
  HloInstruction* exp2;
  // Assert that we have matched both softmax patterns, but they are in separate
  // custom calls.
  ASSERT_THAT(
      root, GmockMatch(m::CustomCall(m::Add(m::Exp(&exp1, m::CustomCall()),
                                            m::Exp(&exp2, m::CustomCall())))));
  EXPECT_EQ(exp1, exp2);
  ASSERT_THAT(exp1, GmockMatch(m::Exp(m::CustomCall(m::Parameter(0)))));
  ASSERT_TRUE(root->has_to_apply());
  ASSERT_THAT(root->to_apply()->root_instruction(),
              GmockMatch(m::Divide(
                  m::Parameter(0),
                  m::Broadcast(m::Reduce(m::Parameter(0), m::Constant())))));
  const HloInstruction* custom_call = exp1->operand(0);
  ASSERT_TRUE(custom_call->has_to_apply());
  ASSERT_THAT(custom_call->to_apply()->root_instruction(),
              GmockMatch(m::Subtract(
                  m::Parameter(0),
                  m::Broadcast(m::Reduce(m::Parameter(0), m::Constant())))));
}

TEST_F(SoftmaxFusionTest, DoubleSoftmaxPatternWithExtraProducerUsage) {
  const std::string& hlo_string = R"(

HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT maximum = f32[] add(arg_0.1, arg_1.1)
}

ENTRY main {
  param_0 = f32[64,128]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[64]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[64,128]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[64,128]{1,0} subtract(param_0, broadcast)
  exponential = f32[64,128]{1,0} exponential(subtract)
  constant_zero = f32[] constant(0)
  second_reduce = f32[64]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f32[64,128]{1,0} broadcast(second_reduce), dimensions={0}
  divide = f32[64,128]{1,0} divide(exponential, second_broadcast)
  ROOT add = f32[64,128]{1,0} add(divide, exponential)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxFusion fusion;
  EXPECT_TRUE(fusion.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  auto* root = module->entry_computation()->root_instruction();
  HloInstruction* exp1;
  HloInstruction* exp2;
  HloInstruction* custom_call;
  // Assert that we have matched both softmax patterns, but they are in separate
  // custom calls.
  ASSERT_THAT(root,
              GmockMatch(m::Add(
                  m::CustomCall(&custom_call, m::Exp(&exp1, m::CustomCall())),
                  m::Exp(&exp2, m::CustomCall()))));
  EXPECT_EQ(exp1, exp2);
  ASSERT_THAT(exp1, GmockMatch(m::Exp(m::CustomCall(m::Parameter(0)))));
  ASSERT_TRUE(custom_call->has_to_apply());
  ASSERT_THAT(custom_call->to_apply()->root_instruction(),
              GmockMatch(m::Divide(
                  m::Parameter(0),
                  m::Broadcast(m::Reduce(m::Parameter(0), m::Constant())))));
  const HloInstruction* custom_call2 = exp1->operand(0);
  ASSERT_TRUE(custom_call2->has_to_apply());
  ASSERT_THAT(custom_call2->to_apply()->root_instruction(),
              GmockMatch(m::Subtract(
                  m::Parameter(0),
                  m::Broadcast(m::Reduce(m::Parameter(0), m::Constant())))));
}

class SoftmaxFusionEnd2EndTest
    : public HloTestBase,
      public ::testing::WithParamInterface<::testing::tuple<int, int>> {
 public:
  void TestSoftmaxPattern(const std::string& hlo_string_template);

 private:
  DebugOptions GetDebugOptionsForTest() override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_softmax_fusion(true);
    return debug_options;
  }
};

void SoftmaxFusionEnd2EndTest::TestSoftmaxPattern(
    const std::string& hlo_string_template) {
  std::string hlo_string = absl::Substitute(
      hlo_string_template, std::get<0>(GetParam()), std::get<1>(GetParam()));
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec(1e-6, 1e-6)));
}

TEST_P(SoftmaxFusionEnd2EndTest, SingleSoftmaxPattern) {
  const std::string& hlo_string_template = R"(
HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f32[$0,$1]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[$0]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[$0,$1]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[$0,$1]{1,0} subtract(param_0, broadcast)
}
)";
  TestSoftmaxPattern(hlo_string_template);
}

TEST_P(SoftmaxFusionEnd2EndTest, DoubleSoftmaxPattern) {
  const std::string& hlo_string_template = R"(
HloModule softmax

max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}

add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT maximum = f32[] add(arg_0.1, arg_1.1)
}

ENTRY main {
  param_0 = f32[$0,$1]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[$0]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[$0,$1]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[$0,$1]{1,0} subtract(param_0, broadcast)
  exponential = f32[$0,$1]{1,0} exponential(subtract)
  constant_zero = f32[] constant(0)
  second_reduce = f32[$0]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f32[$0,$1]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = f32[$0,$1]{1,0} divide(exponential, second_broadcast)
}
)";
  TestSoftmaxPattern(hlo_string_template);
}

std::string TestDataToString(
    const ::testing::TestParamInfo<::testing::tuple<int, int>>& data) {
  return absl::StrCat(std::get<0>(data.param), "x", std::get<1>(data.param));
}

INSTANTIATE_TEST_SUITE_P(
    SoftmaxFusionTestSuite, SoftmaxFusionEnd2EndTest,
    ::testing::ValuesIn({std::make_tuple(1, 10),   std::make_tuple(10, 1),
                         std::make_tuple(2, 10),   std::make_tuple(10, 2),
                         std::make_tuple(32, 2),   std::make_tuple(32, 3),
                         std::make_tuple(32, 4),   std::make_tuple(32, 5),
                         std::make_tuple(32, 6),   std::make_tuple(32, 7),
                         std::make_tuple(32, 8),   std::make_tuple(32, 9),
                         std::make_tuple(32, 10),  std::make_tuple(32, 11),
                         std::make_tuple(32, 12),  std::make_tuple(32, 13),
                         std::make_tuple(32, 14),  std::make_tuple(32, 15),
                         std::make_tuple(32, 16),  std::make_tuple(32, 17),
                         std::make_tuple(32, 18),  std::make_tuple(127, 125),
                         std::make_tuple(128, 128)}),
    TestDataToString);

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
