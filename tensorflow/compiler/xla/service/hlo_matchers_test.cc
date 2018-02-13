/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace op = xla::testing::opcode_matchers;
using ::testing::_;
using ::testing::Eq;

namespace xla {
namespace {

string DescribeHloMatcher(const ::testing::Matcher<const HloInstruction*>& m) {
  std::stringstream ss;
  m.DescribeTo(&ss);
  return ss.str();
}

template <typename M, typename T>
string Explain(const T& t, const M& m) {
  ::testing::StringMatchResultListener listener;
  EXPECT_THAT(t, ::testing::Not(m));  // For the error message.
  EXPECT_FALSE(m.MatchAndExplain(t, &listener));
  return listener.str();
}

TEST(HloMatchersTest, Test) {
  auto shape = ShapeUtil::MakeShape(F32, {1});
  auto param = HloInstruction::CreateParameter(0, shape, "param");
  auto mul = HloInstruction::CreateBinary(shape, HloOpcode::kMultiply,
                                          param.get(), param.get());
  auto add = HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param.get(),
                                          mul.get());

  EXPECT_THAT(add.get(), op::Add());
  EXPECT_THAT(add.get(), op::Add(op::Parameter(), op::Multiply()));
  EXPECT_THAT(add.get(),
              op::Add(op::Parameter(), op::Multiply(_, op::Parameter())));

  // Negative matches: check the explanation string.
  EXPECT_THAT(Explain(add.get(), op::Parameter()), Eq(""));
  EXPECT_THAT(Explain(add.get(), op::Add(op::Parameter())),
              Eq("has too many operands (got 2, want 1)"));
  EXPECT_THAT(
      Explain(add.get(), op::Add(op::Parameter(), op::Parameter())),
      Eq("\noperand 1:\n\t"
         "%multiply = f32[1]{0} multiply(f32[1]{0} %param, f32[1]{0} %param)\n"
         "doesn't match expected:\n\t"
         "parameter"));
  EXPECT_THAT(
      Explain(add.get(),
              op::Add(op::Parameter(), op::Multiply(op::Add(), op::Add()))),
      Eq("\noperand 1:\n\t"
         "%multiply = f32[1]{0} multiply(f32[1]{0} %param, f32[1]{0} %param)\n"
         "doesn't match expected:\n\t"
         "multiply(add, add), \n"
         "operand 0:\n\t"
         "%param = f32[1]{0} parameter(0)\n"
         "doesn't match expected:\n\t"
         "add"));
}

TEST(HloMatchersTest, CustomCallMatcher) {
  auto c1 = HloInstruction::CreateConstant(Literal::CreateR1<float>({1, 2, 3}));
  auto c2 = HloInstruction::CreateConstant(Literal::CreateR1<int32>({1, 2, 3}));
  auto call = HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {1}), {c1.get(), c2.get()}, "foo_target");

  EXPECT_THAT(call.get(), op::CustomCall());
  EXPECT_THAT(call.get(), op::CustomCall(c1.get(), c2.get()));
  EXPECT_THAT(call.get(), op::CustomCall("foo_target"));
  EXPECT_THAT(call.get(), op::CustomCall("foo_target", c1.get(), c2.get()));
  EXPECT_THAT(call.get(), op::CustomCall(::testing::StartsWith("foo")));
  EXPECT_THAT(call.get(),
              op::CustomCall(::testing::Not(::testing::StartsWith("bar"))));

  // Wrong number of operands.
  EXPECT_THAT(call.get(), ::testing::Not(op::CustomCall(c1.get())));

  // Call target does not match.
  EXPECT_THAT(call.get(),
              ::testing::Not(op::CustomCall(::testing::StartsWith("bar"))));

  EXPECT_THAT(Explain(call.get(), op::CustomCall("bar")),
              R"(custom-call with call target that isn't equal to "bar")");
  EXPECT_THAT(DescribeHloMatcher(op::CustomCall("foo_target")),
              R"(custom-call with call target that is equal to "foo_target")");
}

}  // namespace
}  // namespace xla
