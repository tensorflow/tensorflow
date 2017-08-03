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

}  // namespace
}  // namespace xla
