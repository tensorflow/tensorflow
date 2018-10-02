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

#include "tensorflow/compiler/xla/service/hlo_reachability.h"

#include <set>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"

namespace xla {

namespace {

class HloReachabilityTest : public HloVerifiedTestBase {};

TEST_F(HloReachabilityTest, Reachability) {
  // Construct and test a reachability graph of the following form:
  /*
       a
      / \
     b   c
      \ / \
       d   e
  */
  auto builder = HloComputation::Builder(TestName());
  auto a = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto b = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto c = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto d = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto e = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  builder.Build();

  HloReachabilityMap reachability({a, b, c, d, e});
  reachability.SetReachable(a, a);
  EXPECT_TRUE(reachability.SetReachabilityToUnion({a}, b));
  EXPECT_TRUE(reachability.SetReachabilityToUnion({a}, c));
  EXPECT_TRUE(reachability.SetReachabilityToUnion({b, c}, d));
  EXPECT_TRUE(reachability.SetReachabilityToUnion({c}, e));

  EXPECT_TRUE(reachability.IsReachable(a, a));
  EXPECT_TRUE(reachability.IsReachable(a, b));
  EXPECT_TRUE(reachability.IsReachable(a, c));
  EXPECT_TRUE(reachability.IsReachable(a, d));
  EXPECT_TRUE(reachability.IsReachable(a, e));

  EXPECT_FALSE(reachability.IsReachable(b, a));
  EXPECT_TRUE(reachability.IsReachable(b, b));
  EXPECT_FALSE(reachability.IsReachable(b, c));
  EXPECT_TRUE(reachability.IsReachable(b, d));
  EXPECT_FALSE(reachability.IsReachable(b, e));

  EXPECT_FALSE(reachability.IsReachable(e, a));
  EXPECT_FALSE(reachability.IsReachable(e, b));
  EXPECT_FALSE(reachability.IsReachable(e, c));
  EXPECT_FALSE(reachability.IsReachable(e, d));
  EXPECT_TRUE(reachability.IsReachable(e, e));

  // Recomputing the same reachability for a previously computed instruction
  // should return false (no change).
  EXPECT_FALSE(reachability.SetReachabilityToUnion({a}, b));
  EXPECT_FALSE(reachability.SetReachabilityToUnion({b, c}, d));
}

}  // namespace

}  // namespace xla
