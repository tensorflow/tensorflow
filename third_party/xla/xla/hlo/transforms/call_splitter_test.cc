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

#include "xla/hlo/transforms/call_splitter.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/call_parameter_cleanup.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/call_inliner.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

class CallSplitterTest : public HloHardwareIndependentTestBase {
 protected:
  CallSplitterTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/true) {}
};

namespace {

namespace m = ::xla::match;

TEST_F(CallSplitterTest, SplitDownOneInstructionBasic) {
  const std::string module_str = R"hlo(
HloModule module

addmul {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  add = s32[] add(a, b)
  mul = s32[] multiply(add, c)
  ROOT tuple = (s32[]) tuple(mul)
}

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  ROOT call = (s32[]) call(p0, p1, p2), to_apply=addmul
}

)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  auto split = [](const HloInstruction* instruction) -> bool {
    return instruction->opcode() == HloOpcode::kMultiply;
  };

  CallSplitter splitter(/*call_predicate=*/HloPredicateTrue,
                        /*boundary_predicate=*/split);
  EXPECT_TRUE(splitter.Run(module.get()).value());

  CallParameterCleanup cleanup;
  HloDCE dce;
  TupleSimplifier tuple_simplifier;
  CHECK_OK(cleanup.Run(module.get()).status());
  CHECK_OK(dce.Run(module.get()).status());
  CHECK_OK(tuple_simplifier.Run(module.get()).status());

  // Verify we got the two-call structure, with the mul in the second call.
  HloInstruction* call1;
  HloInstruction* call2;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Call(&call2, m::Parameter(2),
                                 m::GetTupleElement(m::Call(&call1), 0))));
  EXPECT_THAT(call1->to_apply()->root_instruction(),
              GmockMatch(m::Tuple(m::Add())));
  EXPECT_THAT(call2->to_apply()->root_instruction(),
              GmockMatch(m::Tuple(m::Multiply())));

  // Verify we hooked up all the parameters correctly by simplifying again and
  // making sure it's equivalent to what we had in the beginning.
  CallInliner call_inliner;
  CHECK_OK(call_inliner.Run(module.get()).status());
  CHECK_OK(tuple_simplifier.Run(module.get()).status());

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Multiply(
                  m::Add(m::Parameter(0), m::Parameter(1)), m::Parameter(2)))));
}

TEST_F(CallSplitterTest, SplitDownOneInstructionIndependent) {
  const std::string module_str = R"hlo(
HloModule module

addmul {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  add = s32[] add(a, b)
  mul = s32[] multiply(b, c)
  ROOT tuple = (s32[], s32[]) tuple(add, mul)
}

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  ROOT call = (s32[], s32[]) call(p0, p1, p2), to_apply=addmul
}

)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  auto split = [](const HloInstruction* instruction) -> bool {
    return instruction->opcode() == HloOpcode::kMultiply;
  };

  CallSplitter splitter(/*call_predicate=*/HloPredicateTrue,
                        /*boundary_predicate=*/split);
  EXPECT_TRUE(splitter.Run(module.get()).value());

  CallParameterCleanup cleanup;
  HloDCE dce;
  TupleSimplifier tuple_simplifier;
  CHECK_OK(cleanup.Run(module.get()).status());
  CHECK_OK(dce.Run(module.get()).status());
  CHECK_OK(tuple_simplifier.Run(module.get()).status());

  HloInstruction* call1;
  HloInstruction* call2;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::GetTupleElement(m::Call(&call1), 0),
                                  m::GetTupleElement(m::Call(&call2), 0))));
  EXPECT_THAT(call1->to_apply()->root_instruction(),
              GmockMatch(m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)))));
  EXPECT_THAT(
      call2->to_apply()->root_instruction(),
      GmockMatch(m::Tuple(m::Multiply(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(CallSplitterTest, SplitDownMultipleInstructionsParallel) {
  const std::string module_str = R"hlo(
HloModule module

func {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  d = s32[] parameter(3)
  x = s32[] add(a, b)
  y = s32[] add(c, d)
  mul = s32[] multiply(x, x)
  sub = s32[] subtract(y, y)
  add = s32[] add(mul, sub)
  ROOT tuple = (s32[], s32[], s32[]) tuple(mul, sub, add)
}

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  ROOT call = (s32[], s32[], s32[]) call(p0, p1, p2, p3), to_apply=func
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  auto split = [](const HloInstruction* instruction) -> bool {
    return instruction->opcode() == HloOpcode::kMultiply ||
           instruction->opcode() == HloOpcode::kSubtract;
  };

  CallSplitter splitter(/*call_predicate=*/HloPredicateTrue,
                        /*boundary_predicate=*/split);
  EXPECT_TRUE(splitter.Run(module.get()).value());

  CallParameterCleanup cleanup;
  HloDCE dce;
  TupleSimplifier tuple_simplifier;
  CHECK_OK(cleanup.Run(module.get()).status());
  CHECK_OK(dce.Run(module.get()).status());
  CHECK_OK(tuple_simplifier.Run(module.get()).status());

  HloInstruction* call1;
  HloInstruction* call1_copy;
  HloInstruction* call2;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Call(&call2, m::GetTupleElement(m::Call(&call1), 0),
                                 m::GetTupleElement(m::Call(&call1_copy), 1))));
  EXPECT_EQ(call1, call1_copy);
  EXPECT_THAT(call1->to_apply()->root_instruction(),
              GmockMatch(m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)),
                                  m::Add(m::Parameter(2), m::Parameter(3)))));
}

TEST_F(CallSplitterTest, SplitDownMultipleInstructionsDependent) {
  const std::string module_str = R"hlo(
HloModule module

func {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  d = s32[] parameter(3)
  x = s32[] add(a, b)
  y = s32[] add(c, d)
  mul = s32[] multiply(x, x)
  sub = s32[] subtract(mul, y)
  ROOT tuple = (s32[], s32[]) tuple(mul, sub)
}

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  ROOT call = (s32[], s32[]) call(p0, p1, p2, p3), to_apply=func
}
)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  auto split = [](const HloInstruction* instruction) -> bool {
    return instruction->opcode() == HloOpcode::kMultiply ||
           instruction->opcode() == HloOpcode::kSubtract;
  };

  CallSplitter splitter(/*call_predicate=*/HloPredicateTrue,
                        /*boundary_predicate=*/split);
  EXPECT_TRUE(splitter.Run(module.get()).value());

  CallParameterCleanup cleanup;
  HloDCE dce;
  TupleSimplifier tuple_simplifier;
  CHECK_OK(cleanup.Run(module.get()).status());
  CHECK_OK(dce.Run(module.get()).status());
  CHECK_OK(tuple_simplifier.Run(module.get()).status());

  HloInstruction* call1;
  HloInstruction* call1_copy;
  HloInstruction* call2;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Call(&call2, m::GetTupleElement(m::Call(&call1), 0),
                                 m::GetTupleElement(m::Call(&call1_copy), 1))));
  EXPECT_EQ(call1, call1_copy);
  EXPECT_THAT(call1->to_apply()->root_instruction(),
              GmockMatch(m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)),
                                  m::Add(m::Parameter(2), m::Parameter(3)))));
  EXPECT_THAT(call2->to_apply()->root_instruction(),
              GmockMatch(m::Tuple(m::Multiply(), m::Subtract())));
}

TEST_F(CallSplitterTest, SplitDownMultipleCallsites) {
  const std::string module_str = R"hlo(
HloModule module

addmul {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  add = s32[] add(a, b)
  mul = s32[] multiply(add, c)
  ROOT tuple = (s32[]) tuple(mul)
}

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  call0 = (s32[]) call(p0, p1, p2), to_apply=addmul
  call1 = (s32[]) call(p2, p1, p0), to_apply=addmul
  ROOT out = tuple(call0, call1)
}

)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  auto split = [](const HloInstruction* instruction) -> bool {
    return instruction->opcode() == HloOpcode::kMultiply;
  };

  CallSplitter splitter(/*call_predicate=*/HloPredicateTrue,
                        /*boundary_predicate=*/split);
  EXPECT_TRUE(splitter.Run(module.get()).value());

  CallParameterCleanup cleanup;
  HloDCE dce;
  TupleSimplifier tuple_simplifier;
  CHECK_OK(cleanup.Run(module.get()).status());
  CHECK_OK(dce.Run(module.get()).status());
  CHECK_OK(tuple_simplifier.Run(module.get()).status());

  HloInstruction* call0_first;
  HloInstruction* call1_first;
  HloInstruction* call0_second;
  HloInstruction* call1_second;

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Call(&call0_second, m::Parameter(2),
                          m::GetTupleElement(m::Call(&call0_first), 0)),
                  m::Call(&call1_second, m::Parameter(0),
                          m::GetTupleElement(m::Call(&call1_first), 0)))));

  EXPECT_EQ(call0_first->to_apply(), call1_first->to_apply());
  EXPECT_EQ(call0_second->to_apply(), call1_second->to_apply());
  EXPECT_NE(call0_first->to_apply(), call0_second->to_apply());
}

TEST_F(CallSplitterTest, ClearCache) {
  const std::string module_str = R"hlo(
HloModule module

addrem {
  a = u32[] parameter(0)
  b = u32[] parameter(1)
  c = u32[] constant(8)
  add = u32[] add(a, b)
  rem = u32[] remainder(add, c)
  ROOT tuple = (u32[]) tuple(rem)
}

ENTRY entry {
  p0 = u32[] parameter(0)
  p1 = u32[] parameter(1)
  ROOT call = (u32[]) call(p0, p1), to_apply=addrem
}

)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  auto split = [](const HloInstruction* instruction) -> bool {
    return instruction->opcode() == HloOpcode::kAnd;
  };

  CallSplitter splitter(/*call_predicate=*/HloPredicateTrue,
                        /*boundary_predicate=*/split);
  EXPECT_FALSE(splitter.Run(module.get()).value());

  AlgebraicSimplifier simplifier(AlgebraicSimplifierOptions{});
  CHECK_OK(simplifier.Run(module.get()).status());

  EXPECT_TRUE(splitter.Run(module.get()).value());
}

}  // namespace

}  // namespace xla
