/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/variadic_op_splitter.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {
using match::Concatenate;

class VariadicOpSplitterTest : public HloTestBase {};

TEST_F(VariadicOpSplitterTest, DontSplit) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    p0 = f16[30,41] parameter(0)
    p1 = f16[30,41] parameter(1)
    ROOT result = f16[60, 41] concatenate(p0, p1), dimensions={0}
  })")
                    .value();
  EXPECT_FALSE(VariadicOpSplitter().Run(module.get()).value());
}

TEST_F(VariadicOpSplitterTest, SplitInto2) {
  auto builder = HloComputation::Builder(TestName());
  auto operand = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32_t>({42})));
  std::vector<HloInstruction*> concat_operands(255, operand);
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(S32, {255}), concat_operands, 0));
  auto module = CreateNewVerifiedModule();
  auto entry_computation = module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(VariadicOpSplitter().Run(module.get()).value());
  EXPECT_TRUE(Match(entry_computation->root_instruction(),
                    Concatenate().WithNumOperands(128).WithOperand(
                        0, Concatenate().WithNumOperands(128))));
}

TEST_F(VariadicOpSplitterTest, SplitInto3) {
  auto builder = HloComputation::Builder(TestName());
  auto operand = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<int32_t>({42})));
  std::vector<HloInstruction*> concat_operands(256, operand);
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(S32, {256}), concat_operands, 0));
  auto module = CreateNewVerifiedModule();
  auto entry_computation = module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(VariadicOpSplitter().Run(module.get()).value());
  EXPECT_TRUE(Match(entry_computation->root_instruction(),
                    Concatenate(Concatenate().WithNumOperands(128),
                                Concatenate().WithNumOperands(128))));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
