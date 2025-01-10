/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/map_inliner.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla_data.pb.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace {

using MapInlinerTest = HloTestBase;

// Test that `map` with `max` is transformed to `max`
TEST_F(MapInlinerTest, MapMax) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});

  auto max_builder = HloComputation::Builder(TestName());
  auto param1 = max_builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "x"));
  auto param2 = max_builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32, "y"));
  max_builder.AddInstruction(HloInstruction::CreateBinary(
      param1->shape(), HloOpcode::kMaximum, param1, param2));
  auto max_f32 = max_builder.Build();

  auto builder = HloComputation::Builder("MapMaxFunction");
  auto lhs = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1, 2, 3, 4})));
  auto rhs = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({4, 3, 2, 1})));
  builder.AddInstruction(
      HloInstruction::CreateMap(lhs->shape(), {lhs, rhs}, max_f32.get()));

  auto computation = builder.Build();
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEmbeddedComputation(std::move(max_f32));
  hlo_module->AddEntryComputation(std::move(computation));

  MapInliner inliner;
  EXPECT_TRUE(inliner.Run(hlo_module.get()).value());
  EXPECT_THAT(hlo_module->entry_computation()->root_instruction(),
              op::Maximum(lhs, rhs));

  // Verify execution on CPU.
  auto result = ExecuteAndTransfer(hlo_module->Clone(), {});
  auto expected = LiteralUtil::CreateR1<float>({4, 3, 3, 4});
  EXPECT_TRUE(LiteralTestUtil::Equal(result, expected));
}

// Test that `constant` function is changed to `broadcast`.
TEST_F(MapInlinerTest, MapConstant) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});

  auto const2_builder = HloComputation::Builder(TestName());
  auto param1 = const2_builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "x"));
  (void)param1;
  const2_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  auto const2_f32 = const2_builder.Build();

  auto builder = HloComputation::Builder("MapConstFunction");
  auto lhs = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{1, 2, 3, 4}, {5, 6, 7, 8}})));
  builder.AddInstruction(
      HloInstruction::CreateMap(lhs->shape(), {lhs}, const2_f32.get()));

  auto computation = builder.Build();
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEmbeddedComputation(std::move(const2_f32));
  hlo_module->AddEntryComputation(std::move(computation));
  HloInstruction* root = hlo_module->entry_computation()->root_instruction();
  MapInliner inliner;
  EXPECT_TRUE(inliner.Run(hlo_module.get()).value());
  root = hlo_module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Broadcast(op::Constant()));

  // Verify execution on CPU.
  auto result = ExecuteAndTransfer(hlo_module->Clone(), {});
  auto expected = LiteralUtil::CreateR2<float>({{2, 2, 2, 2}, {2, 2, 2, 2}});
  EXPECT_TRUE(LiteralTestUtil::Equal(result, expected));
}

TEST_F(MapInlinerTest, MapSubtractOppositeOrder) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});

  // Note that the parameter ordinals are in the opposite order to their
  // position as operands
  auto max_builder = HloComputation::Builder(TestName());
  auto param1 = max_builder.AddInstruction(
          HloInstruction::CreateParameter(1, r0f32, "x"));
  auto param2 = max_builder.AddInstruction(
          HloInstruction::CreateParameter(0, r0f32, "y"));
  max_builder.AddInstruction(HloInstruction::CreateBinary(
          param1->shape(), HloOpcode::kSubtract, param1, param2));
  auto max_f32 = max_builder.Build();

  auto builder = HloComputation::Builder("MapSubFunction");
  auto lhs = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({1, 2, 3, 4})));
  auto rhs = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR1<float>({4, 3, 2, 1})));
  builder.AddInstruction(
    HloInstruction::CreateMap(lhs->shape(), {lhs, rhs}, max_f32.get()));

  auto computation = builder.Build();
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEmbeddedComputation(std::move(max_f32));
  hlo_module->AddEntryComputation(std::move(computation));

  MapInliner inliner;
  EXPECT_TRUE(inliner.Run(hlo_module.get()).value());
  EXPECT_THAT(hlo_module->entry_computation()->root_instruction(),
          op::Subtract(rhs, lhs));

  // Verify execution on CPU.
  auto result = ExecuteAndTransfer(hlo_module->Clone(), {});
  auto expected = LiteralUtil::CreateR1<float>({3, 1, -1, -3});
  EXPECT_TRUE(LiteralTestUtil::Equal(result, expected));
}

TEST_F(MapInlinerTest, MapParameter) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});

  auto param_builder = HloComputation::Builder(TestName());
  param_builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32, "p0"));
  param_builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32, "p1"));
  auto param_f32 = param_builder.Build();

  auto builder = HloComputation::Builder("MapParamFunction");
  auto lhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1)));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(4)));
  builder.AddInstruction(
      HloInstruction::CreateMap(lhs->shape(), {lhs, rhs}, param_f32.get()));

  auto computation = builder.Build();
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEmbeddedComputation(std::move(param_f32));
  hlo_module->AddEntryComputation(std::move(computation));

  MapInliner inliner;
  EXPECT_TRUE(inliner.Run(hlo_module.get()).value());
  EXPECT_THAT(hlo_module->entry_computation()->root_instruction(), rhs);

  // Verify execution on CPU.
  auto result = ExecuteAndTransfer(hlo_module->Clone(), {});
  auto expected = LiteralUtil::CreateR0<float>(4);
  EXPECT_TRUE(LiteralTestUtil::Equal(result, expected));
}

}  // namespace
}  // namespace xla
