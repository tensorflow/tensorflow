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

#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

namespace xla {

class HloSubcomputationUnificationTest : public HloTestBase {
 protected:
  HloSubcomputationUnificationTest() {}

  std::unique_ptr<HloComputation> CreateR0S32IdentityComputation() {
    auto builder = HloComputation::Builder("Identity");
    builder.AddInstruction(HloInstruction::CreateParameter(0, r0s32_, "x"));
    return builder.Build();
  }

  std::unique_ptr<HloComputation> CreateR0S32AdditionComputation() {
    auto builder = HloComputation::Builder("Addition");
    auto x =
        builder.AddInstruction(HloInstruction::CreateParameter(0, r0s32_, "x"));
    auto y =
        builder.AddInstruction(HloInstruction::CreateParameter(1, r0s32_, "y"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(r0s32_, HloOpcode::kAdd, x, y));
    return builder.Build();
  }

  std::unique_ptr<HloComputation> CreateR1S32AdditionComputation(
      const Shape& shape) {
    auto builder = HloComputation::Builder("Addition");
    auto x =
        builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
    auto y =
        builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "y"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, x, y));
    return builder.Build();
  }

  Shape r0s32_ = ShapeUtil::MakeShape(S32, {});
  Shape r0f32_ = ShapeUtil::MakeShape(S32, {});
  Shape r1s32_5_ = ShapeUtil::MakeShape(S32, {5});
  Shape r1s32_3_ = ShapeUtil::MakeShape(S32, {3});
};

TEST_F(HloSubcomputationUnificationTest, UnifyIdentities) {
  auto module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());

  auto callee1 =
      module->AddEmbeddedComputation(CreateR0S32IdentityComputation());
  auto callee2 =
      module->AddEmbeddedComputation(CreateR0S32IdentityComputation());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(5)));
  auto x = builder.AddInstruction(
      HloInstruction::CreateCall(r0s32_, {constant}, callee1));
  auto y = builder.AddInstruction(
      HloInstruction::CreateCall(r0s32_, {constant}, callee2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0s32_, HloOpcode::kAdd, x, y));

  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, module->computation_count());
  EXPECT_NE(x->to_apply(), y->to_apply());
  if (VLOG_IS_ON(1)) {
    hlo_graph_dumper::DumpGraph(*module->entry_computation(),
                                "before unification",
                                module->config().debug_options());
  }
  EXPECT_TRUE(HloSubcomputationUnification().Run(module.get()).ValueOrDie());
  if (VLOG_IS_ON(1)) {
    hlo_graph_dumper::DumpGraph(*module->entry_computation(),
                                "after unification",
                                module->config().debug_options());
  }
  EXPECT_EQ(2, module->computation_count());
  EXPECT_EQ(x->to_apply(), y->to_apply());
}

TEST_F(HloSubcomputationUnificationTest, UnifyAdditions) {
  auto module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());

  auto callee1 =
      module->AddEmbeddedComputation(CreateR0S32AdditionComputation());
  auto callee2 =
      module->AddEmbeddedComputation(CreateR0S32AdditionComputation());

  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(5)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(3)));
  auto x = builder.AddInstruction(
      HloInstruction::CreateCall(r0s32_, {constant1, constant2}, callee1));
  auto y = builder.AddInstruction(
      HloInstruction::CreateCall(r0s32_, {constant1, constant2}, callee2));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0s32_, HloOpcode::kAdd, x, y));

  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, module->computation_count());
  EXPECT_NE(x->to_apply(), y->to_apply());
  if (VLOG_IS_ON(1)) {
    hlo_graph_dumper::DumpGraph(*module->entry_computation(),
                                "before unification",
                                module->config().debug_options());
  }
  EXPECT_TRUE(HloSubcomputationUnification().Run(module.get()).ValueOrDie());
  if (VLOG_IS_ON(1)) {
    hlo_graph_dumper::DumpGraph(*module->entry_computation(),
                                "after unification",
                                module->config().debug_options());
  }
  EXPECT_EQ(2, module->computation_count());
  EXPECT_EQ(x->to_apply(), y->to_apply());
}

// Do not unify subcomputations with different parameter shapes.
TEST_F(HloSubcomputationUnificationTest, DifferentParameterShapes) {
  auto module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());

  auto callee1 =
      module->AddEmbeddedComputation(CreateR1S32AdditionComputation(r1s32_5_));
  auto callee2 =
      module->AddEmbeddedComputation(CreateR1S32AdditionComputation(r1s32_3_));

  auto param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1s32_5_, "param1"));
  auto param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r1s32_5_, "param2"));
  auto x = builder.AddInstruction(
      HloInstruction::CreateCall(r1s32_5_, {param1, param1}, callee1));
  auto y = builder.AddInstruction(
      HloInstruction::CreateCall(r1s32_3_, {param2, param2}, callee2));
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(S32, {8}), {x, y}, 0));

  module->AddEntryComputation(builder.Build());

  EXPECT_EQ(3, module->computation_count());
  EXPECT_NE(x->to_apply(), y->to_apply());
  if (VLOG_IS_ON(1)) {
    hlo_graph_dumper::DumpGraph(*module->entry_computation(),
                                "before unification",
                                module->config().debug_options());
  }
  EXPECT_FALSE(HloSubcomputationUnification().Run(module.get()).ValueOrDie());
  if (VLOG_IS_ON(1)) {
    hlo_graph_dumper::DumpGraph(*module->entry_computation(),
                                "after unification",
                                module->config().debug_options());
  }
  EXPECT_EQ(3, module->computation_count());
  EXPECT_NE(x->to_apply(), y->to_apply());
}

// Regression test for b/31466798. Checks that entry_computation is still valid
// after unification.
TEST_F(HloSubcomputationUnificationTest, TwoIdenticalComputations) {
  auto module = CreateNewModule();
  for (int i = 0; i < 2; ++i) {
    HloComputation::Builder builder("pow");
    auto x =
        builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "x"));
    auto y =
        builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32_, "y"));
    builder.AddInstruction(
        HloInstruction::CreateBinary(r0f32_, HloOpcode::kPower, x, y));
    if (i == 0) {
      module->AddEmbeddedComputation(builder.Build());
    } else {
      module->AddEntryComputation(builder.Build());
    }
  }

  EXPECT_TRUE(HloSubcomputationUnification().Run(module.get()).ValueOrDie());
  EXPECT_EQ(1, module->computation_count());
  EXPECT_EQ(*module->computations().begin(), module->entry_computation());
}

}  // namespace xla
