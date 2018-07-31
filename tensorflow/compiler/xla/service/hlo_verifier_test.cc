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

#include "tensorflow/compiler/xla/service/hlo_verifier.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

using HloVerifierTest = HloTestBase;

TEST_F(HloVerifierTest, NullInstructionParent) {
  HloComputation::Builder builder(TestName());
  const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  HloInstruction* negate = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape, HloOpcode::kNegate, param));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(verifier().Run(module.get()).status());

  negate->set_parent(nullptr);

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("has a null parent pointer"));
}

TEST_F(HloVerifierTest, NullComputationParent) {
  HloComputation::Builder builder(TestName());
  const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape, HloOpcode::kNegate, param));
  auto module = CreateNewModule();
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK(verifier().Run(module.get()).status());

  computation->set_parent(nullptr);

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), HasSubstr("has a null parent pointer"));
}

TEST_F(HloVerifierTest, DifferentOperandParents) {
  HloComputation::Builder builder(TestName());
  const Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  HloInstruction* negate = builder.AddInstruction(
      HloInstruction::CreateUnary(scalar_shape, HloOpcode::kNegate, param));
  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  HloComputation::Builder emb_builder(TestName());
  HloInstruction* emb_param = emb_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  module->AddEmbeddedComputation(emb_builder.Build());

  TF_ASSERT_OK(verifier().Run(module.get()).status());
  TF_ASSERT_OK(negate->ReplaceOperandWith(0, emb_param));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("is in a different computation"));
}

TEST_F(HloVerifierTest, ResetsShapeVerifierState) {
  HloComputation::Builder builder(TestName());
  Shape s1 = ShapeUtil::MakeShape(F32, {1});
  Shape s2 = ShapeUtil::MakeShape(F32, {2});

  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, s1, "param"));

  // Create an add instruction with the incorrect shape.
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(s2, HloOpcode::kAdd, param, param));

  // In order to trigger the bug we're checking for, the instruction with the
  // bad shape can't be the root of the computation.
  builder.AddInstruction(
      HloInstruction::CreateBinary(s2, HloOpcode::kMultiply, add, add));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  // Run the verifier twice.  It should fail both times, because it shouldn't
  // carry state in its DFS visitor between runs.
  EXPECT_FALSE(verifier().Run(module.get()).status().ok());
  EXPECT_FALSE(verifier().Run(module.get()).status().ok());
}

TEST_F(HloVerifierTest, CheckCallOperandParameterShapesMismatch) {
  const char* const hlo_string = R"(
HloModule Module

callme {
  ROOT param = (s32[], f32[4]) parameter(0)
}

ENTRY entry {
  p0 = (f32[4], s32[]) parameter(0)
  ROOT mycall = (s32[], f32[4]) call(p0), to_apply=callme
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("shape does not match parameter"));
}

TEST_F(HloVerifierTest, CheckConditionalOperandParameterShapesMismatch) {
  const char* const hlo_string = R"(
HloModule Module

true_branch {
  tparam = (s32[], f32[4]) parameter(0)
  ROOT tgte1 = f32[4] get-tuple-element(tparam), index=1
}

false_branch {
  fparam = (s32[], f32[4]) parameter(0)
  ROOT fgte1 = f32[4] get-tuple-element(fparam), index=1
}

ENTRY entry {
  p0 = (f32[4], s32[]) parameter(0)
  constant = pred[] constant(true)
  ROOT conditional = f32[4] conditional(constant, p0, p0),
    true_computation=true_branch, false_computation=false_branch
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseHloString(hlo_string));

  auto status = verifier().Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              HasSubstr("shape does not match parameter"));
}

}  // namespace
}  // namespace xla
