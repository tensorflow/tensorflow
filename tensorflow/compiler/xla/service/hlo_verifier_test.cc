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

}  // namespace
}  // namespace xla
