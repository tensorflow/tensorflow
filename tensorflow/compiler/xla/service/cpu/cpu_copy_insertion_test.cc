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

#include "tensorflow/compiler/xla/service/cpu/cpu_copy_insertion.h"

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

int64 CountCopies(const HloComputation& computation) {
  int64 count = 0;
  for (const auto& instruction : computation.instructions()) {
    if (instruction->opcode() == HloOpcode::kCopy) {
      count++;
    }
  }
  return count;
}

int64 CountCopies(const HloModule& module) {
  int64 count = 0;
  for (const auto& computation : module.computations()) {
    count += CountCopies(*computation);
  }
  return count;
}

class CpuCopyInsertionTest : public HloTestBase {
 protected:
  void InsertCopies(HloModule* module) {
    CpuCopyInsertion copy_insertion;
    ASSERT_IS_OK(copy_insertion.Run(module).status());
  }

  const Shape scalar_shape_ = ShapeUtil::MakeShape(F32, {});
};

TEST_F(CpuCopyInsertionTest, WhileBodyWithConstantRoot) {
  // Test a while body and condition which are each simply a constant (root of
  // computation is a constant). Each constant should be copied.
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param_0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param_0"));

  auto body_builder = HloComputation::Builder("body");
  body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  body_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(123.0)));
  HloComputation* body = module->AddEmbeddedComputation(body_builder.Build());

  auto cond_builder = HloComputation::Builder("condition");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
  HloComputation* condition =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto xla_while = builder.AddInstruction(
      HloInstruction::CreateWhile(scalar_shape_, condition, body, param_0));

  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*module), 3);

  EXPECT_THAT(xla_while->operand(0), op::Copy(op::Parameter()));
  EXPECT_THAT(body->root_instruction(), op::Copy(op::Constant()));
  EXPECT_THAT(condition->root_instruction(), op::Copy(op::Constant()));
}

TEST_F(CpuCopyInsertionTest, TupleCall) {
  // Test a kCall instruction which calls a computation which produces a three
  // element tuple: one is a constant, one is a parameter, and one is produced
  // in the computation. The constant and parameter should be copied.
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param_0"));
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape_, scalar_shape_, scalar_shape_});

  auto sub_builder = HloComputation::Builder("subcomputation");
  auto sub_param = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  auto constant = sub_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(123.0)));
  auto add = sub_builder.AddInstruction(HloInstruction::CreateBinary(
      scalar_shape_, HloOpcode::kAdd, sub_param, constant));
  sub_builder.AddInstruction(
      HloInstruction::CreateTuple({sub_param, constant, add}));
  HloComputation* subcomputation =
      module->AddEmbeddedComputation(sub_builder.Build());

  builder.AddInstruction(
      HloInstruction::CreateCall(tuple_shape, {param}, subcomputation));

  module->AddEntryComputation(builder.Build());

  InsertCopies(module.get());

  EXPECT_EQ(CountCopies(*subcomputation), 2);
  EXPECT_THAT(subcomputation->root_instruction(),
              op::Tuple(op::Copy(op::Parameter()), op::Copy(op::Constant()),
                        op::Add()));
}

}  // namespace
}  // namespace xla
