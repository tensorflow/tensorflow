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

#include "tensorflow/compiler/xla/service/reduce_precision_insertion.h"

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

namespace op = xla::testing::opcode_matchers;

namespace xla {

using ::testing::UnorderedElementsAre;

class ReducePrecisionInsertionTest : public HloTestBase {
 protected:
  bool InsertOps(HloModule* module,
                 const std::function<bool(HloOpcode)>& filter) {
    ReducePrecisionInsertion op_insertion(5, 10, filter);
    StatusOr<bool> result = op_insertion.Run(module);
    EXPECT_IS_OK(result.status());
    return result.ValueOrDie();
  }
};

TEST_F(ReducePrecisionInsertionTest, RootInstruction) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  // Create a simple graph with a parameter feeding a unary cosine function.
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, a));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected state before adding ops.
  EXPECT_EQ(computation->root_instruction(), b);

  EXPECT_TRUE(InsertOps(module.get(),
                        [](HloOpcode h) { return h == HloOpcode::kCos; }));

  // Confirm expected graph after adding ops.
  EXPECT_THAT(computation->root_instruction(), op::ReducePrecision());
  EXPECT_EQ(computation->root_instruction()->operand(0), b);
}

TEST_F(ReducePrecisionInsertionTest, NonRootInstruction) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  // Create a graph with two parameters feeding into unary cosine functions,
  // and the output of those feeds into an add function.  Feeding the outputs
  // from the suffixed cosine functions into a binary add function allows us to
  // confirm that the separate operand streams are not crossed when the new
  // instructions are inserted.
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* a_cos = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, a));

  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* b_cos = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, b));

  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a_cos, b_cos));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  // Confirm expected graph before adding ops.
  EXPECT_EQ(c->operand(0), a_cos);
  EXPECT_EQ(c->operand(1), b_cos);

  EXPECT_TRUE(InsertOps(module.get(),
                        [](HloOpcode h) { return h == HloOpcode::kCos; }));

  // Confirm expected graph after adding ops.
  EXPECT_THAT(c->operand(0), op::ReducePrecision());
  EXPECT_EQ(c->operand(0)->operand(0), a_cos);
  EXPECT_THAT(c->operand(1), op::ReducePrecision());
  EXPECT_EQ(c->operand(1)->operand(0), b_cos);
}

TEST_F(ReducePrecisionInsertionTest, OutputIsNotFloat) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, x));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected graph before adding ops.
  EXPECT_THAT(x->users(), UnorderedElementsAre(y));
  EXPECT_EQ(computation->root_instruction(), y);

  // Since none of the instructions produce F32 data, this should not change
  // the graph.
  EXPECT_FALSE(InsertOps(module.get(), [](HloOpcode) { return true; }));

  // Confirm that graph has not changed.
  EXPECT_THAT(x->users(), UnorderedElementsAre(y));
  EXPECT_EQ(computation->root_instruction(), y);
}

TEST_F(ReducePrecisionInsertionTest, ShouldReduceOutputPrecisionIsFalse) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* x =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kCos, x));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected graph before adding ops.
  EXPECT_THAT(x->users(), UnorderedElementsAre(y));
  EXPECT_EQ(computation->root_instruction(), y);

  // Since none of the instructions match the should_reduce_output_precision
  // function, this should not change the graph.
  EXPECT_FALSE(InsertOps(module.get(), [](HloOpcode h) { return false; }));

  // Confirm that graph has not changed.
  EXPECT_THAT(x->users(), UnorderedElementsAre(y));
  EXPECT_EQ(computation->root_instruction(), y);
}

TEST_F(ReducePrecisionInsertionTest, InsertionIsNotRecursive) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateReducePrecision(shape, a, 9, 23));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  // Confirm expected state before adding ops.
  EXPECT_EQ(computation->root_instruction(), b);

  // This should insert a new ReducePrecision after the existing one, but
  // should not then recurse by adding another after the just-inserted one.
  EXPECT_TRUE(InsertOps(module.get(), [](HloOpcode h) {
    return h == HloOpcode::kReducePrecision;
  }));

  // Confirm expected graph after adding ops.
  EXPECT_THAT(computation->root_instruction(), op::ReducePrecision());
  EXPECT_EQ(computation->root_instruction()->operand(0), b);
}

}  // namespace xla

int main(int argc, char** argv) {
  return xla::ParseDebugOptionsFlagsAndRunTests(argc, argv);
}
