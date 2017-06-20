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

#include "tensorflow/compiler/xla/service/hlo_ordering.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

class HloOrderingTest : public HloTestBase {};

TEST_F(HloOrderingTest, LastUseScheduledFirst) {
  // Tests scheduling of the following HLO code:
  //
  //   %ab = abs(%param)
  //   %exp = exp(%param)
  //   %add = add(%ab, %exp)
  //   %negate = negate(%exp)
  //   %sub = subtract(%add, %negate)
  //
  // %add should be scheduled before %negate because %add is the last (and only)
  // use of %ab. Scheduling %add first then frees up %ab's buffer.
  const Shape vec = ShapeUtil::MakeShape(xla::F32, {42});
  auto builder = HloComputation::Builder(TestName());
  auto param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec, "param"));
  auto ab = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kAbs, param));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kExp, param));

  auto add = builder.AddInstruction(
      HloInstruction::CreateBinary(vec, HloOpcode::kAdd, ab, exp));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kNegate, exp));
  auto sub = builder.AddInstruction(
      HloInstruction::CreateBinary(vec, HloOpcode::kSubtract, add, negate));

  auto module = CreateNewModule();
  module->AddEntryComputation(builder.Build());

  TF_ASSIGN_OR_ASSERT_OK(
      SequentialHloOrdering::HloModuleSequence sequence,
      CreateMemoryMinimizingSequence(*module, [](const LogicalBuffer& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape());
      }));
  // Verify that all instructions are in the sequence.
  EXPECT_EQ(module->entry_computation()->instruction_count(),
            sequence.at(module->entry_computation()).size());

  // The first instruction should be the parameter and the last the root "sub".
  EXPECT_EQ(param, sequence.at(module->entry_computation()).front());
  EXPECT_EQ(sub, sequence.at(module->entry_computation()).back());

  SequentialHloOrdering ordering(module.get(), sequence);
  EXPECT_TRUE(ordering.ExecutesBefore(add, negate));
}

TEST_F(HloOrderingTest, InstructionsInDifferentComputations) {
  // Tests the ordering of instructions in different computations using the
  // following HLO code:
  //
  // Entry computation:
  //   %x = Call(A, {})
  //   %y = Call(B, {%x})
  //
  // Computation A:
  //   %a = Call(C, {})
  //
  // Computation B:
  //   %b = Call(C, {})
  //
  // Computation C:
  //   %c = Constant(42.0f)
  //
  // This results in a diamond-shaped callgraph.
  auto module = CreateNewModule();
  const Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});

  auto builder_c = HloComputation::Builder("C");
  HloInstruction* c = builder_c.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  HloComputation* computation_c =
      module->AddEmbeddedComputation(builder_c.Build());

  auto builder_b = HloComputation::Builder("B");
  builder_b.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param"));
  HloInstruction* b = builder_b.AddInstruction(
      HloInstruction::CreateCall(scalar_shape, {}, computation_c));
  HloComputation* computation_b =
      module->AddEmbeddedComputation(builder_b.Build());

  auto builder_a = HloComputation::Builder("A");
  HloInstruction* a = builder_a.AddInstruction(
      HloInstruction::CreateCall(scalar_shape, {}, computation_c));
  HloComputation* computation_a =
      module->AddEmbeddedComputation(builder_a.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* x = builder.AddInstruction(
      HloInstruction::CreateCall(scalar_shape, {}, computation_a));
  HloInstruction* y = builder.AddInstruction(
      HloInstruction::CreateCall(scalar_shape, {x}, computation_b));
  module->AddEntryComputation(builder.Build());

  DependencyHloOrdering ordering(module.get());
  EXPECT_TRUE(ordering.ExecutesBefore(x, y));
  EXPECT_FALSE(ordering.ExecutesBefore(y, x));

  EXPECT_TRUE(ordering.ExecutesBefore(a, b));
  EXPECT_FALSE(ordering.ExecutesBefore(b, a));

  EXPECT_FALSE(ordering.ExecutesBefore(a, x));
  EXPECT_TRUE(ordering.ExecutesBefore(a, y));
  EXPECT_FALSE(ordering.ExecutesBefore(x, a));
  EXPECT_FALSE(ordering.ExecutesBefore(y, a));

  EXPECT_FALSE(ordering.ExecutesBefore(b, x));
  EXPECT_FALSE(ordering.ExecutesBefore(b, y));
  EXPECT_TRUE(ordering.ExecutesBefore(x, b));
  EXPECT_FALSE(ordering.ExecutesBefore(y, b));

  // Instruction 'c' is called from multiple callsites and should be unordered
  // relative to all other instructions in the module.
  EXPECT_FALSE(ordering.ExecutesBefore(c, a));
  EXPECT_FALSE(ordering.ExecutesBefore(c, b));
  EXPECT_FALSE(ordering.ExecutesBefore(c, x));
  EXPECT_FALSE(ordering.ExecutesBefore(c, y));
  EXPECT_FALSE(ordering.ExecutesBefore(a, c));
  EXPECT_FALSE(ordering.ExecutesBefore(b, c));
  EXPECT_FALSE(ordering.ExecutesBefore(x, c));
  EXPECT_FALSE(ordering.ExecutesBefore(y, c));
}

class MinimumMemoryForSequenceTest : public HloTestBase {};

TEST_F(MinimumMemoryForSequenceTest, MultiComputation) {
  auto module = CreateNewModule();
  const Shape scalar_shape = ShapeUtil::MakeShape(xla::F32, {});
  const Shape tuple_shape =
      ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});

  auto cond_builder = HloComputation::Builder("WhileCond");
  // Tuple param: 24 bytes (each elem has 8 byte pointer, 4 byte element)
  HloInstruction* cond_param = cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  HloInstruction* cond_iter = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 0));
  HloInstruction* cond_data = cond_builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(scalar_shape, cond_param, 1));
  // Free cond_param[] (16 bytes), Alloc PRED[] (1 byte)
  HloInstruction* cond_lt = cond_builder.AddInstruction(
      HloInstruction::CreateBinary(ShapeUtil::MakeShape(PRED, {}),
                                   HloOpcode::kLt, cond_iter, cond_data));
  HloComputation* cond_computation =
      module->AddEmbeddedComputation(cond_builder.Build());

  auto body_builder = HloComputation::Builder("WhileBody");
  // Tuple param: 24 bytes (each elem has 8 byte pointer, 4 byte element)
  HloInstruction* body_param = body_builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  HloComputation* body_computation =
      module->AddEmbeddedComputation(body_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  // Entry params: 8 bytes (4 bytes per param), TOTAL=8
  HloInstruction* iter = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "param_iter"));
  HloInstruction* data = builder.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "param_data"));
  // Tuple: 16 bytes (8 bytes per pointer), TOTAL=24
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({iter, data}));
  // While: 8 bytes (4 bytes per element), TOTAL=32
  // Both cond and body use a max of 24 bytes, TOTAL=56
  HloInstruction* while_op = builder.AddInstruction(HloInstruction::CreateWhile(
      tuple_shape, cond_computation, body_computation, tuple));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  auto size_fn = [](const LogicalBuffer& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };

  SequentialHloOrdering::HloModuleSequence module_sequence;
  module_sequence[cond_computation] = {cond_param, cond_iter, cond_data,
                                       cond_lt};
  module_sequence[body_computation] = {body_param};
  module_sequence[entry_computation] = {iter, data, tuple, while_op};
  EXPECT_EQ(56,
            MinimumMemoryForSequence(module_sequence, size_fn).ValueOrDie());
}

}  // namespace

}  // namespace xla

int main(int argc, char** argv) {
  return xla::ParseDebugOptionsFlagsAndRunTests(argc, argv);
}
