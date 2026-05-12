/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/evaluator/hlo_evaluator_interpreter.h"

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

TEST(LinearizedInterpreterTest, BuildSimple) {
  Shape shape = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder b("AddComputation");
  auto param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param0"));
  auto param1 =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "param1"));
  b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  auto computation = b.Build();

  static Literal dummy_literal = LiteralUtil::CreateR0<float>(0.0f);
  auto resolver = [](const HloInstruction* instr) -> const Literal& {
    return dummy_literal;
  };
  absl::flat_hash_map<int, const HloInstruction*> param_to_operand;

  auto interpreter_or = LinearizedInterpreter::Build(
      computation.get(), {}, resolver,
      LinearizedInterpreter::GetDefaultOpRegistry(), param_to_operand);

  TF_ASSERT_OK(interpreter_or.status());
  EXPECT_NE(interpreter_or.value(), nullptr);
}

TEST(LinearizedInterpreterTest, ExecuteSimple) {
  Shape shape = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder b("AddComputation");
  auto param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param0"));
  auto param1 =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "param1"));
  b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  auto computation = b.Build();

  static Literal dummy_literal = LiteralUtil::CreateR0<float>(0.0f);
  auto resolver = [](const HloInstruction* instr) -> const Literal& {
    return dummy_literal;
  };
  absl::flat_hash_map<int, const HloInstruction*> param_to_operand;

  TF_ASSERT_OK_AND_ASSIGN(
      auto interpreter,
      LinearizedInterpreter::Build(
          computation.get(), {}, resolver,
          LinearizedInterpreter::GetDefaultOpRegistry(), param_to_operand));

  auto scratchpad = interpreter->CreateScratchpad();
  void* base = scratchpad.data();

  const auto& param_slots = interpreter->param_slots();
  const auto& result_slots = interpreter->result_slots();

  ASSERT_EQ(param_slots.size(), 2);
  ASSERT_TRUE(param_slots[0].has_value());
  ASSERT_TRUE(param_slots[1].has_value());
  ASSERT_EQ(result_slots.size(), 1);

  float* p0 = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      base, param_slots[0]->offset);
  float* p1 = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      base, param_slots[1]->offset);
  float* res = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      base, result_slots[0].offset);

  *p0 = 1.0f;
  *p1 = 2.0f;

  interpreter->ExecuteSteps(scratchpad);

  EXPECT_EQ(*res, 3.0f);
}

TEST(LinearizedInterpreterTest, InvalidBatchSize) {
  Shape shape = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder b("AddComputation");
  auto param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "param0"));
  auto param1 =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "param1"));
  b.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  auto computation = b.Build();

  static Literal dummy_literal = LiteralUtil::CreateR0<float>(0.0f);
  auto resolver = [](const HloInstruction* instr) -> const Literal& {
    return dummy_literal;
  };
  absl::flat_hash_map<int, const HloInstruction*> param_to_operand;

  auto promotion_policy =
      [](const HloInstruction* instr,
         const absl::flat_hash_map<const HloInstruction*, PrimitiveType>&) {
        return instr->shape().element_type();
      };

  auto interpreter_or1 = LinearizedInterpreter::Build(
      computation.get(), {}, resolver,
      LinearizedInterpreter::GetDefaultOpRegistry(), param_to_operand,
      promotion_policy, 0);
  EXPECT_FALSE(interpreter_or1.ok());

  auto interpreter_or2 = LinearizedInterpreter::Build(
      computation.get(), {}, resolver,
      LinearizedInterpreter::GetDefaultOpRegistry(), param_to_operand,
      promotion_policy, LinearizedInterpreter::kMaxBatchSize + 1);
  EXPECT_FALSE(interpreter_or2.ok());
}

}  // namespace
}  // namespace xla
