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

#include "xla/hlo/evaluator/hlo_evaluator_interpreter_ops.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "xla/comparison_util.h"
#include "xla/hlo/evaluator/hlo_evaluator_interpreter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(HloEvaluatorInterpreterOpsTest, GetDefaultOpRegistry) {
  const auto& registry = LinearizedInterpreter::GetDefaultOpRegistry();

  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  auto dummy_param = HloInstruction::CreateParameter(0, shape, "param");
  auto add_instr = HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, dummy_param.get(), dummy_param.get());

  LinearizedInterpreter::Step step;
  step.opcode = HloOpcode::kAdd;
  step.type = F32;
  step.operand_types = {F32, F32};

  auto status = registry.Populate(step, add_instr.get(), F32);
  EXPECT_TRUE(status.ok());
  EXPECT_NE(step.execute_fn, nullptr);
}

TEST(HloEvaluatorInterpreterOpsTest, AddOp) {
  LinearizedInterpreter::Step step;
  step.element_count = 4;
  step.result_offset = 0;
  step.operand_offsets = {sizeof(float) * 4, sizeof(float) * 8};

  alignas(float) char buffer[sizeof(float) * 12];
  float* result = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      buffer, step.result_offset);
  float* lhs = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      buffer, step.operand_offsets[0]);
  float* rhs = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      buffer, step.operand_offsets[1]);

  lhs[0] = 1.0f;
  lhs[1] = 2.0f;
  lhs[2] = 3.0f;
  lhs[3] = 4.0f;
  rhs[0] = 10.0f;
  rhs[1] = 20.0f;
  rhs[2] = 30.0f;
  rhs[3] = 40.0f;

  LinearizedInterpreter::Ops::Add::Execute<float, float, float>(&step, buffer);

  EXPECT_EQ(result[0], 11.0f);
  EXPECT_EQ(result[1], 22.0f);
  EXPECT_EQ(result[2], 33.0f);
  EXPECT_EQ(result[3], 44.0f);
}

TEST(HloEvaluatorInterpreterOpsTest, MaximumOp) {
  LinearizedInterpreter::Step step;
  step.element_count = 4;
  step.result_offset = 0;
  step.operand_offsets = {sizeof(float) * 4, sizeof(float) * 8};

  alignas(float) char buffer[sizeof(float) * 12];
  float* result = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      buffer, step.result_offset);
  float* lhs = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      buffer, step.operand_offsets[0]);
  float* rhs = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      buffer, step.operand_offsets[1]);

  lhs[0] = 1.0f;
  lhs[1] = 2.0f;
  lhs[2] = 3.0f;
  lhs[3] = std::numeric_limits<float>::quiet_NaN();
  rhs[0] = 10.0f;
  rhs[1] = -20.0f;
  rhs[2] = std::numeric_limits<float>::quiet_NaN();
  rhs[3] = 40.0f;

  LinearizedInterpreter::Ops::Maximum::Execute<float>(&step, buffer);

  EXPECT_EQ(result[0], 10.0f);
  EXPECT_EQ(result[1], 2.0f);
  EXPECT_TRUE(std::isnan(result[2]));
  EXPECT_TRUE(std::isnan(result[3]));
}

TEST(HloEvaluatorInterpreterOpsTest, CompareOp) {
  LinearizedInterpreter::Step step;
  step.element_count = 2;

  alignas(int32_t) char buffer[sizeof(int32_t) * 4 + sizeof(bool) * 2];
  size_t lhs_offset = 0;
  size_t rhs_offset = sizeof(int32_t) * 2;
  size_t result_offset = sizeof(int32_t) * 4;

  step.operand_offsets = {lhs_offset, rhs_offset};
  step.result_offset = result_offset;

  int32_t* lhs = LinearizedInterpreter::Scratchpad::GetPointerFromBase<int32_t>(
      buffer, lhs_offset);
  int32_t* rhs = LinearizedInterpreter::Scratchpad::GetPointerFromBase<int32_t>(
      buffer, rhs_offset);
  bool* result = LinearizedInterpreter::Scratchpad::GetPointerFromBase<bool>(
      buffer, result_offset);

  lhs[0] = 5;
  lhs[1] = 10;
  rhs[0] = 5;
  rhs[1] = 5;

  LinearizedInterpreter::Ops::Compare::Execute<int32_t,
                                               ComparisonDirection::kEq>(
      &step, buffer);
  EXPECT_TRUE(result[0]);
  EXPECT_FALSE(result[1]);

  LinearizedInterpreter::Ops::Compare::Execute<int32_t,
                                               ComparisonDirection::kGt>(
      &step, buffer);
  EXPECT_FALSE(result[0]);
  EXPECT_TRUE(result[1]);
}

TEST(HloEvaluatorInterpreterOpsTest, OrOp) {
  LinearizedInterpreter::Step step;
  step.element_count = 4;
  step.result_offset = 0;
  step.operand_offsets = {sizeof(bool) * 4, sizeof(bool) * 8};

  alignas(bool) char buffer[sizeof(bool) * 12];
  bool* result = LinearizedInterpreter::Scratchpad::GetPointerFromBase<bool>(
      buffer, step.result_offset);
  bool* lhs = LinearizedInterpreter::Scratchpad::GetPointerFromBase<bool>(
      buffer, step.operand_offsets[0]);
  bool* rhs = LinearizedInterpreter::Scratchpad::GetPointerFromBase<bool>(
      buffer, step.operand_offsets[1]);

  lhs[0] = false;
  lhs[1] = false;
  lhs[2] = true;
  lhs[3] = true;
  rhs[0] = false;
  rhs[1] = true;
  rhs[2] = false;
  rhs[3] = true;

  LinearizedInterpreter::Ops::Or::Execute(&step, buffer);

  EXPECT_FALSE(result[0]);
  EXPECT_TRUE(result[1]);
  EXPECT_TRUE(result[2]);
  EXPECT_TRUE(result[3]);
}

TEST(HloEvaluatorInterpreterOpsTest, AndOp) {
  LinearizedInterpreter::Step step;
  step.element_count = 4;
  step.result_offset = 0;
  step.operand_offsets = {sizeof(bool) * 4, sizeof(bool) * 8};

  alignas(bool) char buffer[sizeof(bool) * 12];
  bool* result = LinearizedInterpreter::Scratchpad::GetPointerFromBase<bool>(
      buffer, step.result_offset);
  bool* lhs = LinearizedInterpreter::Scratchpad::GetPointerFromBase<bool>(
      buffer, step.operand_offsets[0]);
  bool* rhs = LinearizedInterpreter::Scratchpad::GetPointerFromBase<bool>(
      buffer, step.operand_offsets[1]);

  lhs[0] = false;
  lhs[1] = false;
  lhs[2] = true;
  lhs[3] = true;
  rhs[0] = false;
  rhs[1] = true;
  rhs[2] = false;
  rhs[3] = true;

  LinearizedInterpreter::Ops::And::Execute(&step, buffer);

  EXPECT_FALSE(result[0]);
  EXPECT_FALSE(result[1]);
  EXPECT_FALSE(result[2]);
  EXPECT_TRUE(result[3]);
}

TEST(HloEvaluatorInterpreterOpsTest, SelectOp) {
  LinearizedInterpreter::Step step;
  step.element_count = 2;

  size_t lhs_offset = 0;
  size_t rhs_offset = sizeof(float) * 2;
  size_t result_offset = rhs_offset + sizeof(float) * 2;
  size_t cond_offset = result_offset + sizeof(float) * 2;

  step.operand_offsets = {cond_offset, lhs_offset, rhs_offset};
  step.result_offset = result_offset;

  alignas(float) char buffer[sizeof(float) * 6 + sizeof(bool) * 2];

  bool* cond = LinearizedInterpreter::Scratchpad::GetPointerFromBase<bool>(
      buffer, cond_offset);
  float* lhs = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      buffer, lhs_offset);
  float* rhs = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      buffer, rhs_offset);
  float* result = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      buffer, result_offset);

  cond[0] = true;
  cond[1] = false;

  lhs[0] = 1.0f;
  lhs[1] = 2.0f;

  rhs[0] = 10.0f;
  rhs[1] = 20.0f;

  LinearizedInterpreter::Ops::Select::Execute<float>(&step, buffer);

  EXPECT_EQ(result[0], 1.0f);
  EXPECT_EQ(result[1], 20.0f);
}

TEST(HloEvaluatorInterpreterOpsTest, ConstantOp) {
  const auto& registry = LinearizedInterpreter::GetDefaultOpRegistry();

  Shape shape = ShapeUtil::MakeShape(F32, {2});
  Literal literal = LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  auto dummy_const = HloInstruction::CreateConstant(std::move(literal));

  LinearizedInterpreter::Step step;
  step.opcode = HloOpcode::kConstant;
  step.type = F32;
  step.element_count = 2;
  step.batch_size = 1;
  step.result_offset = 0;

  auto status = registry.Populate(step, dummy_const.get(), F32);
  EXPECT_TRUE(status.ok());
  EXPECT_NE(step.execute_fn, nullptr);

  alignas(float) char buffer[sizeof(float) * 2];
  float* result = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      buffer, step.result_offset);

  step.execute_fn(&step, buffer);

  EXPECT_EQ(result[0], 1.0f);
  EXPECT_EQ(result[1], 2.0f);
}

}  // namespace
}  // namespace xla
