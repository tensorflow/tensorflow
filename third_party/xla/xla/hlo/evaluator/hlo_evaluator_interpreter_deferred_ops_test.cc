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

#include "xla/hlo/evaluator/hlo_evaluator_interpreter_deferred_ops.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
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

TEST(HloEvaluatorInterpreterDeferredOpsTest, GetDefaultDeferredOpRegistryTest) {
  const auto& registry = GetDefaultDeferredOpRegistry();

  // Create a dummy instruction for an unsupported opcode, e.g., kNegate
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  auto dummy_param = HloInstruction::CreateParameter(0, shape, "param");
  auto negate_instr =
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, dummy_param.get());

  size_t current_offset = 0;
  auto status_or = registry.Process(
      nullptr, negate_instr.get(), 0, std::nullopt,
      [](const HloInstruction*) -> const Literal& {
        static Literal l;
        return l;
      },
      current_offset);

  EXPECT_FALSE(status_or.ok());
  EXPECT_TRUE(absl::IsNotFound(status_or.status()));
}

TEST(HloEvaluatorInterpreterDeferredOpsTest, ExecuteSliceTest) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape operand_shape = ShapeUtil::MakeShape(F32, {4, 4});
  auto dummy_param = HloInstruction::CreateParameter(0, operand_shape, "param");
  auto slice_instr = HloInstruction::CreateSlice(shape, dummy_param.get(),
                                                 {1, 1}, {3, 3}, {1, 1});
  SliceMetadata metadata(slice_instr.get());

  LinearizedInterpreter::Step step;
  step.batch_size = 1;
  step.op_metadata = metadata;

  step.operand_offsets = {0};
  step.result_offset = sizeof(int64_t) * 2;

  alignas(int64_t) char buffer[sizeof(int64_t) * 4];
  int64_t* result_indices =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int64_t>(
          buffer, step.operand_offsets[0]);
  int64_t* operand_indices =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int64_t>(
          buffer, step.result_offset);

  // Output index [0, 0] -> input index [1+0*1, 1+0*1] = [1, 1]
  result_indices[0] = 0;
  result_indices[1] = 0;

  ExecuteSlice(&step, buffer);

  EXPECT_EQ(operand_indices[0], 1);
  EXPECT_EQ(operand_indices[1], 1);

  // Output index [1, 1] -> input index [1+1*1, 1+1*1] = [2, 2]
  result_indices[0] = 1;
  result_indices[1] = 1;

  ExecuteSlice(&step, buffer);

  EXPECT_EQ(operand_indices[0], 2);
  EXPECT_EQ(operand_indices[1], 2);
}

TEST(HloEvaluatorInterpreterDeferredOpsTest, ExecuteBroadcastTest) {
  Shape operand_shape = ShapeUtil::MakeShape(F32, {2});
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  auto dummy_param = HloInstruction::CreateParameter(0, operand_shape, "param");
  auto broadcast_instr =
      HloInstruction::CreateBroadcast(shape, dummy_param.get(), {0});
  BroadcastMetadata metadata(broadcast_instr.get());

  LinearizedInterpreter::Step step;
  step.batch_size = 1;
  step.op_metadata = metadata;

  step.operand_offsets = {0};
  step.result_offset = sizeof(int64_t) * 2;

  alignas(int64_t) char buffer[sizeof(int64_t) * 3];  // result_indices rank 2,
                                                      // operand_indices rank 1
  int64_t* result_indices =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int64_t>(
          buffer, step.operand_offsets[0]);
  int64_t* operand_indices =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int64_t>(
          buffer, step.result_offset);

  // Output index [1, 2] -> operand index [1] (since dimensions is {0})
  result_indices[0] = 1;
  result_indices[1] = 2;

  ExecuteBroadcast(&step, buffer);

  EXPECT_EQ(operand_indices[0], 1);
}

TEST(HloEvaluatorInterpreterDeferredOpsTest, ExecuteIotaTest) {
  Shape shape = ShapeUtil::MakeShape(S32, {2, 3});
  auto iota_instr = HloInstruction::CreateIota(shape, 1);
  IotaMetadata metadata(iota_instr.get());

  LinearizedInterpreter::Step step;
  step.batch_size = 2;
  step.op_metadata = metadata;

  step.operand_offsets = {0};
  step.result_offset = sizeof(int64_t) * 4;

  alignas(int64_t) char buffer[sizeof(int64_t) * 4 + sizeof(int32_t) * 2];
  int64_t* indices =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int64_t>(
          buffer, step.operand_offsets[0]);
  int32_t* result =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int32_t>(
          buffer, step.result_offset);

  indices[0] = 0;
  indices[1] = 1;

  indices[2] = 1;
  indices[3] = 2;

  ExecuteIota<int32_t>(&step, buffer);

  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 2);
}

TEST(HloEvaluatorInterpreterDeferredOpsTest, ExecuteLookupTest) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  auto param_instr = HloInstruction::CreateParameter(0, shape, "param");

  Literal literal = LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {3.0f, 4.0f}});

  auto resolver = [&](const HloInstruction* instr) -> const Literal& {
    return literal;
  };

  LookupMetadata metadata(param_instr.get(), resolver);

  LinearizedInterpreter::Step step;
  step.batch_size = 2;
  step.op_metadata = metadata;

  step.operand_offsets = {0};
  step.result_offset = sizeof(int64_t) * 4;

  alignas(int64_t) char buffer[sizeof(int64_t) * 4 + sizeof(float) * 2];
  int64_t* indices =
      LinearizedInterpreter::Scratchpad::GetPointerFromBase<int64_t>(
          buffer, step.operand_offsets[0]);
  float* result = LinearizedInterpreter::Scratchpad::GetPointerFromBase<float>(
      buffer, step.result_offset);

  indices[0] = 0;
  indices[1] = 1;

  indices[2] = 1;
  indices[3] = 0;

  ExecuteLookup<float>(&step, buffer);

  EXPECT_EQ(result[0], 2.0f);
  EXPECT_EQ(result[1], 3.0f);
}

}  // namespace
}  // namespace xla
