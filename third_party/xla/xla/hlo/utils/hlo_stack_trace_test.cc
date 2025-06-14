/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/utils/hlo_stack_trace.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

class HloStackTraceTest : public HloHardwareIndependentTestBase {
 protected:
  HloStackTraceTest() : module_(CreateNewVerifiedModule()) {}
  ~HloStackTraceTest() override = default;

  std::unique_ptr<xla::HloModule> module_;
};

TEST_F(HloStackTraceTest, FormatStackTraceBreakdownCrashes) {
  std::vector<std::pair<int64_t, const HloValue*>> sized_buffers;
  EXPECT_DEATH(FormatStackTraceBreakdown(sized_buffers, nullptr), "");
}

TEST_F(HloStackTraceTest, FormatStackTraceEmpty) {
  std::vector<std::pair<int64_t, const HloValue*>> sized_buffers;
  std::string expected = R"(  Stack trace breakdown for peak usage: 0 bytes
)";
  std::string actual = FormatStackTraceBreakdown(sized_buffers, module_.get());
  EXPECT_EQ(actual, expected);
}

TEST_F(HloStackTraceTest, StackTraceFormatMatchesExpected) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  // Inner computation: multiply
  auto inner_builder = HloComputation::Builder("InnerComp");
  auto a = inner_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "a"));
  auto b = inner_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "b"));
  auto mul = inner_builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, a, b));
  HloComputation* inner =
      module->AddEmbeddedComputation(inner_builder.Build(mul));

  // Middle computation: call InnerComp
  auto middle_builder = HloComputation::Builder("MiddleComp");
  auto x = middle_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "x"));
  auto y = middle_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "y"));
  auto call_inner = middle_builder.AddInstruction(
      HloInstruction::CreateCall(shape, {x, y}, inner));
  HloComputation* middle =
      module->AddEmbeddedComputation(middle_builder.Build(call_inner));

  // Entry computation: call MiddleComp
  auto entry_builder = HloComputation::Builder("ToString");
  auto arg1 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "arg1"));
  auto arg2 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "arg2"));
  auto call_middle = entry_builder.AddInstruction(
      HloInstruction::CreateCall(shape, {arg1, arg2}, middle));
  module->AddEntryComputation(entry_builder.Build(call_middle));

  // Simulate memory usage (16 bytes each).
  std::vector<std::pair<int64_t, const HloValue*>> sized_buffers;
  auto make_fake_value = [](HloInstruction* instr, int64_t id) {
    return std::make_unique<HloValue>(id, instr, ShapeIndex{}, false);
  };

  std::vector<std::unique_ptr<HloValue>> owned_values;
  owned_values.push_back(make_fake_value(call_middle, 1));
  owned_values.push_back(make_fake_value(arg1, 2));
  owned_values.push_back(make_fake_value(arg2, 3));
  owned_values.push_back(make_fake_value(call_inner, 4));
  owned_values.push_back(make_fake_value(mul, 5));

  for (const auto& v : owned_values) {
    sized_buffers.emplace_back(16, v.get());
  }

  std::string expected = R"(  Stack trace breakdown for peak usage: 80 bytes
    ToString (100.0%, 80 bytes)
      ├── MiddleComp (40.0%, 32 bytes)
      │   ├── InnerComp (20.0%, 16 bytes)
      │   │   └── multiply (20.0%, 16 bytes)
      │   └── call (20.0%, 16 bytes)
      ├── arg1 (20.0%, 16 bytes)
      ├── arg2 (20.0%, 16 bytes)
      └── call.1 (20.0%, 16 bytes)
)";

  std::string actual = FormatStackTraceBreakdown(sized_buffers, module.get());
  EXPECT_EQ(actual, expected);
}

}  // namespace
}  // namespace xla
