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
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

class HloStackTraceTest : public HloHardwareIndependentTestBase {};

// Helper to fabricate an HloValue that only needs instruction().
static std::unique_ptr<HloValue> MakeFakeValue(HloInstruction* instr,
                                               int64_t id) {
  return std::make_unique<HloValue>(id, instr, ShapeIndex{}, false);
}

TEST_F(HloStackTraceTest, FormatStackTraceBreakdownCrashes) {
  std::vector<std::pair<int64_t, const HloValue*>> sized_buffers;
  EXPECT_DEATH(FormatStackTraceBreakdown(sized_buffers, nullptr), "");
}

TEST_F(HloStackTraceTest, FormatStackTraceEmpty) {
  std::vector<std::pair<int64_t, const HloValue*>> sized_buffers;
  std::string expected = "  Stack trace breakdown for peak usage: 0 bytes\n";
  auto module = CreateNewVerifiedModule();
  std::string actual = FormatStackTraceBreakdown(sized_buffers, module.get());
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
  auto entry_builder = HloComputation::Builder("main");
  auto arg1 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "arg1"));
  auto arg2 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "arg2"));
  auto call_middle = entry_builder.AddInstruction(
      HloInstruction::CreateCall(shape, {arg1, arg2}, middle));
  module->AddEntryComputation(entry_builder.Build(call_middle));

  // Simulate memory usage (16 bytes each).
  std::vector<std::pair<int64_t, const HloValue*>> buffers;

  std::vector<std::unique_ptr<HloValue>> owned_values;
  owned_values.push_back(MakeFakeValue(call_middle, 1));
  owned_values.push_back(MakeFakeValue(arg1, 2));
  owned_values.push_back(MakeFakeValue(arg2, 3));
  owned_values.push_back(MakeFakeValue(call_inner, 4));
  owned_values.push_back(MakeFakeValue(mul, 5));

  for (const auto& v : owned_values) {
    buffers.emplace_back(16, v.get());
  }

  std::string expected = R"(  Stack trace breakdown for peak usage: 80 bytes
    main (100.0%, total: 80 bytes, current: 0 bytes, remaining: 80 bytes)
      ├── MiddleComp (40.0%, total: 32 bytes, current: 0 bytes, remaining: 80 bytes)
      │   ├── InnerComp (20.0%, total: 16 bytes, current: 0 bytes, remaining: 80 bytes)
      │   │   └── multiply (20.0%, total: 16 bytes, current: 16 bytes, remaining: 64 bytes)
      │   └── call (20.0%, total: 16 bytes, current: 16 bytes, remaining: 48 bytes)
      ├── arg1 (20.0%, total: 16 bytes, current: 16 bytes, remaining: 32 bytes)
      ├── arg2 (20.0%, total: 16 bytes, current: 16 bytes, remaining: 16 bytes)
      └── call.1 (20.0%, total: 16 bytes, current: 16 bytes, remaining: 0 bytes)
)";

  std::string actual = FormatStackTraceBreakdown(buffers, module.get());
  EXPECT_EQ(actual, expected);
}

TEST_F(HloStackTraceTest, QuotedParameterNamesAppearInStackTrace) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(F32, {4});

  auto builder = HloComputation::Builder("main");

  auto new_param = [&](int64_t n, const std::string& display) {
    auto* p = builder.AddInstruction(
        HloInstruction::CreateParameter(n, shape, absl::StrCat("p", n)));
    OpMetadata meta;
    meta.set_op_name(display);  // keep quotes/slashes
    p->set_metadata(meta);
    return p;
  };

  auto p0 =
      new_param(0, "params['transformer/layer_1/mlp/linear/AqtEinsum_0']");
  auto p1 = new_param(
      1,
      "params['transformer/layer_1/mlp/linear/AqtEinsum_0/AqtDotGeneral_0']");
  auto p2 = new_param(2,
                      "params['transformer/layer_1/mlp/linear/"
                      "AqtEinsum_0/AqtDotGeneral_0/qrhs']['frozen'].qvalue");

  module->AddEntryComputation(builder.Build());

  std::vector<std::unique_ptr<HloValue>> owned;
  owned.push_back(MakeFakeValue(p0, 1));
  owned.push_back(MakeFakeValue(p1, 2));
  owned.push_back(MakeFakeValue(p2, 3));

  std::vector<std::pair<int64_t, const HloValue*>> buffers;
  for (const auto& v : owned) {
    buffers.emplace_back(16, v.get());
  }

  std::string expected = R"(  Stack trace breakdown for peak usage: 48 bytes
    main (100.0%, total: 48 bytes, current: 0 bytes, remaining: 48 bytes)
      └── params (100.0%, total: 48 bytes, current: 0 bytes, remaining: 48 bytes)
          └── transformer (100.0%, total: 48 bytes, current: 0 bytes, remaining: 48 bytes)
              └── layer_1 (100.0%, total: 48 bytes, current: 0 bytes, remaining: 48 bytes)
                  └── mlp (100.0%, total: 48 bytes, current: 0 bytes, remaining: 48 bytes)
                      └── linear (100.0%, total: 48 bytes, current: 0 bytes, remaining: 48 bytes)
                          └── AqtEinsum_0 (100.0%, total: 48 bytes, current: 16 bytes, remaining: 32 bytes)
                              └── AqtDotGeneral_0 (66.7%, total: 32 bytes, current: 16 bytes, remaining: 16 bytes)
                                  └── qrhs (33.3%, total: 16 bytes, current: 0 bytes, remaining: 16 bytes)
                                      └── frozen (33.3%, total: 16 bytes, current: 0 bytes, remaining: 16 bytes)
                                          └── qvalue (33.3%, total: 16 bytes, current: 16 bytes, remaining: 0 bytes)
)";
  std::string actual = FormatStackTraceBreakdown(buffers, module.get());
  EXPECT_EQ(actual, expected);
}

TEST_F(HloStackTraceTest, RemainingNotZeroForDuplicateLeafPaths) {
  auto module = CreateNewVerifiedModule();
  Shape shape = ShapeUtil::MakeShape(F32, {2});

  auto builder = HloComputation::Builder("main");
  auto dup =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "dup"));
  module->AddEntryComputation(builder.Build());

  // Two different HloValue objects, *same* instruction => identical stack path.
  std::vector<std::unique_ptr<HloValue>> owned;
  owned.push_back(MakeFakeValue(dup, 1));
  owned.push_back(MakeFakeValue(dup, 2));

  std::vector<std::pair<int64_t, const HloValue*>> buffers;
  for (const auto& v : owned) {
    buffers.emplace_back(16, v.get());
  }

  // Because total_bytes = 32 but leaf node_size = 16, the last row keeps
  // 16 bytes "remaining".
  std::string expected = R"(  Stack trace breakdown for peak usage: 32 bytes
    main (100.0%, total: 32 bytes, current: 0 bytes, remaining: 32 bytes)
      └── dup (100.0%, total: 32 bytes, current: 16 bytes, remaining: 16 bytes)
)";

  std::string actual = FormatStackTraceBreakdown(buffers, module.get());
  EXPECT_EQ(actual, expected);
}
}  // namespace
}  // namespace xla
