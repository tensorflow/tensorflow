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

#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

class HloExecutionProfileTest : public HloTestBase {
 protected:
  static constexpr int64 kInstructionCyclesIndex = 0;
  static constexpr int64 kInstructionNameIndex = 19;
};

// Splits `lines` into a sequence of lines delimited by newlines and then split
// each of those lines into a sequence of words delimited by spaces.  Filter out
// empty words.
std::vector<std::vector<string>> SplitIntoLinesAndWords(
    tensorflow::StringPiece lines) {
  std::vector<std::vector<string>> result;
  for (const string& line : tensorflow::str_util::Split(lines, '\n')) {
    std::vector<string> words;
    for (const string& word : tensorflow::str_util::Split(line, ' ')) {
      if (!word.empty()) {
        words.push_back(word);
      }
    }
    result.push_back(std::move(words));
  }

  return result;
}

TEST_F(HloExecutionProfileTest, Basic) {
  std::unique_ptr<HloModule> hlo_module = CreateNewModule();

  HloComputation::Builder builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {30, 30});
  HloInstruction* param_lhs =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "lhs"));
  HloInstruction* param_rhs =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "rhs"));
  HloInstruction* add_instruction =
      builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, param_lhs, param_rhs));
  HloInstruction* dot_instruction =
      builder.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kDot, param_lhs, add_instruction));

  hlo_module->AddEntryComputation(builder.Build());

  auto shape_size_function = [&](const Shape& shape) {
    const int64 pointer_size = 8;
    if (ShapeUtil::IsOpaque(shape)) {
      return pointer_size;
    }
    return ShapeUtil::ByteSizeOf(shape, pointer_size);
  };

  HloCostAnalysis cost_analysis(shape_size_function);
  HloExecutionProfile execution_profile(*hlo_module, cost_analysis);

  const int64 add_cycles = 1000;
  const int64 dot_cycles = 4000;

  execution_profile.SetCyclesTakenBy(add_instruction, add_cycles);
  execution_profile.SetCyclesTakenBy(dot_instruction, dot_cycles);

  string rendered_profile = execution_profile.ToString(
      backend().default_stream_executor()->GetDeviceDescription());
  std::vector<std::vector<string>> lines_and_words =
      SplitIntoLinesAndWords(rendered_profile);
  ASSERT_EQ(lines_and_words.size(), 8);

  const std::vector<string>& line_2 = lines_and_words[2];
  const std::vector<string>& line_3 = lines_and_words[3];

  EXPECT_EQ(line_2[kInstructionCyclesIndex], std::to_string(dot_cycles));
  EXPECT_EQ(line_2[kInstructionNameIndex], dot_instruction->name());

  EXPECT_EQ(line_3[kInstructionCyclesIndex], std::to_string(add_cycles));
  EXPECT_EQ(line_3[kInstructionNameIndex], add_instruction->name());
}
}  // namespace
}  // namespace xla
