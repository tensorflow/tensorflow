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
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace {

using tensorflow::strings::StrCat;
using ::testing::AllOf;
using ::testing::ContainsRegex;

class HloExecutionProfileTest : public HloTestBase {};

TEST_F(HloExecutionProfileTest, Basic) {
  auto hlo_module = tools::Parse(R"(
  HloModule test_module
  ENTRY entry_computation {
    lhs = f32[30,30]{1,0} parameter(0)
    rhs = f32[30,30]{1,0} parameter(1)
    add = f32[30,30]{1,0} add(lhs, rhs)
    ROOT dot = f32[30,30]{1,0} dot(lhs, add), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })")
                        .ValueOrDie();
  const HloInstruction* dot_instruction =
      hlo_module->entry_computation()->root_instruction();
  const HloInstruction* add_instruction = dot_instruction->operand(1);
  Shape shape = ShapeUtil::MakeShape(F32, {30, 30});

  auto shape_size_function = [&](const Shape& shape) {
    const int64 pointer_size = 8;
    if (ShapeUtil::IsOpaque(shape)) {
      return pointer_size;
    }
    return ShapeUtil::ByteSizeOf(shape, pointer_size);
  };

  HloCostAnalysis cost_analysis(shape_size_function);
  HloProfileIndexMap profile_index_map(*hlo_module);
  std::unique_ptr<HloProfilePrinterData> profile_printer =
      CreateHloProfilePrinterData(profile_index_map, cost_analysis);
  HloExecutionProfile execution_profile(profile_printer.get(),
                                        &profile_index_map);

  const int64 add_cycles = 1000;
  const int64 dot_cycles = 4000;

  execution_profile.SetCyclesTakenBy(add_instruction, add_cycles);
  execution_profile.SetCyclesTakenBy(dot_instruction, dot_cycles);

  EXPECT_THAT(execution_profile.ToString(
                  backend().default_stream_executor()->GetDeviceDescription()),
              AllOf(ContainsRegex(StrCat(dot_cycles, R"(\b.*%)",
                                         dot_instruction->name())),
                    ContainsRegex(StrCat(add_cycles, R"(\b.*%)",
                                         add_instruction->name()))));
}
}  // namespace
}  // namespace xla
