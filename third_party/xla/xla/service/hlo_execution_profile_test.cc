/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/hlo_execution_profile.h"
#include "absl/strings/str_cat.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

using absl::StrCat;
using ::testing::AllOf;
using ::testing::ContainsRegex;

class HloExecutionProfileTest : public HloTestBase {};

TEST_F(HloExecutionProfileTest, Basic) {
  auto hlo_module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY entry_computation {
    lhs = f32[30,30]{1,0} parameter(0)
    rhs = f32[30,30]{1,0} parameter(1)
    add = f32[30,30]{1,0} add(lhs, rhs)
    ROOT dot = f32[30,30]{1,0} dot(lhs, add), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })")
                        .value();
  const HloInstruction* dot_instruction =
      hlo_module->entry_computation()->root_instruction();
  const HloInstruction* add_instruction = dot_instruction->operand(1);
  Shape shape = ShapeUtil::MakeShape(F32, {30, 30});

  auto shape_size_function = [&](const Shape& shape) {
    const int64_t pointer_size = 8;
    if (shape.IsOpaque()) {
      return pointer_size;
    }
    return ShapeUtil::ByteSizeOf(shape, pointer_size);
  };

  HloCostAnalysis cost_analysis(shape_size_function);
  HloProfileIndexMap profile_index_map(*hlo_module);
  std::unique_ptr<HloProfilePrinterData> profile_printer =
      CreateHloProfilePrinterData(profile_index_map, cost_analysis,
                                  hlo_module->entry_computation()->name());
  HloExecutionProfile execution_profile(profile_printer.get(),
                                        &profile_index_map);

  const int64_t add_cycles = 1000;
  const int64_t dot_cycles = 4000;

  execution_profile.SetCyclesTakenBy(add_instruction, add_cycles);
  execution_profile.SetCyclesTakenBy(dot_instruction, dot_cycles);

  float clock_rate_ghz = backend()
                             .default_stream_executor()
                             ->GetDeviceDescription()
                             .clock_rate_ghz();
  EXPECT_THAT(execution_profile.ToString(clock_rate_ghz),
              AllOf(ContainsRegex(StrCat(dot_cycles, " cycles.*%",
                                         dot_instruction->name())),
                    ContainsRegex(StrCat(add_cycles, " cycles.*%",
                                         add_instruction->name()))));
}
}  // namespace
}  // namespace xla
