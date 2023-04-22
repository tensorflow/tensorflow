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

#include <array>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

// Tests the Reduce HLO in ways that can't be done using the ComputationBuilder
// API.

namespace xla {
namespace {

struct ReduceLayout {
  std::array<int64, 4> input_minor_to_major;
  std::array<int64, 3> output_minor_to_major;

  string ToString() const {
    return absl::StrCat(absl::StrJoin(input_minor_to_major, "x"), "_",
                        absl::StrJoin(output_minor_to_major, "x"));
  }
};

string PrintReduceLayout(
    ::testing::TestParamInfo<ReduceLayout> reduce_layout_param) {
  return reduce_layout_param.param.ToString();
}

void PrintTo(const ReduceLayout& reduce_layout, ::std::ostream* os) {
  *os << reduce_layout.ToString();
}

class ReduceWithLayoutTest
    : public HloTestBase,
      public ::testing::WithParamInterface<ReduceLayout> {
 public:
  StatusOr<std::unique_ptr<HloModule>> GetParsedModule() {
    const char* const hlo_string = R"(
HloModule BadReduce

Sum {
  x.1 = f32[] parameter(0)
  y.1 = f32[] parameter(1)
  ROOT add.1 = f32[] add(x.1, y.1)
}

ENTRY reduce.1 {
  parameter = f32[2,2,2,3]{3,2,1,0} parameter(0)
  init_value = f32[] constant(0)
  reduce = f32[2,2,3]{2,1,0} reduce(parameter, init_value), dimensions={1}, to_apply=Sum
  ROOT copy = f32[2,2,3]{2,1,0} copy(reduce)
}
)";

    return ParseAndReturnVerifiedModule(hlo_string);
  }
};

XLA_TEST_P(ReduceWithLayoutTest, Reduce) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module, GetParsedModule());
  HloInstruction* reduce_instruction =
      module->entry_computation()->root_instruction()->mutable_operand(0);
  ASSERT_EQ(reduce_instruction->opcode(), HloOpcode::kReduce);

  const ReduceLayout& reduce_layout = GetParam();

  Shape* reduce_output_shape = reduce_instruction->mutable_shape();
  *reduce_output_shape->mutable_layout() =
      LayoutUtil::MakeLayout(reduce_layout.output_minor_to_major);

  Shape* reduce_input_shape =
      reduce_instruction->mutable_operand(0)->mutable_shape();
  *reduce_input_shape->mutable_layout() =
      LayoutUtil::MakeLayout(reduce_layout.input_minor_to_major);

  Literal reduce_input = LiteralUtil::CreateR4<float>(
      {{ /*i0=0*/
        {/*i1=0*/
         {-0.246092796, -0.179497838, -0.161181688},
         {-0.151643038, -0.240213156, -0.198156}},
        {/*i1=1*/
         {-0.14222312, -0.162200093, -0.193907976},
         {-0.239411, -0.198166847, -0.172471642}}},
       { /*i0=1*/
        {/*i1=0*/
         {-0.22965157, -0.218723893, -0.129257083},
         {-0.188762426, -0.16123569, -0.181166649}},
        {/*i1=1*/
         {-0.241772294, -0.245131493, -0.160247207},
         {-0.179881215, -0.23383224, -0.121976733}}}});

  auto reduce_input_relaid =
      reduce_input.Relayout(reduce_input_shape->layout());
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), {&reduce_input_relaid}, ErrorSpec(1e-5)));
}

INSTANTIATE_TEST_CASE_P(ReduceWithLayoutTest_Instantiation,
                        ReduceWithLayoutTest,
                        ::testing::Values(                           //
                            ReduceLayout{{3, 2, 1, 0}, {0, 1, 2}},   //
                            ReduceLayout{{3, 2, 1, 0}, {0, 2, 1}},   //
                            ReduceLayout{{3, 2, 1, 0}, {1, 2, 0}},   //
                            ReduceLayout{{3, 2, 1, 0}, {1, 0, 2}},   //
                            ReduceLayout{{3, 2, 1, 0}, {2, 0, 1}},   //
                            ReduceLayout{{3, 2, 1, 0}, {2, 1, 0}},   //
                            ReduceLayout{{3, 1, 2, 0}, {1, 2, 0}},   //
                            ReduceLayout{{1, 2, 3, 0}, {1, 0, 2}},   //
                            ReduceLayout{{0, 2, 1, 3}, {2, 0, 1}}),  //
                        PrintReduceLayout);

}  // namespace
}  // namespace xla
