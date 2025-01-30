/* Copyright 2024 The OpenXLA Authors.
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

#include "xla/service/gpu/transforms/collective_permute_valid_iteration_annotator.h"

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

const char* base_hlo = R"(
    HloModule test, entry_computation_layout={()->(s32[], s32[])}
    %Body (param: (s32[], s32[])) -> (s32[], s32[]) {
      %param = (s32[], s32[]) parameter(0)
      %i = s32[] get-tuple-element((s32[], s32[]) %param), index=1
      %one = s32[] constant(1)
      %i_plus_one = s32[] add(s32[] %i, s32[] %one)
      %permute = s32[] collective-permute(%i_plus_one), channel_id=1,
        source_target_pairs=$source_target_pairs
      ROOT %tuple = (s32[], s32[]) tuple(s32[] %permute, s32[] %i_plus_one)
    }
    %Cond (param.1: (s32[], s32[])) -> pred[] {
      %param.1 = (s32[], s32[]) parameter(0)
      %i.1 = s32[] get-tuple-element((s32[], s32[]) %param.1), index=1
      %trip_count = s32[] constant(10)
      ROOT %done = pred[] compare(s32[] %i.1, s32[] %trip_count), direction=LT
    }
    ENTRY %test () -> (s32[], s32[]) {
      %i_start = s32[] constant(0)
      %p_start = s32[] constant(0)
      %initial_tuple = (s32[], s32[]) tuple(s32[] %i_start, s32[] %p_start)
      ROOT %while = (s32[], s32[]) while((s32[], s32[]) %initial_tuple),
        condition=%Cond, body=%Body,
        frontend_attributes={$is_pipelined_while_loop}
    }
  )";

struct AnnotatorTestCase {
  std::string test_name;
  std::string pairs;
  std::string pipelined_attr;
  bool expect_change;
  std::optional<std::string> validate_attr;

  friend std::ostream& operator<<(std::ostream& os,
                                  const AnnotatorTestCase& tc) {
    os << absl::StrFormat("%s : %s, %s", tc.test_name, tc.pairs,
                          tc.pipelined_attr);
    return os;
  }
};

std::vector<AnnotatorTestCase> GetTestCases() {
  return {
      {.test_name = "NoPipelinedWhileLoopAttribute",
       .pairs = "{{0,1},{1,2},{2,3},{3,0}}",
       .pipelined_attr = "",
       .expect_change = false,
       .validate_attr = std::nullopt},

      {.test_name = "ForwardCycle",
       .pairs = "{{0,1},{1,2},{2,3},{3,0}}",
       .pipelined_attr = "is_pipelined_while_loop=\"true\"",
       .expect_change = true,
       .validate_attr = "{{0,6},{1,7},{2,8},{3,9}}"},

      {.test_name = "BackwardCycle",
       .pairs = "{{0,3},{1,0},{2,1},{3,2}}",
       .pipelined_attr = "is_pipelined_while_loop=\"true\"",
       .expect_change = true,
       .validate_attr = "{{3,9},{2,8},{1,7},{0,6}}"},
  };
}

class AnnotatorTest : public HloHardwareIndependentTestBase,
                      public ::testing::WithParamInterface<AnnotatorTestCase> {
};

TEST_P(AnnotatorTest, ParameterizedAnnotatorTest) {
  const AnnotatorTestCase& test_case = GetParam();
  FixedMapping params = {
      {"$source_target_pairs", test_case.pairs},
      {"$is_pipelined_while_loop", test_case.pipelined_attr}};

  // WhileLoopTripCountAnnotator should discover the while loop and add the
  // trip count to the backend config which is used by the
  // CollectivePermuteValidIterationAnnotator.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      RunAndCheckHloRewrite(base_hlo, WhileLoopTripCountAnnotator(), true,
                            params));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      CollectivePermuteValidIterationAnnotator().Run(module.get()));
  VLOG(2) << "HLO after CollectivePermuteValidIterationAnnotator: \n"
          << module->ToString(HloPrintOptions::ShortParsable()
                                  .set_print_control_dependencies(true));
  EXPECT_EQ(changed, test_case.expect_change);

  HloCollectivePermuteInstruction* cp =
      DynCastOrNull<HloCollectivePermuteInstruction>(
          FindInstruction(module.get(), HloOpcode::kCollectivePermute));
  ASSERT_NE(cp, nullptr);
  ASSERT_EQ(cp->get_frontend_attribute(kSendRecvValidationAttr),
            test_case.validate_attr);
}

INSTANTIATE_TEST_SUITE_P(
    AnnotatorTestSuite, AnnotatorTest, testing::ValuesIn(GetTestCases()),
    [](const testing::TestParamInfo<AnnotatorTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace xla
