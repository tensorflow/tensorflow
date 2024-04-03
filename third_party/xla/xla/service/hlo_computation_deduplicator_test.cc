
/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/hlo_computation_deduplicator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

class HloComputationDeduplicatorTest : public HloTestBase {
 protected:
  std::vector<std::string> RunDeduplicatePass(const std::string_view text,
                                              bool expect_true) {
    std::unique_ptr<HloModule> module =
        ParseAndReturnVerifiedModule(text).value();
    HloComputationDeduplicator dedup;
    bool changed = dedup.Run(module.get()).value();
    EXPECT_EQ(changed, expect_true);
    std::vector<std::string> computation_names;
    for (auto comp : module->computations()) {
      computation_names.emplace_back(comp->name());
    }
    return computation_names;
  }
};

TEST_F(HloComputationDeduplicatorTest, RemoveRegionBandC) {
  const std::string_view text = R"(
  HloModule DeDupTest, entry_computation_layout={(s32[10]{0},s32[15]{0}, s32[20]{0})->s32[]}
  region_A {
    Arg_0.6 = s32[] parameter(0)
    Arg_1.7 = s32[] parameter(1)
    ROOT add.8 = s32[] add(Arg_0.6, Arg_1.7)
  }

  region_B {
    Arg_0.11 = s32[] parameter(0)
    Arg_1.12 = s32[] parameter(1)
    ROOT add.13 = s32[] add(Arg_0.11, Arg_1.12)
  }

  region_C {
    Arg_0.17 = s32[] parameter(0)
    Arg_1.18 = s32[] parameter(1)
   ROOT add.19 = s32[] add(Arg_0.17, Arg_1.18)
  }

  ENTRY main.22 {
    Arg_0.1 = s32[10]{0} parameter(0)
    Arg_1.2 = s32[15]{0} parameter(1)
    Arg_2.3 = s32[20]{0} parameter(2)
    constant.4 = s32[] constant(0)
    reduce.9 = s32[] reduce(Arg_0.1, constant.4), dimensions={0}, to_apply=region_A
    reduce.14 = s32[] reduce(Arg_1.2, constant.4), dimensions={0}, to_apply=region_B
    reduce.20 = s32[] reduce(Arg_2.3, constant.4), dimensions={0}, to_apply=region_C
    multiply.15 = s32[] multiply(reduce.9, reduce.14)
    ROOT multiply.21 = s32[] multiply(multiply.15, reduce.20)
  }
  )";
  auto computation_names = RunDeduplicatePass(text, /*expect_true=*/true);
  // Test should replace region_B usage with region_A and remove Region_B since
  // the computation is the same even though variables are different.
  for (auto name : computation_names) {
    EXPECT_NE(name, "region_B");
    EXPECT_NE(name, "region_C");
  }
  EXPECT_EQ(computation_names.size(), 2);
}
TEST_F(HloComputationDeduplicatorTest, RemoveRegionBExactCopy) {
  const std::string_view text = R"(
  HloModule DeDupTest, entry_computation_layout={(s32[10]{0},s32[15]{0})->s32[]}
  region_A {
    Arg_0.5 = s32[] parameter(0)
    Arg_1.6 = s32[] parameter(1)
    ROOT add.7 = s32[] add(Arg_0.5, Arg_1.6)
  }
  region_B {
    Arg_0.5 = s32[] parameter(0)
    Arg_1.6 = s32[] parameter(1)
    ROOT add.7 = s32[] add(Arg_0.5, Arg_1.6)
  }
  ENTRY main.15 {
    Arg_0.1 = s32[10]{0} parameter(0)
    constant.3 = s32[] constant(0)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=region_A
    Arg_1.2 = s32[15]{0} parameter(1)
    rd2 = s32[] reduce(Arg_1.2, constant.3), dimensions={0}, to_apply=region_B
    ROOT multiply.14 = s32[] multiply(rd1, rd2)
  }
  )";

  auto computation_names = RunDeduplicatePass(text, /*expect_true=*/true);
  // Test should replace region_B usage with region_A and remove Region_B since
  // they are exact duplicates.
  for (auto name : computation_names) {
    EXPECT_NE(name, "region_B");
  }
  EXPECT_EQ(computation_names.size(), 2);
}

TEST_F(HloComputationDeduplicatorTest, RemoveRegionsWithSameSubcomp) {
  const std::string_view text = R"(
  HloModule DeDupTest, entry_computation_layout={(s32[10]{0},s32[15]{0})->s32[]}
  region_X {
    Ag_0 = s32[] parameter(0)
    Arg_1 = s32[] parameter(1)
    ROOT their_sum = s32[] add(Ag_0, Arg_1)
  }

  region_Y {
    Arg_0 = s32[] parameter(0)
    Arg_1 = s32[] parameter(1)
    ROOT the_sum = s32[] add(Arg_0, Arg_1)
  }
  region_A {
    Arg_0.5 = s32[] parameter(0)
    Arg_1.6 = s32[] parameter(1)
    ROOT add.7 = s32[] add(Arg_0.5, Arg_1.6)
  }
  region_B {
    Arg_0.5 = s32[] parameter(0)
    Ar_1.6 = s32[] parameter(1)
    ROOT add.7 = s32[] add(Arg_0.5, Ar_1.6)
  }

  main.15 {
    Arg_0.1 = s32[10]{0} parameter(0)
    constant.3 = s32[] constant(0)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=region_X
    Arg_1.2 = s32[15]{0} parameter(1)
    rd2 = s32[] reduce(Arg_1.2, constant.3), dimensions={0}, to_apply=region_Y
    ROOT multiply.14 = s32[] multiply(rd1, rd2)
  }

  main.16 {
    Arg_0.1 = s32[10]{0} parameter(0)
    constant.3 = s32[] constant(0)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=region_A
    Arg_1.2 = s32[15]{0} parameter(1)
    rd2 = s32[] reduce(Arg_1.2, constant.3), dimensions={0}, to_apply=region_B
    ROOT multiply.14 = s32[] multiply(rd1, rd2)
  }

  main.17 {
    Arg_0 = s32[10]{0} parameter(0)
    Arg_1 = s32[15]{0} parameter(1)
    rd1 = s32[] call(Arg_0, Arg_1), to_apply=main.15
    rd2 = s32[] call(Arg_0, Arg_1), to_apply=main.16
    ROOT ret = add(rd1, rd2)
  } 
  )";

  auto computation_names = RunDeduplicatePass(text, /*expect_true=*/true);
  // Test should replace region_B usage with region_A and remove Region_B since
  // the subcomputations called by region_B (region_Y) is duplicate of region_X
  for (auto name : computation_names) {
    EXPECT_NE(name, "region_B");
    EXPECT_NE(name, "region_A");
    EXPECT_NE(name, "region_Y");
    EXPECT_NE(name, "main.16");
  }
  EXPECT_EQ(computation_names.size(), 3);
}
TEST_F(HloComputationDeduplicatorTest, DontRemoveRegionsWithDifferentSubcomp) {
  const std::string_view text = R"(
  HloModule DeDupTest, entry_computation_layout={(s32[10]{0},s32[15]{0})->s32[]}
  region_X {
    Ag_0 = s32[] parameter(0)
    Arg_1 = s32[] parameter(1)
    ROOT their_sum = s32[] multiply(Ag_0, Arg_1)
  }

  region_Y {
    Arg_0 = s32[] parameter(0)
    Arg_1 = s32[] parameter(1)
    ROOT the_sum = s32[] add(Arg_0, Arg_1)
  }

  region_A {
    Arg_0.5 = s32[] parameter(0)
    Arg_1.6 = s32[] parameter(1)
    ROOT add.7 = s32[] add(Arg_0.5, Arg_1.6)
  }

  region_B {
    Arg_0.5 = s32[] parameter(0)
    Ar_1.6 = s32[] parameter(1)
    ROOT add.7 = s32[] add(Arg_0.5, Ar_1.6)
  }

  main.15 {
    Arg_0.1 = s32[10]{0} parameter(0)
    constant.3 = s32[] constant(0)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=region_X
    Arg_1.2 = s32[15]{0} parameter(1)
    rd2 = s32[] reduce(Arg_1.2, constant.3), dimensions={0}, to_apply=region_Y
    ROOT multiply.14 = s32[] multiply(rd1, rd2)
  }

  main.16 {
    Arg_0.1 = s32[10]{0} parameter(0)
    constant.3 = s32[] constant(0)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=region_A
    Arg_1.2 = s32[15]{0} parameter(1)
    rd2 = s32[] reduce(Arg_1.2, constant.3), dimensions={0}, to_apply=region_B
    ROOT multiply.14 = s32[] multiply(rd1, rd2)
  }

  main.17 {
    Arg_0 = s32[10]{0} parameter(0)
    Arg_1 = s32[15]{0} parameter(1)
    rd1 = s32[] call(Arg_0, Arg_1), to_apply=main.15
    rd2 = s32[] call(Arg_0, Arg_1), to_apply=main.16
    ROOT ret = add(rd1, rd2)
  }
  )";

  auto computation_names = RunDeduplicatePass(text, /*expect_true=*/true);
  // Region_X has a multiply() instead of add(). This one change should just
  // mark region_a, region_b and region_Y as duplicates of each other.
  int region_x_count = 0;
  int region_y_count = 0;
  int main_16_count = 0;
  int main_15_count = 0;
  int region_a_count = 0;
  int region_b_count = 0;
  for (auto name : computation_names) {
    region_x_count += (name == "region_X");
    region_y_count += (name == "region_Y");
    main_15_count += (name == "main.15");
    main_16_count += (name == "main.16");
    region_a_count += (name == "region_A");
    region_b_count += (name == "region_B");
  }
  EXPECT_EQ(region_a_count, 0);
  EXPECT_EQ(region_b_count, 0);
  EXPECT_EQ(main_15_count, 1);
  EXPECT_EQ(main_16_count, 1);
  EXPECT_EQ(region_x_count, 1);
  EXPECT_EQ(region_y_count, 1);
}

TEST_F(HloComputationDeduplicatorTest, RemoveRegionBVarDifferences) {
  const std::string_view text = R"(
  HloModule DeDupTest, entry_computation_layout={(s32[10]{0},s32[15]{0})->s32[]}
  region_A {
    Arg_0.5 = s32[] parameter(0)
    Arg_1.6 = s32[] parameter(1)
    ROOT add.7 = s32[] add(Arg_0.5, Arg_1.6)
  }

  region_B {
    Arg_0.2 = s32[] parameter(0)
    Arg_1.3 = s32[] parameter(1)
    ROOT add.8 = s32[] add(Arg_0.2, Arg_1.3)
  }

  ENTRY main.15 {
    Arg_0.1 = s32[10]{0} parameter(0)
    constant.3 = s32[] constant(0)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=region_A
    Arg_1.2 = s32[15]{0} parameter(1)
    rd2 = s32[] reduce(Arg_1.2, constant.3), dimensions={0}, to_apply=region_B
    ROOT multiply.14 = s32[] multiply(rd1, rd2)
  }
  )";

  auto computation_names = RunDeduplicatePass(text, /*expect_true=*/true);
  // Test should replace region_B usage with region_A and remove Region_B since
  // the computation is the same even though variables are different.
  for (auto name : computation_names) {
    EXPECT_NE(name, "region_B");
  }
  EXPECT_EQ(computation_names.size(), 2);
}

TEST_F(HloComputationDeduplicatorTest, DontRemoveRegionBCommutative) {
  const std::string_view text = R"(
  HloModule DeDupTest, entry_computation_layout={(s32[10]{0},s32[15]{0})->s32[]}
  region_A {
    Arg_0 = s32[] parameter(0)
    Arg_1 = s32[] parameter(1)
    ROOT add.7 = s32[] add(Arg_1, Arg_0)
  }

  region_B {
    Arg_0.2 = s32[] parameter(0)
    Arg_1.3 = s32[] parameter(1)
    ROOT add.8 = s32[] add(Arg_0.2, Arg_1.3)
  }

  ENTRY main.15 {
    Arg_0.1 = s32[10]{0} parameter(0)
    constant.3 = s32[] constant(0)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=region_A
    Arg_1.2 = s32[15]{0} parameter(1)
    rd2 = s32[] reduce(Arg_1.2, constant.3), dimensions={0}, to_apply=region_B
    ROOT multiply.14 = s32[] multiply(rd1, rd2)
  }
  )";

  auto computation_names = RunDeduplicatePass(text, /*expect_true=*/false);
  // Will also take into account commutativety.
  int region_b_count = 0;
  for (auto name : computation_names) {
    region_b_count += (name == "region_B");
  }
  EXPECT_EQ(region_b_count, 1);
  EXPECT_EQ(computation_names.size(), 3);
}

TEST_F(HloComputationDeduplicatorTest, DontRemoveRegionLargeConstant) {
  const std::string_view text = R"(
  HloModule DeDupTest, entry_computation_layout={(s32[10]{0},s32[15]{0})->s32[]}
  region_A {
    Arg_00 = s32[] parameter(0)
    Arg_1_1 = s32[] parameter(1)
    Arg_0 = s32[10, 10] constant({{1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}})
    Arg_1 = s32[10, 10] constant({{1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}})
    Arg_2 = s32[10, 10] constant({{1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}})
    Arg_3 = s32[10, 10] constant({{1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}})
    Arg_4 = s32[10, 10] constant({{1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}})
    Arg_5 = s32[10, 10] constant({{1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}})
    add1 = s32[10, 10] add(Arg_1, Arg_0)
    add2 = s32[10, 10] add(Arg_2, Arg_3)
    add3 = s32[10, 10] add(Arg_4, Arg_5)
    add8 = s32[10, 10] add(add1, add2)
    addv = s32[10, 10] add(add3, add8)
    ROOT ret = add(Arg_00, Arg_1_1)
  }

  region_B {
    Arg_00 = s32[] parameter(0)
    Arg_1_1 = s32[] parameter(1)
    Arg_0 = s32[10, 10] constant({{1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}})
    Arg_1 = s32[10, 10] constant({{1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}})
    Arg_2 = s32[10, 10] constant({{1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}})
    Arg_3 = s32[10, 10] constant({{1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}})
    Arg_4 = s32[10, 10] constant({{1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}})
    Arg_5 = s32[10, 10] constant({{1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10},
    {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}, {1,2,3,4,5,6,7,8,9,10}})
    add1 = s32[10, 10] add(Arg_1, Arg_0)
    add2 = s32[10, 10] add(Arg_2, Arg_3)
    add3 = s32[10, 10] add(Arg_4, Arg_5)
    add8 = s32[10, 10] add(add1, add2)
    addv = s32[10, 10] add(add3, add8)
    ROOT ret = add(Arg_00, Arg_1_1)
  }

  ENTRY main.15 {
    Arg_0.1 = s32[10]{0} parameter(0)
    constant.3 = s32[] constant(0)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=region_A
    Arg_1.2 = s32[15]{0} parameter(1)
    rd2 = s32[] reduce(Arg_1.2, constant.3), dimensions={0}, to_apply=region_B
    ROOT multiply.14 = s32[] multiply(rd1, rd2)
  }
  )";
  auto computation_names = RunDeduplicatePass(text, /*expect_true=*/false);
  // Will bail out when the total constant size is > 1KB.
  int region_b_count = 0;
  for (auto comp : computation_names) {
    region_b_count += (comp == "region_B");
  }
  EXPECT_EQ(region_b_count, 1);
  EXPECT_EQ(computation_names.size(), 3);
}

TEST_F(HloComputationDeduplicatorTest, DontRemoveRegionBDifferentcomp) {
  const std::string_view text = R"(
  HloModule DeDupTest, entry_computation_layout={(s32[10]{0},s32[15]{0})->s32[]}
  region_A {
    Arg_0.5 = s32[] parameter(0)
    Arg_1.6 = s32[] parameter(1)
    ROOT add.7 = s32[] multiply(Arg_0.5, Arg_1.6)
  }

  region_B {
    Arg_0.2 = s32[] parameter(0)
    Arg_1.3 = s32[] parameter(1)
    ROOT add.8 = s32[] add(Arg_0.2, Arg_1.3)
  }

  ENTRY main.15 {
    Arg_0.1 = s32[10]{0} parameter(0)
    constant.3 = s32[] constant(0)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=region_A
    Arg_1.2 = s32[15]{0} parameter(1)
    rd2 = s32[] reduce(Arg_1.2, constant.3), dimensions={0}, to_apply=region_B
    ROOT multiply.14 = s32[] multiply(rd1, rd2)
  }
  )";

  auto computation_names = RunDeduplicatePass(text, /*expect_true=*/false);
  // Region B should be preserved since operations are different (mult vs add).
  int region_b_count = 0;
  for (auto name : computation_names) {
    region_b_count += (name == "region_B");
  }
  EXPECT_EQ(region_b_count, 1);
  EXPECT_EQ(computation_names.size(), 3);
}

TEST_F(HloComputationDeduplicatorTest, DontRemoveRegionBDifferentType) {
  const std::string_view text = R"(
  HloModule DeDupTest, entry_computation_layout={(s32[10]{0},s16[15]{0})->s16[]}
  region_A {
    Arg_0.5 = s32[] parameter(0)
    Arg_1.6 = s32[] parameter(1)
    ROOT add.7 = s32[] multiply(Arg_0.5, Arg_1.6)
  }

  region_B {
    Arg_0.5 = s16[] parameter(0)
    Arg_1.6 = s16[] parameter(1)
    ROOT add.7 = s16[] multiply(Arg_0.5, Arg_1.6)
  }

  ENTRY main.15 {
    Arg_0.1 = s32[10]{0} parameter(0)
    constant.3 = s32[] constant(5)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=region_A
    Arg_1.2 = s16[15]{0} parameter(1)
    constant.4 = s16[] constant(5)
    rd2 = s16[] reduce(Arg_1.2, constant.4), dimensions={0}, to_apply=region_B
  }
  )";

  auto computation_names = RunDeduplicatePass(text, /*expect_true=*/false);
  // Region B should be preserved since operations are different (mult vs add).
  int region_b_count = 0;
  for (auto comp : computation_names) {
    region_b_count += (comp == "region_B");
  }
  EXPECT_EQ(region_b_count, 1);
  EXPECT_EQ(computation_names.size(), 3);
}

TEST_F(HloComputationDeduplicatorTest, DontRemoveRegionBEntryComp) {
  // Note: this test is hypothetical and just to check dedup.
  const std::string_view text = R"(
  HloModule DeDupTest, entry_computation_layout={(s32[10]{0},s32[15]{0})->s32[]}
  region_A1 {
    Arg_0.5 = s32[] parameter(0)
    Arg_1.6 = s32[] parameter(1)
    ROOT add.7 = s32[] multiply(Arg_0.5, Arg_1.6)
  }

   region_B1 {
    Arg_0.2 = s32[] parameter(0)
    Arg_1.3 = s32[] parameter(1)
    ROOT add.8 = s32[] add(Arg_0.2, Arg_1.3)
  }

  ENTRY region_B {
    Arg_0.1 = s32[10]{0} parameter(0)
    constant.3 = s32[] constant(0)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=region_A1
    Arg_1.2 = s32[15]{0} parameter(1)
    rd2 = s32[] reduce(Arg_1.2, constant.3), dimensions={0}, to_apply=region_B1
    ROOT multiply.14 = s32[] multiply(rd1, rd2)
  }

  region_A {
    Arg_0.1 = s32[10]{0} parameter(0)
    constant.3 = s32[] constant(0)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=region_A1
    Arg_1.2 = s32[15]{0} parameter(1)
    rd2 = s32[] reduce(Arg_1.2, constant.3), dimensions={0}, to_apply=region_B1
    ROOT multiply.14 = s32[] multiply(rd1, rd2)
  }
  )";
  auto computation_names = RunDeduplicatePass(text, /*expect_true=*/false);
  // Region B should be preserved since it is an entry. However it will be
  // renamed, so we can't check for region_b, so I check for 4 computations.
  EXPECT_EQ(computation_names.size(), 4);
}

TEST_F(HloComputationDeduplicatorTest, LargeSubComputationTest) {
  // We are creating two identical computation, but it should not dedup them
  // since the number of instructions in each region is > 128.
  const Shape shape = ShapeUtil::MakeScalarShape(S32);
  const int total_regions = 2;
  const int max_insns = 128;
  std::vector<HloComputation> comps;
  auto module = CreateNewVerifiedModule();
  for (int region = 0; region < total_regions; region++) {
    HloComputation::Builder builder("region_" + std::to_string(region));
    auto curr =
        builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a0"));
    auto next =
        builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "a1"));
    for (int i = 0; i < max_insns; i++) {
      next = builder.AddInstruction(
          HloInstruction::CreateBinary(shape, HloOpcode::kAdd, curr, next));
    }
    module->AddComputationAndUnifyNamesAndIds(builder.Build(), false);
  }
  HloComputation::Builder main("main_func");
  std::vector<HloInstruction *> insns;
  std::vector<HloInstruction *> consts;
  for (int region = 0; region < total_regions; region++) {
    insns.push_back(main.AddInstruction(
        HloInstruction::CreateParameter(region, ShapeUtil::MakeShape(S32, {10}),
                                        "a" + std::to_string(region))));
    consts.push_back(main.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32_t>(5 + region))));
  }
  int region = 0;
  for (auto comp : module->computations()) {
    ASSERT_LT(region, total_regions);
    main.AddInstruction(HloInstruction::CreateReduce(
        ShapeUtil::MakeScalarShape(S32), insns[region], consts[region],
        /*dimensions_to_reduce=*/{0}, comp));
  }
  module->AddEntryComputation(main.Build());
  HloComputationDeduplicator dedup;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, dedup.Run(module.get()));
  EXPECT_FALSE(changed);
  std::vector<HloComputation *> computations = module->MakeComputationSorted();
  EXPECT_EQ(computations.size(), (total_regions + 1));
}

TEST_F(HloComputationDeduplicatorTest, DontDeduplicateReduceAllReduce) {
  // Note: this test is hypothetical and just to check dedup.
  const std::string_view text = R"(
  HloModule TestModule

  add.1 {
    Arg_0 = s32[] parameter(0)
    Arg_1 = s32[] parameter(1)
    ROOT add.2 = s32[] add(Arg_0, Arg_1)
  }
  add.2 {
    Arg_0 = s32[] parameter(0)
    Arg_1 = s32[] parameter(1)
    ROOT add.2 = s32[] add(Arg_0, Arg_1)
  }

  ENTRY main {
    Arg_0.1 = s32[10] parameter(0)
    constant.3 = s32[] constant(0)
    rd1 = s32[] reduce(Arg_0.1, constant.3), dimensions={0}, to_apply=add.1
    Arg_1.1 = s32[] parameter(1)
    rd2 = s32[] all-reduce(Arg_1.1), to_apply=add.2
    ROOT multiply.14 = s32[] multiply(rd1, rd2)
  }
  )";
  auto computation_names = RunDeduplicatePass(text, /*expect_true=*/false);
  EXPECT_EQ(computation_names.size(), 3);
}
}  //  namespace
}  //  namespace xla
