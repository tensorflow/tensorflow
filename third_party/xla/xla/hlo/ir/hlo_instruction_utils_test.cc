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

#include "xla/hlo/ir/hlo_instruction_utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"

namespace xla {

namespace hlo_instruction_utils {

namespace {

class HloInstructionUtilsTest : public HloHardwareIndependentTestBase {};

TEST_F(HloInstructionUtilsTest, TestIsUnstridedSlice) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY main {
      param = f32[2,8] parameter(0)
      strided_slice = f32[2,2] slice(param), slice={[0:2:1], [4:8:2]}
      unstrided_slice = f32[2,4] slice(param), slice={[0:2:1], [4:8:1]}
      ROOT tuple = (f32[2,2], f32[2,4]) tuple(strided_slice, unstrided_slice)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  HloInstruction* unstrided_slice =
      hlo_query::FindInstruction(m->entry_computation(), "unstrided_slice");
  HloInstruction* strided_slice =
      hlo_query::FindInstruction(m->entry_computation(), "strided_slice");
  EXPECT_NE(unstrided_slice, nullptr);
  EXPECT_NE(strided_slice, nullptr);
  EXPECT_TRUE(IsUnstridedSlice(unstrided_slice));
  EXPECT_FALSE(IsUnstridedSlice(strided_slice));
}

TEST_F(HloInstructionUtilsTest, TestAddOrUpdateVectorOfPairsAsAttribute) {
  const char* hlo = R"(
    HloModule test
    ENTRY main {
      ROOT param = s32[] parameter(0), frontend_attributes={foo="bar", baz="qux"}
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
  HloInstruction* param = m->entry_computation()->root_instruction();
  EXPECT_EQ(param->frontend_attributes().map().size(), 2);
  EXPECT_EQ(param->frontend_attributes().map().at("foo"), "bar");
  EXPECT_EQ(param->frontend_attributes().map().at("baz"), "qux");

  std::string new_key = "quux";
  std::vector<std::pair<int64_t, int64_t>> value = {{1, 2}, {3, 4}};
  AddOrUpdateVectorOfPairsAsAttribute(param, new_key, value);
  EXPECT_EQ(param->frontend_attributes().map().size(), 3);
  EXPECT_EQ(param->frontend_attributes().map().at("foo"), "bar");
  EXPECT_EQ(param->frontend_attributes().map().at("baz"), "qux");
  EXPECT_EQ(param->frontend_attributes().map().at("quux"), "{{1,2},{3,4}}");

  std::vector<std::pair<int64_t, int64_t>> new_value = {{5, 6}, {7, 8}};
  AddOrUpdateVectorOfPairsAsAttribute(param, new_key, new_value);
  EXPECT_EQ(param->frontend_attributes().map().size(), 3);
  EXPECT_EQ(param->frontend_attributes().map().at("foo"), "bar");
  EXPECT_EQ(param->frontend_attributes().map().at("baz"), "qux");
  EXPECT_EQ(param->frontend_attributes().map().at("quux"), "{{5,6},{7,8}}");
}

}  // namespace

}  // namespace hlo_instruction_utils

}  // namespace xla
