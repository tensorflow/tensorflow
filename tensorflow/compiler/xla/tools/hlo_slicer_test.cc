/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tools/hlo_slicer.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = testing::opcode_matchers;

using HloSlicerTest = HloTestBase;

TEST_F(HloSlicerTest, SingleComputationSlice) {
  const std::string& hlo_string = R"(
  HloModule axpy_module
    ENTRY axpy_computation {
      p.0 = f32[10] parameter(0)
      p.1 = f32[10] parameter(1)
      add.0 = f32[10] add(p.0, p.1)
      alpha = f32[] constant(1)
      broadcast = f32[10] broadcast(alpha), dimensions={}
      p.2 = f32[10] parameter(2)
      y = f32[10] multiply(broadcast, p.2)
      x = f32[10] subtract(y, add.0)
      p.3 = f32[10] parameter(3)
      ROOT add.1 = f32[10] add(x, p.3)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto p2 = FindInstruction(hlo_module.get(), "p.2");
  EXPECT_THAT(p2, op::Parameter());
  auto p3 = FindInstruction(hlo_module.get(), "p.3");
  EXPECT_THAT(p3, op::Parameter());
  auto x = FindInstruction(hlo_module.get(), "x");
  EXPECT_THAT(x, op::Subtract());
  auto y = FindInstruction(hlo_module.get(), "y");
  EXPECT_THAT(y, op::Multiply());
  auto add0 = FindInstruction(hlo_module.get(), "add.0");
  EXPECT_THAT(add0, op::Add());
  auto add1 = FindInstruction(hlo_module.get(), "add.1");
  EXPECT_THAT(add1, op::Add());

  auto entry_comp = FindComputation(hlo_module.get(), "axpy_computation");
  EXPECT_NE(entry_comp, nullptr);

  {
    std::vector<const HloInstruction*> relevant_instructions({p2, x});
    auto sliced_result = SliceModule(hlo_module.get(), relevant_instructions);
    EXPECT_EQ(sliced_result.size(), 1);
    EXPECT_EQ(sliced_result[entry_comp].size(), 4);
    EXPECT_TRUE(sliced_result[entry_comp].contains(p2));
    EXPECT_TRUE(sliced_result[entry_comp].contains(x));
    EXPECT_TRUE(sliced_result[entry_comp].contains(y));
    EXPECT_TRUE(sliced_result[entry_comp].contains(add1));
  }

  {
    std::vector<const HloInstruction*> relevant_instructions({add0, p3});
    auto sliced_result = SliceModule(hlo_module.get(), relevant_instructions);
    EXPECT_EQ(sliced_result.size(), 1);
    EXPECT_EQ(sliced_result[entry_comp].size(), 4);
    EXPECT_TRUE(sliced_result[entry_comp].contains(add0));
    EXPECT_TRUE(sliced_result[entry_comp].contains(x));
    EXPECT_TRUE(sliced_result[entry_comp].contains(p3));
    EXPECT_TRUE(sliced_result[entry_comp].contains(add1));
  }
}

TEST_F(HloSlicerTest, MultipleComputationSlice) {
  const std::string& hlo_string = R"(
  HloModule test
  
  calculate_alpha {
    constant.5 = s32[] constant(2)
    constant.6 = s32[] constant(3)
    ROOT ret = s32[] subtract(constant.5, constant.6)
  }

  While.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get_tuple_element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(23)
    add.3 = s32[] add(get_tuple_element.1, constant.1)
    get_tuple_element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get_tuple_element.2, get_tuple_element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add.3, multiply)
  }

  While.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get_tuple_element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(100)
    ROOT less_than = pred[] compare(get_tuple_element.3, constant.2), direction=LT
  } 
  
  ENTRY Test {
    p.1 = s32[] parameter(0)
    p.2 = s32[] parameter(1)
    add.1 = s32[] add(p.1, p.2)
    constant.3 = s32[] call(), to_apply=calculate_alpha
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    while.1 = (s32[], s32[3]{0}) while(tuple.1), condition=While.condition, body=While.body
    loop_count = s32[] get-tuple-element(while.1), index=0
    ROOT add.2 = s32[] add(loop_count, add.1)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto add1 = FindInstruction(hlo_module.get(), "add.1");
  EXPECT_THAT(add1, op::Add());
  auto while1 = FindInstruction(hlo_module.get(), "while.1");
  EXPECT_THAT(while1, op::While());
  auto loop_count = FindInstruction(hlo_module.get(), "loop_count");
  EXPECT_THAT(loop_count, op::GetTupleElement());
  auto add2 = FindInstruction(hlo_module.get(), "add.2");
  EXPECT_THAT(add2, op::Add());
  auto gte1 = FindInstruction(hlo_module.get(), "get_tuple_element.1");
  EXPECT_THAT(gte1, op::GetTupleElement());
  auto gte2 = FindInstruction(hlo_module.get(), "get_tuple_element.2");
  EXPECT_THAT(gte2, op::GetTupleElement());
  auto constant5 = FindInstruction(hlo_module.get(), "constant.5");
  EXPECT_THAT(constant5, op::Constant());
  auto tuple1 = FindInstruction(hlo_module.get(), "tuple.1");
  EXPECT_THAT(tuple1, op::Tuple());

  auto entry_comp = FindComputation(hlo_module.get(), "Test");
  EXPECT_NE(entry_comp, nullptr);
  auto while_cond_comp = FindComputation(hlo_module.get(), "While.condition");
  EXPECT_NE(while_cond_comp, nullptr);
  auto while_body_comp = FindComputation(hlo_module.get(), "While.body");
  EXPECT_NE(while_body_comp, nullptr);
  auto calculate_alpha_comp =
      FindComputation(hlo_module.get(), "calculate_alpha");
  EXPECT_NE(calculate_alpha_comp, nullptr);

  {
    std::vector<const HloInstruction*> relevant_instructions({add1, while1});
    auto sliced_result = SliceModule(hlo_module.get(), relevant_instructions);
    EXPECT_EQ(sliced_result.size(), 1);
    EXPECT_EQ(sliced_result[entry_comp].size(), 4);
    EXPECT_TRUE(sliced_result[entry_comp].contains(add2));
    EXPECT_TRUE(sliced_result[entry_comp].contains(add1));
    EXPECT_TRUE(sliced_result[entry_comp].contains(while1));
    EXPECT_TRUE(sliced_result[entry_comp].contains(loop_count));
  }

  {
    std::vector<const HloInstruction*> relevant_instructions({constant5});
    auto sliced_result = SliceModule(hlo_module.get(), relevant_instructions);
    EXPECT_EQ(sliced_result.size(), 2);
    EXPECT_TRUE(sliced_result.contains(entry_comp));
    EXPECT_TRUE(sliced_result.contains(calculate_alpha_comp));
    EXPECT_FALSE(sliced_result[entry_comp].contains(add1));
  }

  {
    std::vector<const HloInstruction*> relevant_instructions({gte2});
    auto sliced_result = SliceModule(hlo_module.get(), relevant_instructions);
    EXPECT_EQ(sliced_result.size(), 2);
    EXPECT_TRUE(sliced_result.contains(entry_comp));
    EXPECT_TRUE(sliced_result.contains(while_body_comp));
    EXPECT_FALSE(sliced_result.contains(while_cond_comp));
    EXPECT_FALSE(sliced_result[entry_comp].contains(tuple1));
    EXPECT_FALSE(sliced_result[entry_comp].contains(add1));
    EXPECT_TRUE(sliced_result[entry_comp].contains(add2));
    EXPECT_FALSE(sliced_result[while_body_comp].contains(gte1));
  }
}

}  // namespace
}  // namespace xla
