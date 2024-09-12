/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/tools/hlo_slicer.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = testing::opcode_matchers;

using HloSlicerTest = HloTestBase;

TEST_F(HloSlicerTest, SingleComputationForwardSlice) {
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

  // Select everything
  auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
    return true;
  };

  {
    std::vector<const HloInstruction*> relevant_instructions({p2, x});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_instructions.size(), 1);
    EXPECT_EQ(sliced_instructions[entry_comp].size(), 4);
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(p2));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(x));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(y));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(add1));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 0);
  }

  {
    std::vector<const HloInstruction*> relevant_instructions({add0, p3});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_instructions.size(), 1);
    EXPECT_EQ(sliced_instructions[entry_comp].size(), 4);
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(add0));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(x));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(p3));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(add1));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 0);
  }
}

TEST_F(HloSlicerTest, MultipleComputationForwardSlice) {
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

  // Select everything
  auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
    return true;
  };

  {
    std::vector<const HloInstruction*> relevant_instructions({add1, while1});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_instructions.size(), 1);
    EXPECT_EQ(sliced_instructions[entry_comp].size(), 4);
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(add2));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(add1));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(while1));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(loop_count));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 0);
  }

  {
    std::vector<const HloInstruction*> relevant_instructions({constant5});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_instructions.size(), 2);
    EXPECT_TRUE(sliced_instructions.contains(entry_comp));
    EXPECT_TRUE(sliced_instructions.contains(calculate_alpha_comp));
    EXPECT_FALSE(sliced_instructions[entry_comp].contains(add1));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 0);
  }

  {
    std::vector<const HloInstruction*> relevant_instructions({gte2});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_instructions.size(), 2);
    EXPECT_TRUE(sliced_instructions.contains(entry_comp));
    EXPECT_TRUE(sliced_instructions.contains(while_body_comp));
    EXPECT_FALSE(sliced_instructions.contains(while_cond_comp));
    EXPECT_FALSE(sliced_instructions[entry_comp].contains(tuple1));
    EXPECT_FALSE(sliced_instructions[entry_comp].contains(add1));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(add2));
    EXPECT_FALSE(sliced_instructions[while_body_comp].contains(gte1));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 0);
  }
}

TEST_F(HloSlicerTest, SingleComputationForwardFrontier) {
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
      p.4 = f32[10] parameter(4)
      p.5 = f32[10] parameter(5)
      sub.1 = f32[10] subtract(p.4, p.5)
      add.2 = f32[10] add(p.3, sub.1)
      ROOT add.1 = f32[10] add(x, add.2)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto broadcast = FindInstruction(hlo_module.get(), "broadcast");
  EXPECT_THAT(broadcast, op::Broadcast());
  auto x = FindInstruction(hlo_module.get(), "x");
  EXPECT_THAT(x, op::Subtract());
  auto y = FindInstruction(hlo_module.get(), "y");
  EXPECT_THAT(y, op::Multiply());
  auto add0 = FindInstruction(hlo_module.get(), "add.0");
  EXPECT_THAT(add0, op::Add());
  auto p5 = FindInstruction(hlo_module.get(), "p.5");
  EXPECT_THAT(p5, op::Parameter());
  auto sub1 = FindInstruction(hlo_module.get(), "sub.1");
  EXPECT_THAT(sub1, op::Subtract());

  auto entry_comp = FindComputation(hlo_module.get(), "axpy_computation");
  EXPECT_NE(entry_comp, nullptr);

  {
    // Slice at Subtract
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kSubtract;
    };
    std::vector<const HloInstruction*> relevant_instructions({broadcast, add0});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_result.NumSlicedInstructions(), 4);
    EXPECT_EQ(sliced_instructions.size(), 1);
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(add0));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(broadcast));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(y));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(x));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 1);
    auto frontier_instructions = sliced_result.frontier_instructions();
    EXPECT_TRUE(frontier_instructions[entry_comp].contains(x));
  }

  {
    // Slice at Subtract
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kSubtract;
    };
    std::vector<const HloInstruction*> relevant_instructions({add0, y, p5});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_result.NumSlicedInstructions(), 5);
    EXPECT_EQ(sliced_instructions.size(), 1);
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(add0));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(y));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(x));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(p5));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(sub1));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 2);
    auto frontier_instructions = sliced_result.frontier_instructions();
    EXPECT_TRUE(frontier_instructions[entry_comp].contains(x));
    EXPECT_TRUE(frontier_instructions[entry_comp].contains(sub1));
  }
}

TEST_F(HloSlicerTest, MultipleComputationForwardFrontier) {
  const std::string& hlo_string = R"(
  HloModule axpy_module
    calculate_alpha {
      c.0 = f32[] constant(1)
      c.1 = f32[] constant(2)
      c.2 = f32[] add(c.0, c.1)
      c.3 = f32[] constant(4)
      ROOT ret = f32[] multiply(c.2, c.3)
    }
    
    ENTRY axpy_computation {
      p.0 = f32[] parameter(0)
      alpha = f32[] call(), to_apply=calculate_alpha
      ROOT add = f32[] add(p.0, alpha)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto entry_comp = FindComputation(hlo_module.get(), "axpy_computation");
  EXPECT_NE(entry_comp, nullptr);
  auto calculate_alpha_comp =
      FindComputation(hlo_module.get(), "calculate_alpha");
  EXPECT_NE(calculate_alpha_comp, nullptr);

  auto ret = FindInstruction(hlo_module.get(), "ret");
  EXPECT_THAT(ret, op::Multiply());
  auto c2 = FindInstruction(hlo_module.get(), "c.2");
  EXPECT_THAT(c2, op::Add());
  auto c3 = FindInstruction(hlo_module.get(), "c.3");
  EXPECT_THAT(c3, op::Constant());
  auto alpha = FindInstruction(hlo_module.get(), "alpha");
  EXPECT_THAT(alpha, op::Call());

  // Frontier at the root instruction of a callee computation.
  {
    auto hlo_selector = [&ret](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst != ret;
    };
    std::vector<const HloInstruction*> relevant_instructions({c2});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_result.NumSlicedInstructions(), 2);
    EXPECT_EQ(sliced_instructions.size(), 1);
    EXPECT_TRUE(sliced_instructions.contains(calculate_alpha_comp));
    EXPECT_EQ(sliced_instructions[calculate_alpha_comp].size(), 2);
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(c2));
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(ret));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 1);
    auto frontier_instructions = sliced_result.frontier_instructions();
    EXPECT_TRUE(frontier_instructions.contains(calculate_alpha_comp));
    EXPECT_TRUE(frontier_instructions[calculate_alpha_comp].contains(ret));
  }

  // Frontier at the callsite instruction.
  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kCall;
    };
    std::vector<const HloInstruction*> relevant_instructions({c2});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_instructions.size(), 2);

    EXPECT_TRUE(sliced_instructions.contains(calculate_alpha_comp));
    EXPECT_EQ(sliced_instructions[calculate_alpha_comp].size(), 2);
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(c2));
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(ret));

    EXPECT_TRUE(sliced_instructions.contains(entry_comp));
    EXPECT_EQ(sliced_instructions[entry_comp].size(), 1);
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(alpha));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 1);
    auto frontier_instructions = sliced_result.frontier_instructions();
    EXPECT_TRUE(frontier_instructions.contains(entry_comp));
    EXPECT_TRUE(frontier_instructions[entry_comp].contains(alpha));
  }
}

TEST_F(HloSlicerTest, SingleComputationBackwardSliceAndFrontier) {
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

  auto alpha = FindInstruction(hlo_module.get(), "alpha");
  EXPECT_THAT(alpha, op::Constant());
  auto p0 = FindInstruction(hlo_module.get(), "p.0");
  EXPECT_THAT(p0, op::Parameter());
  auto p1 = FindInstruction(hlo_module.get(), "p.1");
  EXPECT_THAT(p1, op::Parameter());
  auto p2 = FindInstruction(hlo_module.get(), "p.2");
  EXPECT_THAT(p2, op::Parameter());
  auto p3 = FindInstruction(hlo_module.get(), "p.3");
  EXPECT_THAT(p3, op::Parameter());
  auto broadcast = FindInstruction(hlo_module.get(), "broadcast");
  EXPECT_THAT(broadcast, op::Broadcast());
  auto x = FindInstruction(hlo_module.get(), "x");
  EXPECT_THAT(x, op::Subtract());
  auto y = FindInstruction(hlo_module.get(), "y");
  EXPECT_THAT(y, op::Multiply());
  auto add0 = FindInstruction(hlo_module.get(), "add.0");
  EXPECT_THAT(add0, op::Add());

  auto entry_comp = FindComputation(hlo_module.get(), "axpy_computation");
  EXPECT_NE(entry_comp, nullptr);

  // Select everything
  auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
    return true;
  };

  {
    std::vector<const HloInstruction*> relevant_instructions({y});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector,
        /*ignore_control_dependency=*/false, /*forward_slice=*/false);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_instructions.size(), 1);
    EXPECT_EQ(sliced_instructions[entry_comp].size(), 4);
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(y));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(broadcast));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(p2));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(alpha));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 0);
  }

  {
    std::vector<const HloInstruction*> relevant_instructions({add0, y});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector,
        /*ignore_control_dependency=*/false, /*forward_slice=*/false);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_instructions.size(), 1);
    EXPECT_EQ(sliced_instructions[entry_comp].size(), 7);
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(y));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(broadcast));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(p2));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(alpha));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(add0));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(p0));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(p1));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 0);
  }

  // Testing backward frontier

  // Slice at Broadcast.
  auto broadcast_slicer = [](const HloInstruction* hlo_inst) -> bool {
    return hlo_inst->opcode() != HloOpcode::kBroadcast;
  };

  {
    std::vector<const HloInstruction*> relevant_instructions({y});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions),
        broadcast_slicer,
        /*ignore_control_dependency=*/false, /*forward_slice=*/false);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_instructions.size(), 1);
    EXPECT_EQ(sliced_instructions[entry_comp].size(), 3);
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(y));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(p2));
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(broadcast));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 1);
    auto frontier_instructions = sliced_result.frontier_instructions();
    EXPECT_TRUE(frontier_instructions[entry_comp].contains(broadcast));
  }
}

TEST_F(HloSlicerTest, MultipleComputationBackwardSliceAndFrontier) {
  const std::string& hlo_string = R"(
  HloModule axpy_module
    calculate_alpha {
      c.0 = f32[] constant(1)
      c.1 = f32[] constant(2)
      c.2 = f32[] add(c.0, c.1)
      c.3 = f32[] constant(4)
      ROOT ret = f32[] multiply(c.2, c.3)
    }
    
    ENTRY axpy_computation {
      p.0 = f32[] parameter(0)
      alpha = f32[] call(), to_apply=calculate_alpha
      ROOT add = f32[] add(p.0, alpha)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto entry_comp = FindComputation(hlo_module.get(), "axpy_computation");
  EXPECT_NE(entry_comp, nullptr);
  auto calculate_alpha_comp =
      FindComputation(hlo_module.get(), "calculate_alpha");
  EXPECT_NE(calculate_alpha_comp, nullptr);

  auto ret = FindInstruction(hlo_module.get(), "ret");
  EXPECT_THAT(ret, op::Multiply());
  auto c0 = FindInstruction(hlo_module.get(), "c.0");
  EXPECT_THAT(c0, op::Constant());
  auto c1 = FindInstruction(hlo_module.get(), "c.1");
  EXPECT_THAT(c1, op::Constant());
  auto c2 = FindInstruction(hlo_module.get(), "c.2");
  EXPECT_THAT(c2, op::Add());
  auto c3 = FindInstruction(hlo_module.get(), "c.3");
  EXPECT_THAT(c3, op::Constant());
  auto alpha = FindInstruction(hlo_module.get(), "alpha");
  EXPECT_THAT(alpha, op::Call());

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return true;
    };
    std::vector<const HloInstruction*> relevant_instructions({c2});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector,
        /*ignore_control_dependency=*/false, /*forward_slice=*/false);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_result.NumSlicedInstructions(), 3);
    EXPECT_EQ(sliced_instructions.size(), 1);
    EXPECT_TRUE(sliced_instructions.contains(calculate_alpha_comp));
    EXPECT_EQ(sliced_instructions[calculate_alpha_comp].size(), 3);
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(c0));
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(c1));
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(c2));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 0);
  }

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return true;
    };
    std::vector<const HloInstruction*> relevant_instructions({alpha});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), hlo_selector,
        /*ignore_control_dependency=*/false, /*forward_slice=*/false);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_instructions.size(), 2);

    EXPECT_TRUE(sliced_instructions.contains(calculate_alpha_comp));
    EXPECT_EQ(sliced_instructions[calculate_alpha_comp].size(), 5);
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(c0));
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(c1));
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(c2));
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(c3));
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(ret));

    EXPECT_TRUE(sliced_instructions.contains(entry_comp));
    EXPECT_EQ(sliced_instructions[entry_comp].size(), 1);
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(alpha));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 0);
  }

  // Testing backward slicing frontier.

  {
    auto add_slicer = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kAdd;
    };
    std::vector<const HloInstruction*> relevant_instructions({ret});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), add_slicer,
        /*ignore_control_dependency=*/false, /*forward_slice=*/false);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_result.NumSlicedInstructions(), 3);

    EXPECT_EQ(sliced_instructions.size(), 1);
    EXPECT_TRUE(sliced_instructions.contains(calculate_alpha_comp));
    EXPECT_EQ(sliced_instructions[calculate_alpha_comp].size(), 3);
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(ret));
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(c3));
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(c2));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 1);
    auto frontier_instructions = sliced_result.frontier_instructions();
    EXPECT_TRUE(frontier_instructions.contains(calculate_alpha_comp));
    EXPECT_TRUE(frontier_instructions[calculate_alpha_comp].contains(c2));
  }

  {
    auto mul_slicer = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kMultiply;
    };
    std::vector<const HloInstruction*> relevant_instructions({alpha});
    auto sliced_result = SliceModule(
        hlo_module.get(), absl::MakeSpan(relevant_instructions), mul_slicer,
        /*ignore_control_dependency=*/false, /*forward_slice=*/false);

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_result.NumSlicedInstructions(), 2);

    EXPECT_EQ(sliced_instructions.size(), 2);

    EXPECT_TRUE(sliced_instructions.contains(calculate_alpha_comp));
    EXPECT_EQ(sliced_instructions[calculate_alpha_comp].size(), 1);
    EXPECT_TRUE(sliced_instructions[calculate_alpha_comp].contains(ret));

    EXPECT_TRUE(sliced_instructions.contains(entry_comp));
    EXPECT_EQ(sliced_instructions[entry_comp].size(), 1);
    EXPECT_TRUE(sliced_instructions[entry_comp].contains(alpha));

    EXPECT_EQ(sliced_result.NumFrontierInstructions(), 1);
    auto frontier_instructions = sliced_result.frontier_instructions();
    EXPECT_TRUE(frontier_instructions.contains(calculate_alpha_comp));
    EXPECT_TRUE(frontier_instructions[calculate_alpha_comp].contains(ret));
  }
}

TEST_F(HloSlicerTest, ForwardSlicingNearestCommonAncestor) {
  const std::string& hlo_string = R"(
  HloModule module
    ENTRY computation {
      p.0 = f32[10] parameter(0)
      p.1 = f32[10] parameter(1)
      add.0 = f32[10] add(p.0, p.1)
      p.2 = f32[10] parameter(2)
      mul.0 = f32[10] multiply(p.1, p.2)
      sub.0 = f32[10] subtract(add.0, mul.0)
      add.1 = f32[10] add(add.0, p.2)
      ROOT add.2 = f32[10] add(sub.0, add.1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto p0 = FindInstruction(hlo_module.get(), "p.0");
  auto p2 = FindInstruction(hlo_module.get(), "p.2");
  auto mul0 = FindInstruction(hlo_module.get(), "mul.0");
  auto add0 = FindInstruction(hlo_module.get(), "add.0");
  auto sub0 = FindInstruction(hlo_module.get(), "sub.0");
  auto add1 = FindInstruction(hlo_module.get(), "add.1");
  const HloComputation* computation = hlo_module->entry_computation();

  {
    std::vector<const HloInstruction*> relevant_instructions({p0});
    auto sliced_result =
        SliceModule(hlo_module.get(), absl::MakeSpan(relevant_instructions),
                    /*frontier_selector=*/nullptr,
                    /*ignore_control_dependency=*/false, /*forward_slice=*/true,
                    /*nearest_common_ancestor_as_root=*/true);

    EXPECT_NE(sliced_result.nearest_common_ancestor_root(), nullptr);
    EXPECT_EQ(sliced_result.nearest_common_ancestor_root(), p0);
    EXPECT_EQ(sliced_result.NumSlicedInstructions(), 1);
  }

  {
    std::vector<const HloInstruction*> relevant_instructions({p0, p2});
    auto sliced_result =
        SliceModule(hlo_module.get(), absl::MakeSpan(relevant_instructions),
                    /*frontier_selector=*/nullptr,
                    /*ignore_control_dependency=*/false, /*forward_slice=*/true,
                    /*nearest_common_ancestor_as_root=*/true);

    EXPECT_NE(sliced_result.nearest_common_ancestor_root(), nullptr);
    EXPECT_TRUE(sliced_result.nearest_common_ancestor_root() == sub0 ||
                sliced_result.nearest_common_ancestor_root() == add1);
    EXPECT_TRUE(sliced_result.sliced_instructions().contains(computation));

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_TRUE(sliced_instructions[computation].contains(add0));
  }

  {
    std::vector<const HloInstruction*> relevant_instructions({p0, mul0});
    auto sliced_result =
        SliceModule(hlo_module.get(), absl::MakeSpan(relevant_instructions),
                    /*frontier_selector=*/nullptr,
                    /*ignore_control_dependency=*/false,
                    /*forward_slice=*/true,
                    /*nearest_common_ancestor_as_root=*/true);

    EXPECT_NE(sliced_result.nearest_common_ancestor_root(), nullptr);
    EXPECT_EQ(sliced_result.nearest_common_ancestor_root(), sub0);
    EXPECT_EQ(sliced_result.NumSlicedInstructions(), 4);

    EXPECT_TRUE(sliced_result.sliced_instructions().contains(computation));
    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_TRUE(sliced_instructions[computation].contains(p0));
    EXPECT_TRUE(sliced_instructions[computation].contains(add0));
    EXPECT_TRUE(sliced_instructions[computation].contains(mul0));
    EXPECT_TRUE(sliced_instructions[computation].contains(sub0));
  }
}

TEST_F(HloSlicerTest, MultipleComputationForwardSlicingNearestCommonAncestor) {
  const std::string& hlo_string = R"(
  HloModule axpy_module
    calculate_alpha {
      c.0 = f32[] constant(1)
      c.1 = f32[] constant(2)
      ROOT ret.0 = f32[] multiply(c.0, c.1)
    }
    
    calculate_y {
      c.2 = f32[] constant(2)
      c.3 = f32[] constant(3)
      ROOT ret.1 = f32[] add(c.2, c.3)
    }
    
    ENTRY axpy_computation {
      alpha = f32[] call(), to_apply=calculate_alpha
      y = f32[] call(), to_apply=calculate_y
      add.0 = f32[] add(alpha, y)
      p.0 = f32[] parameter(0)
      ROOT add.1 = f32[] add(add.0, p.0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto c0 = FindInstruction(hlo_module.get(), "c.0");
  auto ret0 = FindInstruction(hlo_module.get(), "ret.0");
  auto c2 = FindInstruction(hlo_module.get(), "c.2");
  auto ret1 = FindInstruction(hlo_module.get(), "ret.1");
  auto alpha = FindInstruction(hlo_module.get(), "alpha");
  auto y = FindInstruction(hlo_module.get(), "y");
  auto add0 = FindInstruction(hlo_module.get(), "add.0");

  const HloComputation* computation = hlo_module->entry_computation();
  const HloComputation* calculate_alpha =
      FindComputation(hlo_module.get(), "calculate_alpha");
  const HloComputation* calculate_y =
      FindComputation(hlo_module.get(), "calculate_y");

  {
    std::vector<const HloInstruction*> relevant_instructions({c0, c2});
    auto sliced_result =
        SliceModule(hlo_module.get(), absl::MakeSpan(relevant_instructions),
                    /*frontier_selector=*/nullptr,
                    /*ignore_control_dependency=*/false,
                    /*forward_slice=*/true,
                    /*nearest_common_ancestor_as_root=*/true);

    EXPECT_NE(sliced_result.nearest_common_ancestor_root(), nullptr);
    EXPECT_EQ(sliced_result.nearest_common_ancestor_root(), add0);

    EXPECT_EQ(sliced_result.sliced_instructions().size(), 3);
    EXPECT_TRUE(sliced_result.sliced_instructions().contains(computation));
    EXPECT_TRUE(sliced_result.sliced_instructions().contains(calculate_alpha));
    EXPECT_TRUE(sliced_result.sliced_instructions().contains(calculate_y));

    auto sliced_instructions = sliced_result.sliced_instructions();
    EXPECT_EQ(sliced_result.NumSlicedInstructions(), 7);
    EXPECT_TRUE(sliced_instructions[calculate_alpha].contains(c0));
    EXPECT_TRUE(sliced_instructions[calculate_alpha].contains(ret0));
    EXPECT_TRUE(sliced_instructions[calculate_y].contains(c2));
    EXPECT_TRUE(sliced_instructions[calculate_y].contains(ret1));
    EXPECT_TRUE(sliced_instructions[computation].contains(alpha));
    EXPECT_TRUE(sliced_instructions[computation].contains(y));
    EXPECT_TRUE(sliced_instructions[computation].contains(add0));
  }
}

TEST_F(HloSlicerTest, TestSliceModuleAndExtract) {
  const std::string& hlo_string = R"(
  HloModule axpy_module
    calculate_alpha {
      c.0 = f32[] constant(1)
      c.1 = f32[] constant(2)
      ROOT ret.0 = f32[] multiply(c.0, c.1)
    }
    
    calculate_y {
      c.2 = f32[] constant(2)
      c.3 = f32[] constant(3)
      ROOT ret.1 = f32[] add(c.2, c.3)
    }
    
    ENTRY axpy_computation {
      alpha = f32[] call(), to_apply=calculate_alpha
      y = f32[] call(), to_apply=calculate_y
      add.0 = f32[] add(alpha, y)
      p.0 = f32[] parameter(0)
      ROOT add.1 = f32[] add(add.0, p.0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto alpha = FindInstruction(hlo_module.get(), "alpha");
  auto y = FindInstruction(hlo_module.get(), "y");
  auto add0 = FindInstruction(hlo_module.get(), "add.0");

  // slice_starting_instructions: {alpha, y}.
  // forward_slicing: kNca.
  // backward_slicing: true.
  {
    std::vector<const HloInstruction*> relevant_instructions({alpha, y});
    SlicingConfiguration slicing_config = {
        /*forward_slicing=*/SlicingConfiguration::ForwardSlicingConfig::kNca,
        /*backward_slicing=*/true};
    std::vector<std::unique_ptr<HloModule>> sliced_modules =
        SliceModuleAndExtract(hlo_module.get(),
                              /*slice_starting_instructions=*/
                              absl::MakeSpan(relevant_instructions),
                              /*slicing_configuration=*/slicing_config);
    CHECK_EQ(sliced_modules.size(), 1);
    auto sliced_module = std::move(sliced_modules[0]);

    // Test forward slicing: the extracted module should root at `add.0`, which
    // is the nearest common ancestor of `alpha` and `y`.
    EXPECT_EQ(sliced_module->entry_computation()->root_instruction()->name(),
              "add.0");
    EXPECT_EQ(sliced_module->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kAdd);

    // Test backward slicing: the extracted module should contain all three
    // computations and all the "leaf instructions".
    EXPECT_EQ(sliced_module->computation_count(), 3);
    HloInstruction* c0 = FindInstruction(sliced_module.get(), "c.0");
    EXPECT_NE(c0, nullptr);
    HloInstruction* c1 = FindInstruction(sliced_module.get(), "c.1");
    EXPECT_NE(c1, nullptr);
    HloInstruction* c2 = FindInstruction(sliced_module.get(), "c.2");
    EXPECT_NE(c2, nullptr);
    HloInstruction* c3 = FindInstruction(sliced_module.get(), "c.3");
    EXPECT_NE(c3, nullptr);
  }

  // slice_starting_instructions: {alpha, y}.
  // forward_slicing: kRoot.
  // backward_slicing: true.
  {
    std::vector<const HloInstruction*> relevant_instructions({alpha, y});
    SlicingConfiguration slicing_config = {
        /*forward_slicing=*/SlicingConfiguration::ForwardSlicingConfig::kRoot,
        /*backward_slicing=*/true};
    std::vector<std::unique_ptr<HloModule>> sliced_modules =
        SliceModuleAndExtract(hlo_module.get(),
                              /*slice_starting_instructions=*/
                              absl::MakeSpan(relevant_instructions),
                              /*slicing_configuration=*/slicing_config);
    CHECK_EQ(sliced_modules.size(), 1);
    auto sliced_module = std::move(sliced_modules[0]);

    // Test forward slicing: the extracted module should root at `add.1`, which
    // is the original root instruction of entry computation.
    EXPECT_EQ(sliced_module->entry_computation()->root_instruction()->name(),
              "add.1");
    EXPECT_EQ(sliced_module->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kAdd);

    // Test backward slicing: the extracted module should contain all three
    // computations and all the "leaf instructions".
    EXPECT_EQ(sliced_module->computation_count(), 3);
    HloInstruction* c0 = FindInstruction(sliced_module.get(), "c.0");
    EXPECT_NE(c0, nullptr);
    HloInstruction* c1 = FindInstruction(sliced_module.get(), "c.1");
    EXPECT_NE(c1, nullptr);
    HloInstruction* c2 = FindInstruction(sliced_module.get(), "c.2");
    EXPECT_NE(c2, nullptr);
    HloInstruction* c3 = FindInstruction(sliced_module.get(), "c.3");
    EXPECT_NE(c3, nullptr);
  }

  // slice_starting_instructions: {y}.
  // forward_slicing: kRoot.
  // backward_slicing: true.
  {
    std::vector<const HloInstruction*> relevant_instructions({y});
    SlicingConfiguration slicing_config = {
        /*forward_slicing=*/SlicingConfiguration::ForwardSlicingConfig::kRoot,
        /*backward_slicing=*/true};
    std::vector<std::unique_ptr<HloModule>> sliced_modules =
        SliceModuleAndExtract(hlo_module.get(),
                              /*slice_starting_instructions=*/
                              absl::MakeSpan(relevant_instructions),
                              /*slicing_configuration=*/slicing_config);
    CHECK_EQ(sliced_modules.size(), 1);
    auto sliced_module = std::move(sliced_modules[0]);

    // Test forward slicing: the extracted module should root at `add.1`, which
    // is the original root instruction of entry computation.
    EXPECT_EQ(sliced_module->entry_computation()->root_instruction()->name(),
              "add.1");
    EXPECT_EQ(sliced_module->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kAdd);

    // Test backward slicing: The computation `axpy_computation` and
    // `calculate_y` should be included (so as instructions `c2` and `c3`),
    // while the computation `calculate_alpha` should not be included (so as
    // instructions `c0` and `c1`).
    EXPECT_EQ(sliced_module->computation_count(), 2);
    HloInstruction* c0 = FindInstruction(sliced_module.get(), "c.0");
    EXPECT_EQ(c0, nullptr);
    HloInstruction* c1 = FindInstruction(sliced_module.get(), "c.1");
    EXPECT_EQ(c1, nullptr);
    HloInstruction* c2 = FindInstruction(sliced_module.get(), "c.2");
    EXPECT_NE(c2, nullptr);
    HloInstruction* c3 = FindInstruction(sliced_module.get(), "c.3");
    EXPECT_NE(c3, nullptr);
  }

  // slice_starting_instructions: {alpha, y}.
  // forward_slicing: kRoot.
  // backward_slicing: false.
  {
    std::vector<const HloInstruction*> relevant_instructions({add0});
    SlicingConfiguration slicing_config = {
        /*forward_slicing=*/SlicingConfiguration::ForwardSlicingConfig::kRoot,
        /*backward_slicing=*/false};
    std::vector<std::unique_ptr<HloModule>> sliced_modules =
        SliceModuleAndExtract(hlo_module.get(),
                              /*slice_starting_instructions=*/
                              absl::MakeSpan(relevant_instructions),
                              /*slicing_configuration=*/slicing_config);
    CHECK_EQ(sliced_modules.size(), 1);
    auto sliced_module = std::move(sliced_modules[0]);

    // Test forward slicing: the extracted module should root at `add.1`, which
    // is the original root instruction of entry computation.
    EXPECT_EQ(sliced_module->entry_computation()->root_instruction()->name(),
              "add.1");
    EXPECT_EQ(sliced_module->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kAdd);

    // Test backward slicing: The computation `calculate_alpha` and
    // `calculate_y` should not be included.
    EXPECT_EQ(sliced_module->computation_count(), 1);
  }
}

TEST_F(HloSlicerTest, TestSliceModuleAndExtractRemoveSharding) {
  const std::string& hlo_string = R"(
  HloModule axpy_module
    ENTRY axpy_computation {
    %constant.39733 = bf16[] constant(111)
    %broadcast.39734 = bf16[8,1,12288]{2,1,0} broadcast(bf16[] %constant.39733), dimensions={}
    %multiply.39766 = bf16[8,1,12288]{2,1,0} multiply(bf16[8,1,12288]{2,1,0} %broadcast.39734, bf16[8,1,12288]{2,1,0} %broadcast.39734)
    %custom-call.39767 = bf16[8,1,12288]{2,1,0} custom-call(bf16[8,1,12288]{2,1,0} %multiply.39766), custom_call_target="Sharding", sharding={replicated}
    ROOT %add.39786 = bf16[8,1,12288]{2,1,0} add(bf16[8,1,12288]{2,1,0} %custom-call.39767, bf16[8,1,12288]{2,1,0} %custom-call.39767)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* multiply_39766 =
      FindInstruction(hlo_module.get(), "multiply.39766");

  // slice_starting_instructions: {multiply_39766 }.
  // forward_slicing: kRoot.
  // backward_slicing: false.
  // remove_sharding: true.
  {
    std::vector<const HloInstruction*> relevant_instructions({multiply_39766});
    SlicingConfiguration slicing_config = {
        /*forward_slicing=*/SlicingConfiguration::ForwardSlicingConfig::kRoot,
        /*backward_slicing=*/false, /*remove_sharding=*/true};
    std::vector<std::unique_ptr<HloModule>> sliced_modules =
        SliceModuleAndExtract(hlo_module.get(),
                              /*slice_starting_instructions=*/
                              absl::MakeSpan(relevant_instructions),
                              /*slicing_configuration=*/slicing_config);
    EXPECT_EQ(sliced_modules.size(), 1);
    auto sliced_module = std::move(sliced_modules[0]);

    // Test if the custom-call to sharding is removed.
    for (HloInstruction* instruction :
         sliced_module->entry_computation()->instructions()) {
      EXPECT_NE(instruction->opcode(), HloOpcode::kCustomCall);
    }

    // Check that both the operands of %add.39786 are %multiply.39766.
    for (HloInstruction* instruction :
         sliced_module->entry_computation()->root_instruction()->operands()) {
      EXPECT_EQ(instruction->name(), "multiply.39766");
    }
  }
}

TEST_F(HloSlicerTest, TestSliceModuleAndExtractReduceTupleParameter) {
  const std::string& hlo_string = R"(
  HloModule axpy_module
    ENTRY axpy_computation (p.0: (s32[], s32[3]{0}), p.1: (s32[3]{0}, s32[])) -> s32[] {
      p.0 = (s32[], s32[3]{0}) parameter(0)
      gte.0 = s32[] get-tuple-element(p.0), index=0    
      p.1 = (s32[3]{0}, s32[]) parameter(1)
      gte.1 = s32[] get-tuple-element(p.1), index=1    
      ROOT add.0 = s32[] add(gte.0, gte.1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* add_0 = FindInstruction(hlo_module.get(), "add.0");
  CHECK_NE(add_0, nullptr);

  // slice_starting_instructions: {add.0}.
  // forward_slicing: kRoot.
  // backward_slicing: true.
  // remove_sharding: false.
  // reduce_tuple_parameter: true.
  {
    // Slice the whole hlo module and reduce the tuple parameter (p.0 and p.1).
    std::vector<const HloInstruction*> relevant_instructions({add_0});
    SlicingConfiguration slicing_config = {
        /*forward_slicing=*/SlicingConfiguration::ForwardSlicingConfig::kRoot,
        /*backward_slicing=*/true, /*remove_sharding=*/false,
        /*reduce_tuple_parameter=*/true};
    std::vector<std::unique_ptr<HloModule>> sliced_modules =
        SliceModuleAndExtract(hlo_module.get(),
                              /*slice_starting_instructions=*/
                              absl::MakeSpan(relevant_instructions),
                              /*slicing_configuration=*/slicing_config);
    EXPECT_EQ(sliced_modules.size(), 1);
    auto sliced_module = std::move(sliced_modules[0]);

    // Check that the new p.0 only has one element.
    HloInstruction* p_0 = FindInstruction(sliced_module.get(), "p.0");
    EXPECT_NE(p_0, nullptr);
    EXPECT_EQ(p_0->shape().tuple_shapes_size(), 1);

    // Check that the new p.1 only has one element.
    HloInstruction* p_1 = FindInstruction(sliced_module.get(), "p.1");
    EXPECT_NE(p_1, nullptr);
    EXPECT_EQ(p_1->shape().tuple_shapes_size(), 1);
  }
}

TEST_F(HloSlicerTest, TestSliceModuleAndExtractSlicingGroup) {
  const std::string& hlo_string = R"(
  HloModule axpy_module
    ENTRY axpy_computation (p.0: (s32[], s32[3]{0}), p.1: (s32[3]{0}, s32[])) -> s32[] {
      p.0 = (s32[], s32[3]{0}) parameter(0)
      gte.0 = s32[] get-tuple-element(p.0), index=0    
      p.1 = (s32[3]{0}, s32[]) parameter(1)
      gte.1 = s32[] get-tuple-element(p.1), index=1    
      ROOT add.0 = s32[] add(gte.0, gte.1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* gte_0 = FindInstruction(hlo_module.get(), "gte.0");
  CHECK_NE(gte_0, nullptr);
  HloInstruction* gte_1 = FindInstruction(hlo_module.get(), "gte.1");
  CHECK_NE(gte_1, nullptr);

  // slice_starting_instructions: {gte.0, gte.1}.
  // forward_slicing: kNca.
  // backward_slicing: true.
  // remove_sharding: false.
  // reduce_tuple_parameter: false.
  // slicing_group: 1
  {
    // Generate two sliced modules, sliced from gte.0 and gte.1, respectively
    // (`slicing_group` = 1).
    std::vector<const HloInstruction*> relevant_instructions({gte_0, gte_1});
    SlicingConfiguration slicing_config = {
        /*forward_slicing=*/SlicingConfiguration::ForwardSlicingConfig::kNca,
        /*backward_slicing=*/true, /*remove_sharding=*/false,
        /*reduce_tuple_parameter=*/false, /*slicing_group=*/1};
    std::vector<std::unique_ptr<HloModule>> sliced_modules =
        SliceModuleAndExtract(hlo_module.get(),
                              /*slice_starting_instructions=*/
                              absl::MakeSpan(relevant_instructions),
                              /*slicing_configuration=*/slicing_config);

    // There are two sliced module.
    EXPECT_EQ(sliced_modules.size(), 2);

    // The first sliced module contains gte.0 and p.0.
    auto sliced_module_0 = std::move(sliced_modules[0]);
    EXPECT_EQ(sliced_module_0->entry_computation()->instruction_count(), 2);
    HloInstruction* p_0 = FindInstruction(sliced_module_0.get(), "p.0");
    EXPECT_NE(p_0, nullptr);

    // The second sliced module contains gte.1 and p.1.
    auto sliced_module_1 = std::move(sliced_modules[1]);
    EXPECT_EQ(sliced_module_0->entry_computation()->instruction_count(), 2);
    HloInstruction* p_1 = FindInstruction(sliced_module_1.get(), "p.1");
    EXPECT_NE(p_1, nullptr);
  }
}

}  // namespace
}  // namespace xla
