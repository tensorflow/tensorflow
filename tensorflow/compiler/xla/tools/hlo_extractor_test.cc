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

#include "tensorflow/compiler/xla/tools/hlo_extractor.h"

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = testing::opcode_matchers;

using HloExtractorTest = HloTestBase;

TEST_F(HloExtractorTest, ExtractTopLevel) {
  const std::string& hlo_string = R"(
HloModule test

ENTRY %entry {
  param.0 = f32[4]{0} parameter(0)
  negate = f32[4]{0} negate(f32[4]{0} param.0)
  ROOT exp = f32[4]{0} exponential(f32[4]{0} negate)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "exp"));
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Exp(op::Negate(op::Parameter(0))));
  }

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "exp"), /*height=*/0);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Exp(op::Parameter(0)));
  }

  {
    auto extracted_module = ExtractModule(
        FindInstruction(hlo_module.get(), "negate"), /*height=*/0);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Negate(op::Parameter(0)));
  }
}

TEST_F(HloExtractorTest, ExtractDag) {
  const std::string& hlo_string = R"(
HloModule test

ENTRY %entry {
  param.0 = f32[4]{0} parameter(0)
  tanh = f32[4]{0} tanh(f32[4]{0} param.0)
  negate = f32[4]{0} negate(f32[4]{0} tanh)
  exp = f32[4]{0} exponential(f32[4]{0} negate)
  ROOT add = f32[4]{0} add(f32[4]{0} negate, f32[4]{0} exp)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "exp"));
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Exp(op::Negate(op::Tanh(op::Parameter(0)))));
  }

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"), /*height=*/0);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Parameter(0), op::Parameter(1)));
  }

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"), /*height=*/1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Negate(op::Parameter(0)),
                        op::Exp(op::Negate(op::Parameter(0)))));
  }

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"), /*height=*/2);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Negate(op::Tanh(op::Parameter(0))),
                        op::Exp(op::Negate(op::Tanh(op::Parameter(0))))));
  }
}

TEST_F(HloExtractorTest, ExtractWithConstant) {
  const std::string& hlo_string = R"(
HloModule test

ENTRY %entry {
  p = f32[4]{0} parameter(0)
  tanh = f32[4]{0} tanh(p)
  c = f32[4]{0} constant({1, 2, 3, 4})
  ROOT add = f32[4]{0} add(tanh, c)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"), /*height=*/0);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Parameter(0), op::Parameter(1)));
  }

  {
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "add"), /*height=*/1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Tanh(op::Parameter(0)), op::Constant()));
  }
}

TEST_F(HloExtractorTest, ExtractFromMultipleComputation) {
  const std::string& hlo_string = R"(
  HloModule axpy_module
    calculate_alpha {
      c.1 = f32[] constant(1)
      c.2 = f32[] constant(2)
      c.3 = f32[] add(c.1, c.2)
      c.4 = f32[] constant(4)
      ROOT ret = f32[] subtract(c.4, c.3)
    }
    
    ENTRY axpy_computation {
      alpha = f32[] call(), to_apply=calculate_alpha
      broadcast = f32[10] broadcast(alpha), dimensions={}
      x = f32[10] parameter(0)
      ax = f32[10] multiply(broadcast, x)
      y = f32[10] parameter(1)
      ROOT add = f32[10] add(ax, y)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  {
    // Find the kSubtract instruction in computation: calculate_alpha
    HloInstruction* inst =
        FindInstruction(hlo_module.get(), HloOpcode::kSubtract);
    EXPECT_NE(inst, nullptr);
    EXPECT_THAT(inst, op::Subtract());

    // Extract from the non-entry computation, with kSubstract instruction as
    // the new root instruction
    auto extracted_module = ExtractModule(inst, /*height=*/1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Subtract(op::Constant(), op::Add()));
    EXPECT_EQ(extracted_module->computation_count(), 1);
  }

  {
    // FindInstruction iterates from the ENTRY computation, therefore, it
    // matches to the kAdd instruction at the entry computation, instead of the
    // kAdd instruction in the Computation:calculate_alpha
    HloInstruction* inst = FindInstruction(hlo_module.get(), "add");
    EXPECT_NE(inst, nullptr);
    EXPECT_THAT(inst, op::Add(op::Multiply(), op::Parameter()));

    auto extracted_module = ExtractModule(inst);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Multiply(), op::Parameter()));
    EXPECT_EQ(extracted_module->computation_count(), 2);
  }
}

TEST_F(HloExtractorTest, HloSelector) {
  const std::string& hlo_string = R"(
  HloModule axpy_module
    calculate_alpha {
      c.1 = f32[] constant(1)
      c.2 = f32[] constant(2)
      c.3 = f32[] add(c.1, c.2)
      c.4 = f32[] constant(4)
      ROOT ret = f32[] multiply(c.4, c.3)
    }
    
    ENTRY axpy_computation {
      p.0 = f32[10] parameter(0)
      p.1 = f32[10] parameter(1)
      add.0 = f32[10] add(p.0, p.1)
      alpha = f32[] call(), to_apply=calculate_alpha
      broadcast = f32[10] broadcast(alpha), dimensions={}
      p.2 = f32[10] parameter(2)
      y = f32[10] multiply(broadcast, p.2)
      x = f32[10] subtract(y, add.0)
      p.3 = f32[10] parameter(3)
      ROOT add = f32[10] add(x, p.3)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Find the kSubtract instruction in ENTRY computation
  HloInstruction* inst =
      FindInstruction(hlo_module.get(), HloOpcode::kSubtract);
  EXPECT_NE(inst, nullptr);
  EXPECT_THAT(inst, op::Subtract(op::Multiply(), op::Add()));

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kCall;
    };
    auto extracted_module = ExtractModule(inst, /*height=*/-1, hlo_selector);
    EXPECT_EQ(extracted_module->computation_count(), 1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Subtract(op::Multiply(op::Broadcast(op::Parameter()),
                                          op::Parameter()),
                             op::Add(op::Parameter(), op::Parameter())));
  }

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kBroadcast;
    };
    auto extracted_module = ExtractModule(inst, /*height=*/2, hlo_selector);
    EXPECT_EQ(extracted_module->computation_count(), 1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Subtract(op::Multiply(op::Parameter(), op::Parameter()),
                             op::Add(op::Parameter(), op::Parameter())));
  }

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kBroadcast;
    };
    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      return ReplaceType::kReplaceConst;
    };
    auto extracted_module =
        ExtractModule(inst, /*height=*/2, hlo_selector, replace_type_selector);
    EXPECT_EQ(extracted_module->computation_count(), 1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Subtract(op::Multiply(op::Constant(), op::Parameter()),
                             op::Add(op::Parameter(), op::Parameter())));
  }

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kAdd;
    };
    auto extracted_module = ExtractModule(inst, /*height=*/-1, hlo_selector);
    // Here the extracted module should contain Computation: calculate_alpha
    EXPECT_EQ(extracted_module->computation_count(), 2);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Subtract(op::Multiply(), op::Parameter()));
  }

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kSubtract;
    };
    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      return ReplaceType::kReplaceConst;
    };
    auto extracted_module =
        ExtractModule(hlo_module->entry_computation()->root_instruction(),
                      /*height=*/2, hlo_selector, replace_type_selector);
    EXPECT_EQ(extracted_module->computation_count(), 1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Add(op::Constant(), op::Parameter()));
  }

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      if (hlo_inst->opcode() != HloOpcode::kBroadcast &&
          hlo_inst->opcode() != HloOpcode::kAdd) {
        return true;
      }
      return false;
    };
    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      if (hlo_inst->opcode() == HloOpcode::kBroadcast) {
        return ReplaceType::kReplaceConst;
      }
      return ReplaceType::kReplaceParam;
    };
    auto extracted_module =
        ExtractModule(inst, /*height=*/2, hlo_selector, replace_type_selector);
    EXPECT_EQ(extracted_module->computation_count(), 1);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Subtract(op::Multiply(op::Constant(), op::Parameter()),
                             op::Parameter()));
  }
}

TEST_F(HloExtractorTest, ReplaceTupleWithConstant) {
  const std::string& hlo_string = R"(
HloModule test

ENTRY %entry {
  param.0 = f32[4]{0} parameter(0)
  negate = f32[4]{0} negate(f32[4]{0} param.0)
  tuple = (f32[4]{0}, f32[4]{0}) tuple(param.0, negate)
  element = f32[4]{0} get-tuple-element((f32[4]{0}, f32[4]{0}) tuple), index=1
  ROOT exp = f32[4]{0} exponential(f32[4]{0} element)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));

  {
    auto hlo_selector = [](const HloInstruction* hlo_inst) -> bool {
      return hlo_inst->opcode() != HloOpcode::kTuple;
    };

    auto replace_type_selector =
        [](const HloInstruction* hlo_inst) -> ReplaceType {
      return ReplaceType::kReplaceConst;
    };
    auto extracted_module =
        ExtractModule(FindInstruction(hlo_module.get(), "exp"), /*height=*/-1,
                      hlo_selector, replace_type_selector);
    EXPECT_THAT(extracted_module->entry_computation()->root_instruction(),
                op::Exp(op::GetTupleElement(op::Constant())));
  }
}

}  // namespace
}  // namespace xla
