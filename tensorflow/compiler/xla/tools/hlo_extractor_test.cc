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

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = testing::opcode_matchers;

using HloExtractorTest = HloTestBase;

TEST_F(HloExtractorTest, ExtractTopLevel) {
  const string& hlo_string = R"(
HloModule test

ENTRY %entry {
  param.0 = f32[4]{0} parameter(0)
  negate = f32[4]{0} negate(f32[4]{0} param.0)
  ROOT exp = f32[4]{0} exponential(f32[4]{0} negate)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      HloRunner::CreateModuleFromString(hlo_string, GetDebugOptionsForTest()));

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
  const string& hlo_string = R"(
HloModule test

ENTRY %entry {
  param.0 = f32[4]{0} parameter(0)
  tanh = f32[4]{0} tanh(f32[4]{0} param.0)
  negate = f32[4]{0} negate(f32[4]{0} tanh)
  exp = f32[4]{0} exponential(f32[4]{0} negate)
  ROOT add = f32[4]{0} add(f32[4]{0} negate, f32[4]{0} exp)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      HloRunner::CreateModuleFromString(hlo_string, GetDebugOptionsForTest()));

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
  const string& hlo_string = R"(
HloModule test

ENTRY %entry {
  p = f32[4]{0} parameter(0)
  tanh = f32[4]{0} tanh(p)
  c = f32[4]{0} constant({1, 2, 3, 4})
  ROOT add = f32[4]{0} add(tanh, c)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> hlo_module,
      HloRunner::CreateModuleFromString(hlo_string, GetDebugOptionsForTest()));

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

}  // namespace
}  // namespace xla
