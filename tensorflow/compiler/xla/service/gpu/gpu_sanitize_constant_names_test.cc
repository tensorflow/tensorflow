/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_sanitize_constant_names.h"

#include <utility>

#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;
using SanitizeConstantNamesTest = HloTestBase;

TEST_F(SanitizeConstantNamesTest, InstructionNameWithHyphenSanitized) {
  const char *const kHloString = R"(
    HloModule HyphenInInstructionName
      ENTRY kernelEntry {
        ROOT equal-to = s32[2]{0} constant({42, 73})
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_TRUE(GpuSanitizeConstantNames().Run(module.get()).value());
  HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->name(), "equal_to");
}

TEST_F(SanitizeConstantNamesTest, InstructionNameWithDotSanitized) {
  const char *const kHloString = R"(
    HloModule HyphenInInstructionName
      ENTRY kernelEntry {
        ROOT equal.to = s32[2]{0} constant({42, 73})
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_TRUE(GpuSanitizeConstantNames().Run(module.get()).value());
  HloInstruction *root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->name(), "equal_to");
}

TEST_F(SanitizeConstantNamesTest, BufferSanitizedNameCollisionResolved) {
  const char *const kHloString = R"(
    HloModule BufferSanitizedName
      ENTRY kernelEntry {
      equal.to = s32[2]{0} constant({42, 73})
      equal-to = s32[2]{0} constant({67, 3})
      ROOT equal_to = s32[2]{0} add(equal.to, equal-to)
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  EXPECT_TRUE(GpuSanitizeConstantNames().Run(module.get()).value());
  EXPECT_THAT(FindInstruction(module.get(), "equal_to_1"), op::Constant());
  EXPECT_THAT(FindInstruction(module.get(), "equal_to_2"), op::Constant());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
