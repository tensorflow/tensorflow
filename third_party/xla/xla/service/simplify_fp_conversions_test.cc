/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/simplify_fp_conversions.h"

#include <memory>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using ::testing::AllOf;
using ::tsl::testing::IsOkAndHolds;

using SimplifyFPConversionsTest = HloTestBase;

// This marks all ops in `module` as user-provided, meaning the
// simplifier won't remove any of the converts
static void InitializeCreationPassIds(HloModule* module) {
  constexpr int kUserSuppliedOpCreationPassId = -1;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      instruction->set_creation_pass_id(kUserSuppliedOpCreationPassId);
      instruction->set_logical_creation_pass_id(kUserSuppliedOpCreationPassId);
    }
  }
}

// This marks all converts ops in `module` as being created by the
// optimization pass `creation_pass_id`.
static void SetCreationPassIdInAllConvertOps(HloModule* module,
                                             int creation_pass_id) {
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kConvert) {
        instruction->set_creation_pass_id(creation_pass_id);
        instruction->set_logical_creation_pass_id(creation_pass_id);
      }
    }
  }
}

TEST_F(SimplifyFPConversionsTest, DoesNotChangeSingleConvert) {
  const absl::string_view kModuleStr = R"(
    HloModule test

    ENTRY entry {
      p0 = f32[2,3] parameter(0)
      c0 = bf16[2,3] convert(p0)
      ROOT ret = (bf16[2,3]) tuple(c0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  InitializeCreationPassIds(module.get());

  SimplifyFPConversions simplifier{
      SimplifyFPConversions::Scope::kSimplifyAllConversions};
  EXPECT_THAT(simplifier.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(SimplifyFPConversionsTest, SimplifiesF32ToBF16ToF32) {
  const absl::string_view kModuleStr = R"(
    HloModule test

    ENTRY entry {
      p0 = f32[2,3] parameter(0)
      c0 = bf16[2,3] convert(p0)
      c1 = f32[2,3] convert(c0)
      ROOT ret = (f32[2,3]) tuple(c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  InitializeCreationPassIds(module.get());

  SimplifyFPConversions simplifier{
      SimplifyFPConversions::Scope::kSimplifyAllConversions};
  EXPECT_THAT(simplifier.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Parameter(0)));
}

TEST_F(SimplifyFPConversionsTest, SimplifiesCompilerGeneratedF32ToBF16ToF32) {
  const absl::string_view kModuleStr = R"(
    HloModule test

    ENTRY entry {
      p0 = f32[2,3] parameter(0)
      c0 = bf16[2,3] convert(p0)
      c1 = f32[2,3] convert(c0)
      ROOT ret = (f32[2,3]) tuple(c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  InitializeCreationPassIds(module.get());

  constexpr int kRandomCreationPassId = 42;
  SetCreationPassIdInAllConvertOps(module.get(), kRandomCreationPassId);

  SimplifyFPConversions simplifier{
      SimplifyFPConversions::Scope::kOnlySimplifyCompilerGeneratedConversions};
  EXPECT_THAT(simplifier.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Parameter(0)));
}

TEST_F(SimplifyFPConversionsTest, DoesNotChangeUserInsertedConverts) {
  const absl::string_view kModuleStr = R"(
    HloModule test

    ENTRY entry {
      p0 = f32[2,3] parameter(0)
      c0 = bf16[2,3] convert(p0)
      c1 = f32[2,3] convert(c0)
      ROOT ret = (f32[2,3]) tuple(c1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  InitializeCreationPassIds(module.get());

  SimplifyFPConversions simplifier{
      SimplifyFPConversions::Scope::kOnlySimplifyCompilerGeneratedConversions};
  EXPECT_THAT(simplifier.Run(module.get()), IsOkAndHolds(false));
}

TEST_F(SimplifyFPConversionsTest, SimplifiesF64ToF16ToF32ToBF16) {
  const absl::string_view kModuleStr = R"(
    HloModule test

    ENTRY entry {
      p0 = f64[2,3] parameter(0)
      c0 = f16[2,3] convert(p0)
      c1 = f32[2,3] convert(c0)
      c2 = bf16[2,3] convert(c1)
      ROOT ret = (bf16[2,3]) tuple(c2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  InitializeCreationPassIds(module.get());

  SimplifyFPConversions simplifier{
      SimplifyFPConversions::Scope::kSimplifyAllConversions};
  EXPECT_THAT(simplifier.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(AllOf(op::Shape("bf16[2,3]"), op::Convert(op::Parameter(0)))));
}

}  // namespace
}  // namespace xla
