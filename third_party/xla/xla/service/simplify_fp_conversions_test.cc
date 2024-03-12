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

  SimplifyFPConversions simplifier;
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

  SimplifyFPConversions simplifier;
  EXPECT_THAT(simplifier.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Tuple(op::Parameter(0)));
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

  SimplifyFPConversions simplifier;
  EXPECT_THAT(simplifier.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(AllOf(op::Shape("bf16[2,3]"), op::Convert(op::Parameter(0)))));
}

}  // namespace
}  // namespace xla
