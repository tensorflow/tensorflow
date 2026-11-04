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

#include "xla/hlo/transforms/simplifiers/simplify_fp_conversions.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using ::testing::AllOf;

using SimplifyFPConversionsTest = HloHardwareIndependentTestBase;

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
  EXPECT_THAT(simplifier.Run(module.get()), absl_testing::IsOkAndHolds(false));
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
  EXPECT_THAT(simplifier.Run(module.get()), absl_testing::IsOkAndHolds(true));
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
  EXPECT_THAT(simplifier.Run(module.get()), absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Tuple(AllOf(op::Shape("bf16[2,3]"), op::Convert(op::Parameter(0)))));
}

TEST_F(SimplifyFPConversionsTest, PreservesF8E4M3FNQuantizationRoundTrips) {
  for (absl::string_view chain :
       {"f32[2,3] parameter(0)", "bf16[2,3] parameter(0)"}) {
    for (absl::string_view after_f8 : {"", "c0b = bf16[2,3] convert(c0)\n"}) {
      const std::string module_str = absl::StrFormat(
          R"(
    HloModule test
    ENTRY entry {
      p0 = %s
      c0 = f8e4m3fn[2,3] convert(p0)
      %sc1 = f32[2,3] convert(%s)
      ROOT ret = (f32[2,3]) tuple(c1)
    })",
          chain, after_f8, after_f8.empty() ? "c0" : "c0b");
      ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                           ParseAndReturnVerifiedModule(module_str));
      SimplifyFPConversions simplifier;
      EXPECT_THAT(simplifier.Run(module.get()),
                  absl_testing::IsOkAndHolds(false))
          << module_str;
    }
  }
}

}  // namespace
}  // namespace xla
