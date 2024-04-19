/* Copyright 2023 The OpenXLA Authors.

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

#include <string_view>

#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class SimplifyFPConversionsTest : public HloTestBase {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_allow_excess_precision(
        enable_simplify_all_fp_conversions_);
    return debug_options;
  }

  void SetEnableSimplifyFpConversions(bool enable_simplify_all_fp_conversions) {
    enable_simplify_all_fp_conversions_ = enable_simplify_all_fp_conversions;
  }

  static constexpr std::string_view kHloText = R"(
HloModule module

ENTRY main {
  param0 = bf16[1536]{0} parameter(0)
  param1 = bf16[4,1536]{1,0} parameter(1)

  s = bf16[1536]{0} rsqrt(param0)
  // Redundant conversions appear here when the algebraic simplifier
  // pushes the broadcast op further down
  b = bf16[4,1536]{1,0} broadcast(s), dimensions={1}

  ROOT d = bf16[4,1536]{1,0} multiply(b, param1)
}
  )";

 private:
  bool enable_simplify_all_fp_conversions_ = false;
};

TEST_F(SimplifyFPConversionsTest, RedundantTypeConversionsGetCleanedUp) {
  // The algebraic simplifier might expose redundant type conversions,
  // i.e. f32 -> bf16 -> f32. This test ensures that they will get cleaned up
  // eventually by the SimplifyFPConversion pass.

  SetEnableSimplifyFpConversions(true);

  // This matcher ensures that there will be no convert in between the rsqrt and
  // the broadcast instruction.
  MatchOptimizedHlo(kHloText, R"(
// CHECK: rsqrt(
// CHECK-NOT: convert(
// CHECK: broadcast(
)");
}

TEST_F(SimplifyFPConversionsTest, RedundantTypeConversionsArePresentInTest) {
  // This test ensures that the HLO that we use in the previous test is actually
  // meaningful and would lead to redundant type conversions if the simplifier
  // didn't clean them up.

  SetEnableSimplifyFpConversions(false);

  MatchOptimizedHlo(kHloText, R"(
// CHECK: rsqrt(
// CHECK-NEXT: convert(
// CHECK-NEXT: convert(
// CHECK-NEXT: broadcast(
)");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
