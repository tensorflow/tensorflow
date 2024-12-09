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

#include "absl/strings/string_view.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class SimplifyFPConversionsTest : public HloTestBase {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_allow_excess_precision(
        enable_simplify_all_fp_conversions_);
    return debug_options;
  }

  bool SupportsMultiplyBF16() {
    const auto& device_description =
        backend().default_stream_executor()->GetDeviceDescription();
    const auto& cc = device_description.gpu_compute_capability();
    return std::holds_alternative<se::CudaComputeCapability>(cc) &&
           std::get<se::CudaComputeCapability>(cc).IsAtLeastHopper();
  }

  void SetEnableSimplifyFpConversions(bool enable_simplify_all_fp_conversions) {
    enable_simplify_all_fp_conversions_ = enable_simplify_all_fp_conversions;
  }

  static constexpr absl::string_view kHloText = R"(
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

  if (SupportsMultiplyBF16()) {
    // If the GPU supports multiplication of bf16 values, only rsqrt is wrapped
    // in the convert operations.
    MatchOptimizedHlo(kHloText, R"(
// CHECK: %[[P0:.*]] = bf16{{.*}} parameter({{.*}})
// CHECK: %[[C0:.*]] = f32{{.*}} convert(%[[P0]])
// CHECK: %[[RSQRT:.*]] = f32{{.*}} rsqrt(%[[C0]])
// CHECK: %[[RCONV:.*]] = bf16{{.*}} convert(%[[RSQRT]])
// CHECK: %[[BCAST:.*]] = bf16{{.*}} broadcast(%[[RCONV]])
// CHECK: %[[P1:.*]] = bf16{{.*}} parameter({{.*}})
// CHECK: ROOT {{.*}} = bf16{{.*}} multiply(%[[BCAST]], %[[P1]])
)");
  } else {
    // If the GPU only supports multiplication of f32 values, make sure that
    // there's no conversion between rsqrt and broadcast operations.
    MatchOptimizedHlo(kHloText, R"(
// CHECK: %[[P0:.*]] = bf16{{.*}} parameter({{.*}})
// CHECK: %[[C0:.*]] = f32{{.*}} convert(%[[P0]])
// CHECK: %[[RSQRT:.*]] = f32{{.*}} rsqrt(%[[C0]])
// CHECK: %[[BCAST:.*]] = f32{{.*}} broadcast(%[[RSQRT]])
// CHECK-DAG: %[[P1:.*]] = bf16{{.*}} parameter({{.*}})
// CHECK-DAG: %[[C1:.*]] = f32{{.*}} convert(%[[P1]])
// CHECK: %[[MUL:.*]] = f32{{.*}} multiply(%[[BCAST]], %[[C1]])
// CHECK: ROOT {{.*}} = bf16{{.*}} convert(%[[MUL]])
)");
  }
}

TEST_F(SimplifyFPConversionsTest, RedundantTypeConversionsArePresentInTest) {
  if (SupportsMultiplyBF16()) {
    GTEST_SKIP() << "No double convert is expected on Hopper";
  }

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
