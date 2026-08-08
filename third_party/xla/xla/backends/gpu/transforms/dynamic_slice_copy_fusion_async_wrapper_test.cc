/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/dynamic_slice_copy_fusion_async_wrapper.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

using DynamicSliceCopyFusionAsyncWrapperTest = HloHardwareIndependentTestBase;

TEST_F(DynamicSliceCopyFusionAsyncWrapperTest, WrapsDynamicMemcpyFusion) {
  constexpr char kHlo[] = R"(
    dynamic_slice {
      p0 = s32[4] parameter(0)
      c1 = s32[] constant(1)

      ROOT slice = s32[1] dynamic-slice(p0, c1), dynamic_slice_sizes={1},
          backend_config={"dynamic_slice_config":
              {"byte_offset":"4","byte_stride":"0"}}
    }

    ENTRY main {
      p0 = s32[4] parameter(0)
      ROOT fusion = s32[1] fusion(p0), kind=kLoop, calls=dynamic_slice
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));

  DynamicSliceCopyFusionAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(true));

  EXPECT_THAT(RunFileCheck(module->ToString({}), R"(
    ; CHECK: %dynamic_slice
    ; CHECK: %dynamic_slice.clone
    ; CHECK:   ROOT {{.*}} = s32[1]{0} dynamic-slice(
    ; CHECK: ENTRY %main
    ; CHECK:   [[P0:%[^ ]+]] = s32[4]{0} parameter(0)
    ; CHECK:   [[START:%[^ ]+]] = ((s32[4]{0}), s32[1]{0}, u32[]) fusion-start([[P0]]), kind=kLoop, calls=%dynamic_slice.clone
    ; CHECK:   ROOT {{.*}} = s32[1]{0} fusion-done([[START]])
  )"),
              absl_testing::IsOkAndHolds(true));

  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));
}

TEST_F(DynamicSliceCopyFusionAsyncWrapperTest,
       WrapsDynamicUpdateSliceMemcpyFusion) {
  constexpr char kHlo[] = R"(
    dynamic_update_slice {
      p0 = s32[4] parameter(0)
      p1 = s32[1] parameter(1)
      c1 = s32[] constant(1)

      ROOT update = s32[4] dynamic-update-slice(p0, p1, c1),
          backend_config={"dynamic_slice_config":
              {"byte_offset":"4","byte_stride":"0"}}
    }

    ENTRY main {
      p0 = s32[4] parameter(0)
      p1 = s32[1] parameter(1)
      ROOT fusion = s32[4] fusion(p0, p1), kind=kLoop,
          calls=dynamic_update_slice
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));

  DynamicSliceCopyFusionAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(true));

  EXPECT_THAT(RunFileCheck(module->ToString({}), R"(
    ; CHECK: %dynamic_update_slice.clone
    ; CHECK:   ROOT {{.*}} = s32[4]{0} dynamic-update-slice(
    ; CHECK: ENTRY %main
    ; CHECK:   [[P0:%[^ ]+]] = s32[4]{0} parameter(0)
    ; CHECK:   [[P1:%[^ ]+]] = s32[1]{0} parameter(1)
    ; CHECK:   [[START:%[^ ]+]] = ((s32[4]{0}, s32[1]{0}), s32[4]{0}, u32[]) fusion-start([[P0]], [[P1]]), kind=kLoop, calls=%dynamic_update_slice.clone
    ; CHECK:   ROOT {{.*}} = s32[4]{0} fusion-done([[START]])
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(DynamicSliceCopyFusionAsyncWrapperTest, DoesNotWrapNonCopyHeroFusion) {
  constexpr char kHlo[] = R"(
    dsf_computation {
      p0 = s32[4] parameter(0)
      c1 = s32[] constant(1)
      ds = s32[1] dynamic-slice(p0, c1), dynamic_slice_sizes={1},
          backend_config={"dynamic_slice_config":
              {"byte_offset":"4","byte_stride":"0"}}
      ROOT add = s32[1] add(ds, ds)
    }

    ENTRY main {
      p0 = s32[4] parameter(0)
      ROOT fusion = s32[1] fusion(p0), kind=kCustom, calls=dsf_computation,
          backend_config={"fusion_backend_config":{"kind":"__custom_fusion",
              "custom_fusion_config":{"name":"dynamic_slice_fusion"}}}
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));

  DynamicSliceCopyFusionAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));
  EXPECT_THAT(RunFileCheck(module->ToString({}), R"(
    ; CHECK: ENTRY %main
    ; CHECK-NOT: fusion-start
    ; CHECK:   ROOT {{.*}} = s32[1]{0} fusion(
    ; CHECK: custom_fusion_config
    ; CHECK: dynamic_slice_fusion
  )"),
              absl_testing::IsOkAndHolds(true));
}

}  // namespace
}  // namespace xla::gpu
