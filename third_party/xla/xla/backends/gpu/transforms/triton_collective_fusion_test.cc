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

#include "xla/backends/gpu/transforms/triton_collective_fusion.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

class TritonCollectiveFusionTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription device_info_ = TestGpuDeviceInfo::H100SXMDeviceInfo();
};

TEST_F(TritonCollectiveFusionTest, FuseAllReduceWithGemm) {
  const std::string hlo_text = R"(
HloModule TritonCollectiveFusionTest

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

fcomp {
  p0 = f32[1024,1024]{1,0} parameter(0)
  p1 = f32[1024,1024]{1,0} parameter(1)
  ROOT gemm = f32[1024,1024]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  p0 = f32[1024,1024]{1,0} parameter(0)
  p1 = f32[1024,1024]{1,0} parameter(1)
  gemm_fusion = f32[1024,1024]{1,0} fusion(p0, p1), kind=kCustom, calls=fcomp,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  ar-start = f32[1024,1024]{1,0} all-reduce-start(gemm_fusion), to_apply=add, replica_groups={{0,1}}
  ROOT ar-done = f32[1024,1024]{1,0} all-reduce-done(ar-start)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_unsupported_use_all_reduce_one_shot_kernel(true);

  TritonCollectiveFusion pass(device_info_);
  ASSERT_THAT(pass.Run(module.get()), IsOkAndHolds(true));

  EXPECT_THAT(RunFileCheck(module->ToString(), R"(
// CHECK: ENTRY %main
// CHECK: %[[P0:.*]] = f32[1024,1024]{1,0} parameter(0)
// CHECK: %[[P1:.*]] = f32[1024,1024]{1,0} parameter(1)
// CHECK: %[[FUSION:.*]] = f32[1024,1024]{1,0} fusion(%[[P0]], %[[P1]]), kind=kCustom, calls=%fcomp, backend_config={{{.*}}"kind":"__triton_collective"
// CHECK-NOT: all-reduce-start
// CHECK: ROOT %[[AR_DONE:.*]] = f32[1024,1024]{1,0} all-reduce-done(%[[FUSION]])
)"),
              IsOkAndHolds(true));
}

TEST_F(TritonCollectiveFusionTest, DoNotFuseAllReduceWithGemmOnAmpere) {
  const std::string hlo_text = R"(
HloModule TritonCollectiveFusionTest

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

fcomp {
  p0 = f32[1024,1024]{1,0} parameter(0)
  p1 = f32[1024,1024]{1,0} parameter(1)
  ROOT gemm = f32[1024,1024]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  p0 = f32[1024,1024]{1,0} parameter(0)
  p1 = f32[1024,1024]{1,0} parameter(1)
  gemm_fusion = f32[1024,1024]{1,0} fusion(p0, p1), kind=kCustom, calls=fcomp,
    backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  ar-start = f32[1024,1024]{1,0} all-reduce-start(gemm_fusion), to_apply=add, replica_groups={{0,1}}
  ROOT ar-done = f32[1024,1024]{1,0} all-reduce-done(ar-start)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_unsupported_use_all_reduce_one_shot_kernel(true);

  TritonCollectiveFusion pass(TestGpuDeviceInfo::RTXA6000DeviceInfo());
  ASSERT_THAT(pass.Run(module.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
