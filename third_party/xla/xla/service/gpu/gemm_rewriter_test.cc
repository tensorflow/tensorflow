/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/gemm_rewriter.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class GemmRewriterTest : public HloTestBase {
 public:
  GemmRewriterTest()
      : HloTestBase(/*verifier_layout_sensitive=*/true,
                    /*allow_mixed_precision_in_hlo_verifier=*/false) {}

  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_cublaslt(false);
    return debug_options;
  }

  se::GpuComputeCapability gpu_version_{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
};

TEST_F(GemmRewriterTest, MatrixVectorMultiplication) {
  const char* hlo = R"(
HloModule m

ENTRY e {
  p0 = f32[2048] parameter(0)
  p1 = f32[2048, 16384] parameter(1)
  ROOT d = f32[16384] dot(p0, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})";

  const char* expected = R"(
// CHECK:  %[[P0:.+]] = f32[2048]{0} parameter(0)
// CHECK:  %[[P1:.+]] = f32[2048,16384]{1,0} parameter(1)
// CHECK:  %[[CUSTOM_CALL:.+]] = (f32[16384]{0}, s8[4194304]{0}) custom-call(%[[P0]], %[[P1]]), custom_call_target="__cublas$gemm"
)";

  RunAndFilecheckHloRewrite(hlo, GemmRewriter(gpu_version_), expected);
}

TEST_F(GemmRewriterTest, MatrixVectorMultiplicationWithBatch) {
  const char* hlo = R"(
HloModule m

ENTRY e {
  p0 = f32[10, 10, 2048] parameter(0)
  p1 = f32[10, 10, 2048, 16384] parameter(1)
  ROOT d = f32[10, 10, 16384] dot(p0, p1),
   lhs_batch_dims={0, 1}, rhs_batch_dims={0, 1},
   lhs_contracting_dims={2}, rhs_contracting_dims={2}
})";

  const char* expected = R"(
// CHECK:  %[[P0:.+]] = f32[10,10,2048]{2,1,0} parameter(0)
// CHECK:  %[[P1:.+]] = f32[10,10,2048,16384]{3,2,1,0} parameter(1)
// CHECK:  %[[CUSTOM_CALL:.+]] = (f32[10,10,16384]{2,1,0}, s8[4194304]{0}) custom-call(%[[P0]], %[[P1]]), custom_call_target="__cublas$gemm"
)";

  RunAndFilecheckHloRewrite(hlo, GemmRewriter(gpu_version_), expected);
}

}  // namespace
}  // namespace xla::gpu
