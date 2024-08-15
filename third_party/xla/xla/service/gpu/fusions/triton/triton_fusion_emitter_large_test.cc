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

#include <string>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/error_spec.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class TritonGemmTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_cublas_fallback(false);
    return debug_options;
  }
};

TEST_F(TritonGemmTest, IndexUsing64Bits) {
  const char* kHloTextRef = R"(
HloModule r

ENTRY e {
  arg0 = f16[65536,32800] parameter(0)
  arg1 = f16[32800,32] parameter(1)
  gemm = (f16[65536,32], s8[0]) custom-call(arg0, arg1),
    custom_call_target="__cublas$gemm",
    backend_config="{\"gemm_backend_config\": {\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}}"
  ROOT get-tuple-element = f16[65536,32] get-tuple-element((f16[65536,32], s8[0]) gemm), index=0
}
)";

  const char* kHloTextTest = R"(
HloModule t

triton_dot {
  p0 = f16[65536,32800] parameter(0)
  p1 = f16[32800,32] parameter(1)
  ROOT dot = f16[65536,32] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f16[65536,32800] parameter(0)
  p1 = f16[32800,32] parameter(1)
  ROOT _ = f16[65536,32] fusion(p0, p1), kind=kCustom, calls=triton_dot,
    backend_config="{\"fusion_backend_config\": {kind: \"__triton_gemm\", triton_gemm_config: {\"block_m\":\"32\",\"block_n\":\"32\",\"block_k\":\"32\",\"split_k\":\"1\",\"num_stages\":\"1\",\"num_warps\":\"1\",\"num_ctas\":\"1\"}}}"
}
)";

  EXPECT_TRUE(RunAndCompareTwoModules(kHloTextRef, kHloTextTest,
                                      ErrorSpec{1e-3, 1e-3},
                                      /*run_hlo_passes=*/false));
}

TEST_F(TritonGemmTest, LargeNonContractingProductWorks) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
  p0 = s8[1310720,2] parameter(0)
  c0 = f16[1310720,2] convert(p0)
  p1 = f16[2,15] parameter(1)
  ROOT dot.12 = f16[1310720,15] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  // Make sure the output size is sufficient to use the X grid dimension
  // for the non-contracting dimensions of the output. 16x16 is the smallest
  // MxN block used currently.
  CHECK_GT(1310720 * 15 / (16 * 16), 65535);

  MatchOptimizedHlo(kHloText, R"(
; CHECK: triton
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonGemmTest, LargeBatchWorks) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
  Arg_0.8 = pred[102400,10,10] parameter(0)
  convert.11 = f32[102400,10,10] convert(Arg_0.8)
  Arg_1.9 = f32[102400,10,100] parameter(1)
  ROOT dot.12 = f32[102400,10,100] dot(convert.11, Arg_1.9),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK: triton
)");

  // Batch size of 102400 is over 65535 so the X grid dimension has to be used
  // for it.

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

class TritonSoftmaxTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_triton_softmax_fusion(true);
    return debug_options;
  }
};

TEST_F(TritonSoftmaxTest,
       CanFuseAndEmitDiamondWithInputNumberOfElementsLargerThanInt32Max) {
  const std::string hlo_text = R"(
HloModule softmax

max_computation {
  arg_0 = f16[] parameter(0)
  arg_1 = f16[] parameter(1)
  ROOT maximum = f16[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = f16[65538,32768]{1,0} parameter(0)
  constant_neg_inf = f16[] constant(-inf)
  reduce = f16[65538]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f16[65538,32768]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f16[65538,32768]{1,0} subtract(param_0, broadcast)
}
)";

  MatchOptimizedHlo(hlo_text, R"(
; CHECK:    ENTRY
; CHECK:      %[[P0:.*]] = f16[65538,32768]{1,0} parameter(0)
; CHECK:      ROOT
; CHECK-SAME: fusion(%[[P0]])
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton
)");

  // Checking that this does not crash should be enough.
  EXPECT_TRUE(Run(hlo_text));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
