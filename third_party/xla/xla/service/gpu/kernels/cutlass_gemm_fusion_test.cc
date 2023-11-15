/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/kernels/cutlass_gemm_fusion.h"

#include <utility>

#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/service/gpu/custom_fusion_rewriter.h"
#include "xla/service/gpu/kernels/custom_fusion_pattern.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla::gpu {

class CutlassFusionTest : public HloTestBase {
  // Custom fusions are not supported by XLA runtime.
  DebugOptions GetDebugOptionsForTest() override {
    auto debug_options = GetDebugOptionsFromFlags();
    debug_options.set_xla_gpu_enable_xla_runtime_executable(false);
    return debug_options;
  }
};

//===----------------------------------------------------------------------===//
// Pattern matching tests
//===----------------------------------------------------------------------===//

TEST_F(CutlassFusionTest, RowMajorGemm) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main (p0: f32[15,19], p1: f32[19,17]) -> f32[15,17] {
      %p0 = f32[15,19]{1,0} parameter(0)
      %p1 = f32[19,17]{1,0} parameter(1)
      ROOT %r = f32[15,17]{1,0} dot(%p0, %p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  const char* expected = R"(
    ; CHECK: %cutlass_gemm {{.*}} {
    ; CHECK:   [[P0:%[^ ]+]] = f32[15,19]{1,0} parameter(0)
    ; CHECK:   [[P1:%[^ ]+]] = f32[19,17]{1,0} parameter(1)
    ; CHECK:   ROOT [[DOT:%[^ ]+]] = f32[15,17]{1,0} dot([[P0]], [[P1]]),
    ; CEHCK:     lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ; CHECK: }

    ; CHECK: ENTRY %main {{.*}} {
    ; CHECK:   ROOT [[FUSION:%[^ ]+]] = f32[15,17]{1,0} fusion
    ; CHECK:     kind=kCustom, calls=%cutlass_gemm,
    ; CHECK:     backend_config={
    ; CHECK:       "kind":"__custom_fusion",
    ; CHECK:       "custom_fusion_config":{"name":"cutlass_gemm"}
    ; CHECK:     }
    ; CHECK: }
  )";

  CustomFusionPatternRegistry patterns;
  patterns.Emplace<CutlassGemmPattern>();

  CustomFusionRewriter pass(&patterns);
  RunAndFilecheckHloRewrite(hlo, std::move(pass), expected);
}

TEST_F(CutlassFusionTest, RowMajorGemmWithUpcast) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main (p0: bf16[15,19], p1: s8[19,17]) -> bf16[15,17] {
      %p0 = bf16[15,19]{1,0} parameter(0)
      %p1 = s8[19,17]{1,0} parameter(1)
      %c1 = bf16[19,17]{1,0} convert(%p1)
      ROOT %r = bf16[15,17]{1,0} dot(%p0, %c1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  const char* expected = R"(
    ; CHECK: %cutlass_gemm_with_upcast {{.*}} {
    ; CHECK-DAG: [[P0:%[^ ]+]] = bf16[15,19]{1,0} parameter
    ; CHECK-DAG: [[P1:%[^ ]+]] = s8[19,17]{1,0} parameter
    ; CHECK:     [[C1:%[^ ]+]] = bf16[19,17]{1,0} convert([[P1]])
    ; CHECK:     ROOT [[DOT:%[^ ]+]] = bf16[15,17]{1,0} dot([[P0]], [[C1]]),
    ; CEHCK:       lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ; CHECK: }

    ; CHECK: ENTRY %main {{.*}} {
    ; CHECK:   ROOT [[FUSION:%[^ ]+]] = bf16[15,17]{1,0} fusion
    ; CHECK:     kind=kCustom, calls=%cutlass_gemm_with_upcast,
    ; CHECK:     backend_config={
    ; CHECK:       "kind":"__custom_fusion",
    ; CHECK:       "custom_fusion_config":{"name":"cutlass_gemm_with_upcast"}
    ; CHECK:     }
    ; CHECK: }
  )";

  CustomFusionPatternRegistry patterns;
  patterns.Emplace<CutlassGemmWithUpcastPattern>();

  CustomFusionRewriter pass(&patterns);
  RunAndFilecheckHloRewrite(hlo, std::move(pass), expected);
}

//===----------------------------------------------------------------------===//
// Run And Compare Tests
//===----------------------------------------------------------------------===//

TEST_F(CutlassFusionTest, RowMajorGemmKernel) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_text_cublas = R"(
  HloModule cublas

  ENTRY e {
    arg0 = f32[100,784]{1,0} parameter(0)
    arg1 = f32[784,10]{1,0} parameter(1)
    gemm = (f32[100,10]{1,0}, s8[0]{0}) custom-call(arg0, arg1),
      custom_call_target="__cublas$gemm",
      backend_config={"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[1],"rhs_contracting_dimensions":[0],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}
    ROOT get-tuple-element = f32[100,10]{1,0} get-tuple-element((f32[100,10]{1,0}, s8[0]{0}) gemm), index=0
  })";

  const char* hlo_text_custom_fusion = R"(
  HloModule cutlass

  cutlass_gemm {
    arg0 = f32[100,784]{1,0} parameter(0)
    arg1 = f32[784,10]{1,0} parameter(1)
    ROOT dot = f32[100,10]{1,0} dot(arg0, arg1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY e {
    arg0 = f32[100,784]{1,0} parameter(0)
    arg1 = f32[784,10]{1,0} parameter(1)
    ROOT _ = f32[100,10]{1,0} fusion(arg0, arg1), kind=kCustom, calls=cutlass_gemm,
      backend_config={kind: "__custom_fusion", custom_fusion_config: {"name":"cutlass_gemm"}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_cublas, hlo_text_custom_fusion,
                                      error_spec, /*run_hlo_passes=*/false));
}

TEST_F(CutlassFusionTest, RowMajorGemmWithUpcastKernel) {
  GTEST_SKIP() << "Requires CUTLASS 3.3.0+";

  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_text_cublas = R"(
  HloModule cublas

  ENTRY e {
    p0 = bf16[16,32]{1,0} parameter(0)
    p1 = s8[32,8]{1,0} parameter(1)
    c1 = bf16[32,8]{1,0} convert(p1)
    gemm = (bf16[16,8]{1,0}, s8[0]{0}) custom-call(p0, c1),
      custom_call_target="__cublas$gemm",
      backend_config={"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[1],"rhs_contracting_dimensions":[0],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}
    ROOT get-tuple-element = bf16[16,8]{1,0} get-tuple-element((bf16[16,8]{1,0}, s8[0]{0}) gemm), index=0
  })";

  const char* hlo_text_custom_fusion = R"(
  HloModule cutlass

  cutlass_gemm_with_upcast {
    p0 = bf16[16,32]{1,0} parameter(0)
    p1 = s8[32,8]{1,0} parameter(1)
    c1 = bf16[32,8]{1,0} convert(p1)
    ROOT dot = bf16[16,8]{1,0} dot(p0, c1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY e {
    p0 = bf16[16,32]{1,0} parameter(0)
    p1 = s8[32,8]{1,0} parameter(1)
    ROOT _ = bf16[16,8]{1,0} fusion(p0, p1), kind=kCustom, calls=cutlass_gemm_with_upcast,
      backend_config={kind: "__custom_fusion", custom_fusion_config: {"name":"cutlass_gemm_with_upcast"}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_cublas, hlo_text_custom_fusion,
                                      error_spec, /*run_hlo_passes=*/false));
}

}  // namespace xla::gpu
