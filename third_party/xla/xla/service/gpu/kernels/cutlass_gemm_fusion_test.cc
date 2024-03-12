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

#include "xla/service/gpu/kernels/cutlass_gemm_fusion.h"

#include <cstdint>
#include <utility>

#include "xla/array.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/error_spec.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/custom_kernel_fusion_rewriter.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion_pattern.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/types.h"
#include "tsl/platform/test.h"

namespace xla::gpu {

class CutlassFusionTest : public HloTestBase {};

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
    ; CHECK:     lhs_contracting_dims={1}, rhs_contracting_dims={0}
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

  CustomKernelFusionPatternRegistry patterns;
  patterns.Emplace<CutlassGemmPattern>();

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  CustomKernelFusionRewriter pass(&device, &patterns);
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
    ; CHECK:       lhs_contracting_dims={1}, rhs_contracting_dims={0}
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

  CustomKernelFusionPatternRegistry patterns;
  patterns.Emplace<CutlassGemmWithUpcastPattern>();

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  CustomKernelFusionRewriter pass(&device, &patterns);
  RunAndFilecheckHloRewrite(hlo, std::move(pass), expected);
}

TEST_F(CutlassFusionTest, RowMajorGemmWithDynamicUpdateSlice) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main (p0: f32[2,2,2], p1: f32[2,2], i: s32[]) -> f32[2,2,2] {
      %p0 = f32[2,2,2]{2,1,0} parameter(0)
      %p1 = f32[2,2]{1,0} parameter(1)
      %i = s32[] parameter(2)

      %dot = f32[2,2]{1,0} dot(%p1, %p1),
               lhs_contracting_dims={1},
               rhs_contracting_dims={0}
      %bc = f32[1,2,2]{2,1,0} bitcast(%dot)

      ROOT %r = f32[2,2,2]{2,1,0} dynamic-update-slice(%p0, %bc, %i, %i, %i)
    }
  )";

  const char* expected = R"(
    ; CHECK: %cutlass_gemm_with_dynamic_update_slice {{.*}} {
    ; CHECK-DAG: [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter
    ; CHECK-DAG: [[P1:%[^ ]+]] = f32[2,2,2]{2,1,0} parameter
    ; CHECK-DAG: [[P2:%[^ ]+]] = s32[] parameter
    ; CHECK-DAG: [[DOT:%[^ ]+]] = f32[2,2]{1,0} dot([[P0]], [[P0]])
    ; CHECK-DAG: [[CAST:%[^ ]+]] = f32[1,2,2]{2,1,0} bitcast([[DOT]])
    ; CHECK:     ROOT [[DUS:%[^ ]+]] = f32[2,2,2]{2,1,0} dynamic-update-slice(
    ; CHECK:       [[P1]], [[CAST]], [[P2]], [[P2]], [[P2]]
    ; CHECK:     )
    ; CHECK: }

    ; CHECK: ENTRY %main {{.*}} {
    ; CHECK:   ROOT [[FUSION:%[^ ]+]] = f32[2,2,2]{2,1,0} fusion
    ; CHECK:     kind=kCustom, calls=%cutlass_gemm_with_dynamic_update_slice,
    ; CHECK:     backend_config={
    ; CHECK:       "kind":"__custom_fusion",
    ; CHECK:       "custom_fusion_config":{
    ; CHECK:         "name":"cutlass_gemm_with_dynamic_update_slice"
    ; CHECK:       }
    ; CHECK:     }
    ; CHECK: }
  )";

  CustomKernelFusionPatternRegistry patterns;
  patterns.Emplace<CutlassGemmWithDynamicUpdateSlicePattern>();

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  CustomKernelFusionRewriter pass(&device, &patterns);
  RunAndFilecheckHloRewrite(hlo, std::move(pass), expected);
}

TEST_F(CutlassFusionTest, RowMajorGemmWithDynamicUpdateSliceMultipleUses) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main {
      %p0 = f32[2,2,2]{2,1,0} parameter(0)
      %p1 = f32[2,2]{1,0} parameter(1)
      %i = s32[] parameter(2)

      %dot = f32[2,2]{1,0} dot(%p1, %p1),
               lhs_contracting_dims={1},
               rhs_contracting_dims={0}
      %add = f32[2,2]{1,0} add(%dot, %dot)

      %cast = f32[1,2,2]{2,1,0} bitcast(%dot)
      %dus = f32[2,2,2]{2,1,0} dynamic-update-slice(%p0, %cast, %i, %i, %i)

      ROOT %r = (f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(%add, %dus)
    }
  )";

  const char* expected = R"(
    ; CHECK: %cutlass_gemm_with_dynamic_update_slice {{.*}} {
    ; CHECK-DAG: [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter
    ; CHECK-DAG: [[P1:%[^ ]+]] = f32[2,2,2]{2,1,0} parameter
    ; CHECK-DAG: [[P2:%[^ ]+]] = s32[] parameter
    ; CHECK-DAG: [[DOT:%[^ ]+]] = f32[2,2]{1,0} dot([[P0]], [[P0]])
    ; CHECK-DAG: [[CAST:%[^ ]+]] = f32[1,2,2]{2,1,0} bitcast([[DOT]])
    ; CHECK:     ROOT [[DUS:%[^ ]+]] = f32[2,2,2]{2,1,0} dynamic-update-slice(
    ; CHECK:       [[P1]], [[CAST]], [[P2]], [[P2]], [[P2]]
    ; CHECK:     )
    ; CHECK: }

    ; CHECK: ENTRY %main {{.*}} {
    ; CHECK:   [[OFFSET:%[^ ]+]] = s32[] parameter(2)
    ; CHECK:   [[FUSION:%[^ ]+]] = f32[2,2,2]{2,1,0} fusion
    ; CHECK:     kind=kCustom, calls=%cutlass_gemm_with_dynamic_update_slice,
    ; CHECK:     backend_config={
    ; CHECK:       "kind":"__custom_fusion",
    ; CHECK:       "custom_fusion_config":{
    ; CHECK:         "name":"cutlass_gemm_with_dynamic_update_slice"
    ; CHECK:       }
    ; CHECK:     }
    ; CHECK:   [[SLICE:%[^ ]+]] = f32[1,2,2]{2,1,0} dynamic-slice(
    ; CHECK:     [[FUSION]], [[OFFSET]], [[OFFSET]], [[OFFSET]]),
    ; CHECK:     dynamic_slice_sizes={1,2,2}
    ; CHECK:   [[CAST:%[^. ]+]] = f32[2,2]{1,0} bitcast([[SLICE]])
    ; CHECK:   [[ADD:%[^. ]+]] = f32[2,2]{1,0} add([[CAST]], [[CAST]])
    ; CHECK: }
  )";

  CustomKernelFusionPatternRegistry patterns;
  patterns.Emplace<CutlassGemmWithDynamicUpdateSlicePattern>();

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  CustomKernelFusionRewriter pass(&device, &patterns);
  RunAndFilecheckHloRewrite(hlo, std::move(pass), expected);
}

TEST_F(CutlassFusionTest, RowMajorGemmWithDynamicUpdateSliceWithoutBitcast) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main (p0: f32[4,2], p1: f32[2,2], i: s32[]) -> f32[4,2] {
      %p0 = f32[4,2]{1,0} parameter(0)
      %p1 = f32[2,2]{1,0} parameter(1)
      %i = s32[] parameter(2)

      %dot = f32[2,2]{1,0} dot(%p1, %p1),
               lhs_contracting_dims={1},
               rhs_contracting_dims={0}

      ROOT %r = f32[4,2]{1,0} dynamic-update-slice(%p0, %dot, %i, %i)
    }
  )";

  const char* expected = R"(
    ; CHECK: %cutlass_gemm_with_dynamic_update_slice {{.*}} {
    ; CHECK-DAG: [[P1:%[^ ]+]] = f32[4,2]{1,0} parameter
    ; CHECK-DAG: [[P0:%[^ ]+]] = f32[2,2]{1,0} parameter
    ; CHECK-DAG: [[DOT:%[^ ]+]] = f32[2,2]{1,0} dot([[P0]], [[P0]])
    ; CHECK-DAG: [[P2:%[^ ]+]] = s32[] parameter
    ; CHECK:     ROOT [[DUS:%[^ ]+]] = f32[4,2]{1,0} dynamic-update-slice([[P1]], [[DOT]], [[P2]], [[P2]])
    ; CHECK: }

    ; CHECK: ENTRY %main {{.*}} {
    ; CHECK:   ROOT [[FUSION:%[^ ]+]] = f32[4,2]{1,0} fusion
    ; CHECK:     kind=kCustom, calls=%cutlass_gemm_with_dynamic_update_slice,
    ; CHECK:     backend_config={
    ; CHECK:       "kind":"__custom_fusion",
    ; CHECK:       "custom_fusion_config":{
    ; CHECK:         "name":"cutlass_gemm_with_dynamic_update_slice"
    ; CHECK:       }
    ; CHECK:     }
    ; CHECK: }
  )";

  CustomKernelFusionPatternRegistry patterns;
  patterns.Emplace<CutlassGemmWithDynamicUpdateSlicePattern>();

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  CustomKernelFusionRewriter pass(&device, &patterns);
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
      backend_config={"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[1],"rhs_contracting_dimensions":[0],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
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
      backend_config={"fusion_backend_config":{kind: "__custom_fusion", custom_fusion_config: {"name":"cutlass_gemm"}}}
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
      backend_config={"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[1],"rhs_contracting_dimensions":[0],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
    ROOT get-tuple-element = bf16[16,8]{1,0} get-tuple-element(gemm), index=0
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
      backend_config={"fusion_backend_config":{kind: "__custom_fusion", custom_fusion_config: {"name":"cutlass_gemm_with_upcast"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_cublas, hlo_text_custom_fusion,
                                      error_spec, /*run_hlo_passes=*/false));
}

TEST_F(CutlassFusionTest, RowMajorGemmWithDynamicUpdateSliceKernel) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_text_cublas = R"(
  HloModule cublas

  ENTRY e {
    p0 = bf16[2,8,8]{2,1,0} parameter(0)
    p1 = bf16[8,8]{1,0} parameter(1)
    p2 = s32[] parameter(2)
    p3 = s32[] parameter(3)

    gemm.tuple = (bf16[8,8]{1,0}, s8[0]{0}) custom-call(p1, p1),
      custom_call_target="__cublas$gemm",
      backend_config={"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[1],"rhs_contracting_dimensions":[0],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
    gemm = bf16[8,8]{1,0} get-tuple-element(gemm.tuple), index=0
    cast = bf16[1,8,8]{2,1,0} bitcast(gemm)

    ROOT r = bf16[2,8,8]{2,1,0} dynamic-update-slice(p0, cast, p2, p3, p3)
  })";

  const char* hlo_text_custom_fusion = R"(
  HloModule cutlass

  cutlass_gemm {
    p0.1 = bf16[8,8]{1,0} parameter(0)
    p1.1 = bf16[2,8,8]{2,1,0} parameter(1)
    p2 = s32[] parameter(2)
    p3 = s32[] parameter(3)
    dot.1 = bf16[8,8]{1,0} dot(p0.1, p0.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    bc.1 = bf16[1,8,8]{2,1,0} bitcast(dot.1)
    r.1 = bf16[2,8,8]{2,1,0} dynamic-update-slice(p1.1, bc.1, p2, p3, p3)
    workspace = u8[1024]{0} custom-call(),
      custom_call_target="__custom_kernel_fusion$workspace",
      api_version=API_VERSION_TYPED_FFI
    ROOT tuple = (bf16[2,8,8]{2,1,0}, u8[1024]{0}) tuple(r.1, workspace)
  }

  ENTRY e {
    p0 = bf16[2,8,8]{2,1,0} parameter(0)
    p1 = bf16[8,8]{1,0} parameter(1)
    p2 = s32[] parameter(2)
    p3 = s32[] parameter(3)
    r.0 = (bf16[2,8,8]{2,1,0}, u8[1024]{0}) fusion(p1, p0, p2, p3), kind=kCustom,
      calls=%cutlass_gemm,
      backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"cutlass_gemm_with_dynamic_update_slice"}}}
    ROOT %get-tuple-element = bf16[2,8,8]{2,1,0} get-tuple-element(r.0), index=0
  })";

  Array3D<bfloat16> p0_arr(2, 8, 8);  // bf16[2,8,8]
  Array2D<bfloat16> p1_arr(8, 8);     // bf16[8,8]
  p1_arr.Each([](int64_t i, int64_t j, bfloat16* out) {
    *out = bfloat16{1.0f * i * j};
  });

  Array<int32_t> p2_arr({}, 1);
  Array<int32_t> p3_arr({}, 0);

  auto p0 = LiteralUtil::CreateFromArray(p0_arr);
  auto p1 = LiteralUtil::CreateFromArray(p1_arr);
  auto p2 = LiteralUtil::CreateFromArray(p2_arr);
  auto p3 = LiteralUtil::CreateFromArray(p3_arr);

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_cublas, hlo_text_custom_fusion,
                                      {&p0, &p1, &p2, &p3}, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(CutlassFusionTest,
       RowMajorGemmWithDynamicUpdateSliceKernelWithoutBitcast) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_text_cublas = R"(
  HloModule cublas

  ENTRY e {
    p0 = bf16[16,8]{1,0} parameter(0)
    p1 = bf16[8,8]{1,0} parameter(1)
    p2 = s32[] parameter(2)
    p3 = s32[] parameter(3)

    gemm.tuple = (bf16[8,8]{1,0}, s8[0]{0}) custom-call(p1, p1),
      custom_call_target="__cublas$gemm",
      backend_config={"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":[1],"rhs_contracting_dimensions":[0],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},"epilogue":"DEFAULT"}}
    gemm = bf16[8,8]{1,0} get-tuple-element(gemm.tuple), index=0

    ROOT r = bf16[16,8]{1,0} dynamic-update-slice(p0, gemm, p2, p3)
  }
  )";

  const char* hlo_text_custom_fusion = R"(
  HloModule cutlass

  cutlass_gemm {
    p0.1 = bf16[8,8]{1,0} parameter(0)
    p1.1 = bf16[16,8]{1,0} parameter(1)
    p2 = s32[] parameter(2)
    p3 = s32[] parameter(3)
    dot.1 = bf16[8,8]{1,0} dot(p0.1, p0.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    r.1 = bf16[16,8]{1,0} dynamic-update-slice(p1.1, dot.1, p2, p3)
    workspace = u8[1024]{0} custom-call(),
      custom_call_target="__custom_kernel_fusion$workspace",
      api_version=API_VERSION_TYPED_FFI
    ROOT tuple = (bf16[16,8]{1,0}, u8[1024]{0}) tuple(r.1, workspace)
  }

  ENTRY e {
    p0 = bf16[16,8]{1,0} parameter(0)
    p1 = bf16[8,8]{1,0} parameter(1)
    p2 = s32[] parameter(2)
    p3 = s32[] parameter(3)
    r.0 = (bf16[16,8]{1,0}, u8[1024]{0}) fusion(p1, p0, p2, p3), kind=kCustom,
      calls=%cutlass_gemm,
      backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"cutlass_gemm_with_dynamic_update_slice"}}}
    ROOT %get-tuple-element = bf16[16,8]{1,0} get-tuple-element(r.0), index=0
  })";

  Array2D<bfloat16> p0_arr(16, 8);  // bf16[16,8]
  Array2D<bfloat16> p1_arr(8, 8);   // bf16[8,8]
  p1_arr.Each([](int64_t i, int64_t j, bfloat16* out) {
    *out = bfloat16{1.0f * i * j};
  });

  Array<int32_t> p2_arr({}, 0);
  Array<int32_t> p3_arr({}, 1);

  auto p0 = LiteralUtil::CreateFromArray(p0_arr);
  auto p1 = LiteralUtil::CreateFromArray(p1_arr);
  auto p2 = LiteralUtil::CreateFromArray(p2_arr);
  auto p3 = LiteralUtil::CreateFromArray(p3_arr);

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_text_cublas, hlo_text_custom_fusion,
                                      {&p0, &p1, &p2, &p3}, error_spec,
                                      /*run_hlo_passes=*/false));
}

}  // namespace xla::gpu
