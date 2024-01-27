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
#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/error_spec.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/types.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class AddressComputationFusionTest : public HloTestBase {};

TEST_F(AddressComputationFusionTest, CublasGemmSimple) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = bf16[2,8,8]{2,1,0} parameter(0), sharding={replicated}
    %p1 = bf16[2,8,8]{2,1,0} parameter(1), sharding={replicated}
    %slice.13 = bf16[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
    %bitcast.41 = bf16[8,8]{1,0} bitcast(%slice.13)
    %slice.14 = bf16[1,8,8]{2,1,0} slice(%p1), slice={[1:2], [0:8], [0:8]}
    %bitcast.42 = bf16[8,8]{1,0} bitcast(%slice.14)

    ROOT %custom-call.1 = bf16[8,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
      custom_call_target="__cublas$gemm",
      backend_config={"gemm_backend_config":{
        "alpha_real":1,
        "beta":0,
        "dot_dimension_numbers":{
          "lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["0"],
          "lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]
        },
        "alpha_imag":0,
        "precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},
        "epilogue":"DEFAULT",
        "lhs_stride":"64",
        "rhs_stride":"64",
        "grad_x":false,
        "grad_y":false
      }}
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  %fused_computation {
    %param_0_0 = bf16[2,8,8]{2,1,0} parameter(0)
    %slice.13 = bf16[1,8,8]{2,1,0} slice(%param_0_0), slice={[1:2], [0:8], [0:8]}
    %bitcast.41 = bf16[8,8]{1,0} bitcast(%slice.13)
    %param_1_0 = bf16[2,8,8]{2,1,0} parameter(1)
    %slice.14 = bf16[1,8,8]{2,1,0} slice(%param_1_0), slice={[1:2], [0:8], [0:8]}
    %bitcast.42 = bf16[8,8]{1,0} bitcast(%slice.14)

    ROOT %custom-call.1 = bf16[8,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
      custom_call_target="__cublas$gemm",
      backend_config={"gemm_backend_config":{
        "alpha_real":1,
        "beta":0,
        "dot_dimension_numbers":{
          "lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["0"],
          "lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]
        },
        "alpha_imag":0,
        "precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},
        "epilogue":"DEFAULT",
        "lhs_stride":"64",
        "rhs_stride":"64",
        "grad_x":false,
        "grad_y":false
      }}
  }

  ENTRY %main.9 {
    %p0 = bf16[2,8,8]{2,1,0} parameter(0), sharding={replicated}
    %p1 = bf16[2,8,8]{2,1,0} parameter(1), sharding={replicated}
    ROOT %fusion.2 = bf16[8,8]{1,0} fusion(%p0, %p1), kind=kCustom, calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"address_computation"}}}
  })";

  Array3D<bfloat16> arr0(2, 8, 8);  // bf16[2,8,8]
  Array3D<bfloat16> arr1(2, 8, 8);  // bf16[2,8,8]
  arr0.FillIota(static_cast<bfloat16>(1.0));
  arr1.FillRandom(bfloat16(0.01f), 0.02);

  auto a0 = LiteralUtil::CreateFromArray(arr0);
  auto a1 = LiteralUtil::CreateFromArray(arr1);

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, {&a0, &a1}, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(AddressComputationFusionTest, CublasGemmWithWorkspace) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0), sharding={replicated}
    %p1 = f16[2,8,8]{2,1,0} parameter(1), sharding={replicated}
    %slice.13 = f16[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
    %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
    %slice.14 = f16[1,8,8]{2,1,0} slice(%p1), slice={[1:2], [0:8], [0:8]}
    %bitcast.42 = f16[8,8]{1,0} bitcast(%slice.14)

    ROOT %custom-call.1 = (f16[8,8]{1,0}, s8[256]{0}) custom-call(%bitcast.41, %bitcast.42),
      custom_call_target="__cublas$gemm",
      backend_config={"gemm_backend_config":{
        "alpha_real":1,
        "beta":0,
        "dot_dimension_numbers":{
          "lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["0"],
          "lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]
        },
        "alpha_imag":0,
        "precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},
        "epilogue":"DEFAULT",
        "lhs_stride":"64",
        "rhs_stride":"64",
        "grad_x":false,
        "grad_y":false
      }}
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  %fused_computation {
    %param_0_0 = f16[2,8,8]{2,1,0} parameter(0)
    %slice.13 = f16[1,8,8]{2,1,0} slice(%param_0_0), slice={[1:2], [0:8], [0:8]}
    %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
    %param_1_0 = f16[2,8,8]{2,1,0} parameter(1)
    %slice.14 = f16[1,8,8]{2,1,0} slice(%param_1_0), slice={[1:2], [0:8], [0:8]}
    %bitcast.42 = f16[8,8]{1,0} bitcast(%slice.14)

    %custom-call.1 = (f16[8,8]{1,0}, s8[256]{0}) custom-call(%bitcast.41, %bitcast.42),
      custom_call_target="__cublas$gemm",
      backend_config={"gemm_backend_config":{
        "alpha_real":1,
        "beta":0,
        "dot_dimension_numbers":{
          "lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["0"],
          "lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]
        },
        "alpha_imag":0,
        "precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},
        "epilogue":"DEFAULT",
        "lhs_stride":"64",
        "rhs_stride":"64",
        "grad_x":false,
        "grad_y":false
      }}
    %get-tuple-element.0 = f16[8,8]{1,0} get-tuple-element(%custom-call.1), index=0
    %get-tuple-element.1 = s8[256]{0} get-tuple-element(%custom-call.1), index=1
    ROOT %tuple = (f16[8,8]{1,0}, s8[256]{0}) tuple(%get-tuple-element.0, %get-tuple-element.1)
  }

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0), sharding={replicated}
    %p1 = f16[2,8,8]{2,1,0} parameter(1), sharding={replicated}
    ROOT %fusion.2 = (f16[8,8]{1,0}, s8[256]{0}) fusion(%p0, %p1), kind=kCustom, calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"address_computation"}}}
  })";

  Array3D<bfloat16> arr0(2, 8, 8);  // bf16[2,8,8]
  Array3D<bfloat16> arr1(2, 8, 8);  // bf16[2,8,8]
  arr0.FillRandom(bfloat16(0.01f), 0.02);
  arr1.FillIota(static_cast<bfloat16>(10.0));

  auto a0 = LiteralUtil::CreateFromArray(arr0);
  auto a1 = LiteralUtil::CreateFromArray(arr1);

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, {&a0, &a1}, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(AddressComputationFusionTest, ContiguousSlice) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = bf16[2,8,8]{2,1,0} parameter(0), sharding={replicated}
    %p1 = bf16[8,8,10,8]{3,2,1,0} parameter(1), sharding={replicated}
    %slice.13 = bf16[1,4,8]{2,1,0} slice(%p0), slice={[1:2], [0:4], [0:8]}
    %bitcast.41 = bf16[4,8]{1,0} bitcast(%slice.13)
    %slice.14 = bf16[1,1,8,8]{3,2,1,0} slice(%p1), slice={[0:1], [5:6], [2:10], [0:8]}
    %bitcast.42 = bf16[8,8]{1,0} bitcast(%slice.14)

    ROOT %custom-call.1 = bf16[4,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
      custom_call_target="__cublas$gemm",
      backend_config={"gemm_backend_config":{
        "alpha_real":1,
        "beta":0,
        "dot_dimension_numbers":{
          "lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["0"],
          "lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]
        },
        "alpha_imag":0,
        "precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},
        "epilogue":"DEFAULT",
        "lhs_stride":"64",
        "rhs_stride":"64",
        "grad_x":false,
        "grad_y":false
      }}
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  %fused_computation {
    %param_0_0 = bf16[2,8,8]{2,1,0} parameter(0)
    %slice.13 = bf16[1,4,8]{2,1,0} slice(%param_0_0), slice={[1:2], [0:4], [0:8]}
    %bitcast.41 = bf16[4,8]{1,0} bitcast(%slice.13)
    %param_1_0 = bf16[8,8,10,8]{3,2,1,0} parameter(1)
    %slice.14 = bf16[1,1,8,8]{3,2,1,0} slice(%param_1_0), slice={[0:1], [5:6], [2:10], [0:8]}
    %bitcast.42 = bf16[8,8]{1,0} bitcast(%slice.14)

    ROOT %custom-call.1 = bf16[4,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
      custom_call_target="__cublas$gemm",
      backend_config={"gemm_backend_config":{
        "alpha_real":1,
        "beta":0,
        "dot_dimension_numbers":{
          "lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["0"],
          "lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]
        },
        "alpha_imag":0,
        "precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},
        "epilogue":"DEFAULT",
        "lhs_stride":"64",
        "rhs_stride":"64",
        "grad_x":false,
        "grad_y":false
      }}
  }

  ENTRY %main.9 {
    %p0 = bf16[2,8,8]{2,1,0} parameter(0), sharding={replicated}
    %p1 = bf16[8,8,10,8]{3,2,1,0} parameter(1), sharding={replicated}
    ROOT %fusion.2 = bf16[4,8]{1,0} fusion(%p0, %p1), kind=kCustom, calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"address_computation"}}}
  })";

  Array3D<bfloat16> arr0(2, 8, 8);      // bf16[2,8,8]
  Array4D<bfloat16> arr1(8, 8, 10, 8);  // bf16[8,8,10,8]
  arr0.FillIota(static_cast<bfloat16>(1.0));
  arr1.FillRandom(bfloat16(0.01f), 0.02);

  auto a0 = LiteralUtil::CreateFromArray(arr0);
  auto a1 = LiteralUtil::CreateFromArray(arr1);

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, {&a0, &a1}, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(AddressComputationFusionTest, ContiguousSliceNonDefaultLayout) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = bf16[2,8,8]{1,2,0} parameter(0), sharding={replicated}
    %p1 = bf16[8,8,10,8]{1,2,3,0} parameter(1), sharding={replicated}
    %slice.13 = bf16[1,8,4]{1,2,0} slice(%p0), slice={[1:2], [0:8], [0:4]}
    %bitcast.41 = bf16[4,8]{1,0} bitcast(%slice.13)
    %slice.14 = bf16[1,8,8,1]{1,2,3,0} slice(%p1), slice={[0:1], [0:8], [2:10], [5:6]}
    %bitcast.42 = bf16[8,8]{1,0} bitcast(%slice.14)

    ROOT %custom-call.1 = bf16[4,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
      custom_call_target="__cublas$gemm",
      backend_config={"gemm_backend_config":{
        "alpha_real":1,
        "beta":0,
        "dot_dimension_numbers":{
          "lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["0"],
          "lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]
        },
        "alpha_imag":0,
        "precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},
        "epilogue":"DEFAULT",
        "lhs_stride":"64",
        "rhs_stride":"64",
        "grad_x":false,
        "grad_y":false
      }}
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  %fused_computation {
    %param_0_0 = bf16[2,8,8]{1,2,0} parameter(0)
    %slice.13 = bf16[1,8,4]{1,2,0} slice(%param_0_0), slice={[1:2], [0:8], [0:4]}
    %bitcast.41 = bf16[4,8]{1,0} bitcast(%slice.13)
    %param_1_0 = bf16[8,8,10,8]{1,2,3,0} parameter(1)
    %slice.14 = bf16[1,8,8,1]{1,2,3,0} slice(%param_1_0), slice={[0:1], [0:8], [2:10], [5:6]}
    %bitcast.42 = bf16[8,8]{1,0} bitcast(%slice.14)

    ROOT %custom-call.1 = bf16[4,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
      custom_call_target="__cublas$gemm",
      backend_config={"gemm_backend_config":{
        "alpha_real":1,
        "beta":0,
        "dot_dimension_numbers":{
          "lhs_contracting_dimensions":["1"],
          "rhs_contracting_dimensions":["0"],
          "lhs_batch_dimensions":[],
          "rhs_batch_dimensions":[]
        },
        "alpha_imag":0,
        "precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},
        "epilogue":"DEFAULT",
        "lhs_stride":"64",
        "rhs_stride":"64",
        "grad_x":false,
        "grad_y":false
      }}
  }

  ENTRY %main.9 {
    %p0 = bf16[2,8,8]{1,2,0} parameter(0), sharding={replicated}
    %p1 = bf16[8,8,10,8]{1,2,3,0} parameter(1), sharding={replicated}
    ROOT %fusion.2 = bf16[4,8]{1,0} fusion(%p0, %p1), kind=kCustom, calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"address_computation"}}}
  })";

  Array3D<bfloat16> arr0(2, 8, 8);      // bf16[2,8,8]
  Array4D<bfloat16> arr1(8, 8, 10, 8);  // bf16[8,8,10,8]
  arr0.FillIota(static_cast<bfloat16>(1.0));
  arr1.FillRandom(bfloat16(0.01f), 0.02);

  auto a0 = LiteralUtil::CreateFromArray(arr0);
  auto a1 = LiteralUtil::CreateFromArray(arr1);

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, {&a0, &a1}, error_spec,
                                      /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
