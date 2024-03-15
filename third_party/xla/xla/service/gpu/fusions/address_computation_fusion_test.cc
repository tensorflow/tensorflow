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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include "xla/error_spec.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

#define PLATFORM "CUDA"
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"  // IWYU pragma: keep
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#define PLATFORM "ROCM"
#endif

#if GOOGLE_CUDA
#define gpuSuccess cudaSuccess
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#elif TENSORFLOW_USE_ROCM
#define gpuSuccess hipSuccess
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#endif

namespace xla {
namespace gpu {
namespace {

class AddressComputationFusionTest : public HloTestBase {
 public:
  HloModuleConfig GetRefModuleConfig() {
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_address_computation_fusion(false);
    HloModuleConfig config;
    config.set_debug_options(debug_options);
    return config;
  }

  HloModuleConfig GetOptModuleConfig() {
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_address_computation_fusion(true);
    HloModuleConfig config;
    config.set_debug_options(debug_options);
    return config;
  }
};

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

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, GetRefModuleConfig(),
                                      GetOptModuleConfig(), error_spec,
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

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, GetRefModuleConfig(),
                                      GetOptModuleConfig(), error_spec,
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

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, GetRefModuleConfig(),
                                      GetOptModuleConfig(), error_spec,
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

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, GetRefModuleConfig(),
                                      GetOptModuleConfig(), error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(AddressComputationFusionTest, OperandIsSlicedGetTupleElement) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main {
    %p0 = (f32[100,100]{1,0}, f32[100,100]{1,0}) parameter(0)
    %get-tuple-element.240 = f32[100,100]{1,0} get-tuple-element(%p0), index=0
    %get-tuple-element.241 = f32[100,100]{1,0} get-tuple-element(%p0), index=1
    %concatenate.10 = f32[200,100]{1,0} concatenate(%get-tuple-element.240, %get-tuple-element.241), dimensions={0}
    %custom-call.16 = (f32[200,100]{1,0}, s8[120000]{0}) custom-call(%concatenate.10, %get-tuple-element.240),
      custom_call_target="__cublas$gemm",
      backend_config={
        "gemm_backend_config":{
          "alpha_real":1,
          "beta":0,
          "dot_dimension_numbers":{
            "lhs_contracting_dimensions":["1"],
            "rhs_contracting_dimensions":["0"],
            "lhs_batch_dimensions":[],
            "rhs_batch_dimensions":[]
          },
          "alpha_imag":0,
          "precision_config":{"operand_precision":["HIGHEST","HIGHEST"]},
          "epilogue":"DEFAULT",
          "lhs_stride":"20000",
          "rhs_stride":"10000",
          "grad_x":false,
          "grad_y":false
        }
      }
    %get-tuple-element.97 = f32[200,100]{1,0} get-tuple-element(%custom-call.16), index=0
    %slice.26 = f32[100,100]{1,0} slice(%get-tuple-element.97), slice={[0:100], [0:100]}
    ROOT %custom-call.17 = (f32[100,100]{1,0}, s8[80000]{0}) custom-call(%slice.26, %get-tuple-element.240),
      custom_call_target="__cublas$gemm",
      backend_config={
        "gemm_backend_config":{
          "alpha_real":1,
          "beta":0,
          "dot_dimension_numbers":{
            "lhs_contracting_dimensions":["1"],
            "rhs_contracting_dimensions":["0"],
            "lhs_batch_dimensions":[],
            "rhs_batch_dimensions":[]
          },
          "alpha_imag":0,
          "precision_config":{"operand_precision":["HIGHEST","HIGHEST"]},
          "epilogue":"DEFAULT",
          "lhs_stride":"10000",
          "rhs_stride":"10000",
          "grad_x":false,
          "grad_y":false
        }
      }
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  %address-computation {
    %p0.3 = f32[200,100]{1,0} parameter(0)
    %p1.3 = f32[100,100]{1,0} parameter(1)
    %slice.56 = f32[100,100]{1,0} slice(%p0.3), slice={[0:100], [0:100]}
    %cublas-gemm.23 = (f32[100,100]{1,0}, s8[80000]{0}) custom-call(%slice.56, %p1.3),
      custom_call_target="__cublas$gemm",
      backend_config={
        "gemm_backend_config":{
          "alpha_real":1,
          "beta":0,
          "dot_dimension_numbers":{
            "lhs_contracting_dimensions":["1"],
            "rhs_contracting_dimensions":["0"],
            "lhs_batch_dimensions":[],
            "rhs_batch_dimensions":[]
          },
          "alpha_imag":0,
          "precision_config":{"operand_precision":["HIGHEST","HIGHEST"]},
          "epilogue":"DEFAULT",
          "lhs_stride":"10000",
          "rhs_stride":"10000",
          "grad_x":false,
          "grad_y":false
        }
      }
    %get-tuple-element.221 = f32[100,100]{1,0} get-tuple-element(%cublas-gemm.23), index=0
    %get-tuple-element.222 = s8[80000]{0} get-tuple-element(%cublas-gemm.23), index=1
    ROOT %tuple.58 = (f32[100,100]{1,0}, s8[80000]{0}) tuple(%get-tuple-element.221, %get-tuple-element.222)
  }

  ENTRY %main {
    %p0 = (f32[100,100]{1,0}, f32[100,100]{1,0}) parameter(0)
    %get-tuple-element.240 = f32[100,100]{1,0} get-tuple-element(%p0), index=0
    %get-tuple-element.241 = f32[100,100]{1,0} get-tuple-element(%p0), index=1
    %concatenate.10 = f32[200,100]{1,0} concatenate(%get-tuple-element.240, %get-tuple-element.241), dimensions={0}
    %custom-call.16 = (f32[200,100]{1,0}, s8[120000]{0}) custom-call(%concatenate.10, %get-tuple-element.240),
      custom_call_target="__cublas$gemm",
      backend_config={
        "gemm_backend_config":{
          "alpha_real":1,
          "beta":0,
          "dot_dimension_numbers":{
            "lhs_contracting_dimensions":["1"],
            "rhs_contracting_dimensions":["0"],
            "lhs_batch_dimensions":[],
            "rhs_batch_dimensions":[]
          },
          "alpha_imag":0,
          "precision_config":{"operand_precision":["HIGHEST","HIGHEST"]},
          "epilogue":"DEFAULT",
          "lhs_stride":"20000",
          "rhs_stride":"10000",
          "grad_x":false,
          "grad_y":false
        }
      }
    %get-tuple-element.97 = f32[200,100]{1,0} get-tuple-element(%custom-call.16), index=0
    ROOT %address_computation.6 = (f32[100,100]{1,0}, s8[80000]{0}) fusion(%get-tuple-element.97, %get-tuple-element.240),
      kind=kCustom,
      calls=%address-computation,
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"address_computation"}
        }
      }
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, GetRefModuleConfig(),
                                      GetOptModuleConfig(), error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(AddressComputationFusionTest, ReversedOperandOrder) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %slice.13 = f16[1,8,8]{2,1,0} slice(%p0), slice={[0:1], [0:8], [0:8]}
    %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %slice.14 = f16[1,8,8]{2,1,0} slice(%p1), slice={[1:2], [0:8], [0:8]}
    %bitcast.42 = f16[8,8]{1,0} bitcast(%slice.14)

    ROOT %custom-call.1 = f16[8,8]{1,0} custom-call(%bitcast.42, %bitcast.41),
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

  %address-computation {
    %p0.1 = f16[2,8,8]{2,1,0} parameter(0)
    %slice.1 = f16[1,8,8]{2,1,0} slice(%p0.1), slice={[1:2], [0:8], [0:8]}
    %bitcast.1 = f16[8,8]{1,0} bitcast(%slice.1)
    %p1.1 = f16[2,8,8]{2,1,0} parameter(1)
    %slice.0 = f16[1,8,8]{2,1,0} slice(%p1.1), slice={[0:1], [0:8], [0:8]}
    %bitcast.0 = f16[8,8]{1,0} bitcast(%slice.0)
    ROOT %custom-call.0 = f16[8,8]{1,0} custom-call(%bitcast.1, %bitcast.0),
      custom_call_target="__cublas$gemm",
      backend_config={
        "gemm_backend_config":{
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
        }
      }
  }

  ENTRY %main {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    ROOT %address_computation.6 = f16[8,8]{1,0} fusion(%p1, %p0),
      kind=kCustom,
      calls=%address-computation,
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"address_computation"}
        }
      }
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, GetRefModuleConfig(),
                                      GetOptModuleConfig(), error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(AddressComputationFusionTest, SingleOperandComputation) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main {
    %p0 = (f32[100,100]{1,0}, f32[100,100]{1,0}) parameter(0)
    %get-tuple-element.240 = f32[100,100]{1,0} get-tuple-element(%p0), index=0
    %get-tuple-element.241 = f32[100,100]{1,0} get-tuple-element(%p0), index=1
    %concatenate.10 = f32[200,100]{1,0} concatenate(%get-tuple-element.240, %get-tuple-element.241), dimensions={0}
    %custom-call.16 = (f32[200,100]{1,0}, s8[120000]{0}) custom-call(%concatenate.10, %get-tuple-element.240),
      custom_call_target="__cublas$gemm",
      backend_config={
        "gemm_backend_config":{
          "alpha_real":1,
          "beta":0,
          "dot_dimension_numbers":{
            "lhs_contracting_dimensions":["1"],
            "rhs_contracting_dimensions":["0"],
            "lhs_batch_dimensions":[],
            "rhs_batch_dimensions":[]
          },
          "alpha_imag":0,
          "precision_config":{"operand_precision":["HIGHEST","HIGHEST"]},
          "epilogue":"DEFAULT",
          "lhs_stride":"20000",
          "rhs_stride":"10000",
          "grad_x":false,
          "grad_y":false
        }
      }
    %get-tuple-element.97 = f32[200,100]{1,0} get-tuple-element(%custom-call.16), index=0
    %slice.26 = f32[100,100]{1,0} slice(%get-tuple-element.97), slice={[0:100], [0:100]}
    ROOT %custom-call.17 = (f32[100,100]{1,0}, s8[80000]{0}) custom-call(%slice.26, %slice.26),
      custom_call_target="__cublas$gemm",
      backend_config={
        "gemm_backend_config":{
          "alpha_real":1,
          "beta":0,
          "dot_dimension_numbers":{
            "lhs_contracting_dimensions":["1"],
            "rhs_contracting_dimensions":["0"],
            "lhs_batch_dimensions":[],
            "rhs_batch_dimensions":[]
          },
          "alpha_imag":0,
          "precision_config":{"operand_precision":["HIGHEST","HIGHEST"]},
          "epilogue":"DEFAULT",
          "lhs_stride":"10000",
          "rhs_stride":"10000",
          "grad_x":false,
          "grad_y":false
        }
      }
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  %address-computation {
    %p0.3 = f32[200,100]{1,0} parameter(0)
    %slice.56 = f32[100,100]{1,0} slice(%p0.3), slice={[0:100], [0:100]}
    %cublas-gemm.23 = (f32[100,100]{1,0}, s8[80000]{0}) custom-call(%slice.56, %slice.56),
      custom_call_target="__cublas$gemm",
      backend_config={
        "gemm_backend_config":{
          "alpha_real":1,
          "beta":0,
          "dot_dimension_numbers":{
            "lhs_contracting_dimensions":["1"],
            "rhs_contracting_dimensions":["0"],
            "lhs_batch_dimensions":[],
            "rhs_batch_dimensions":[]
          },
          "alpha_imag":0,
          "precision_config":{"operand_precision":["HIGHEST","HIGHEST"]},
          "epilogue":"DEFAULT",
          "lhs_stride":"10000",
          "rhs_stride":"10000",
          "grad_x":false,
          "grad_y":false
        }
      }
    %get-tuple-element.221 = f32[100,100]{1,0} get-tuple-element(%cublas-gemm.23), index=0
    %get-tuple-element.222 = s8[80000]{0} get-tuple-element(%cublas-gemm.23), index=1
    ROOT %tuple.58 = (f32[100,100]{1,0}, s8[80000]{0}) tuple(%get-tuple-element.221, %get-tuple-element.222)
  }

  ENTRY %main {
    %p0 = (f32[100,100]{1,0}, f32[100,100]{1,0}) parameter(0)
    %get-tuple-element.240 = f32[100,100]{1,0} get-tuple-element(%p0), index=0
    %get-tuple-element.241 = f32[100,100]{1,0} get-tuple-element(%p0), index=1
    %concatenate.10 = f32[200,100]{1,0} concatenate(%get-tuple-element.240, %get-tuple-element.241), dimensions={0}
    %custom-call.16 = (f32[200,100]{1,0}, s8[120000]{0}) custom-call(%concatenate.10, %get-tuple-element.240),
      custom_call_target="__cublas$gemm",
      backend_config={
        "gemm_backend_config":{
          "alpha_real":1,
          "beta":0,
          "dot_dimension_numbers":{
            "lhs_contracting_dimensions":["1"],
            "rhs_contracting_dimensions":["0"],
            "lhs_batch_dimensions":[],
            "rhs_batch_dimensions":[]
          },
          "alpha_imag":0,
          "precision_config":{"operand_precision":["HIGHEST","HIGHEST"]},
          "epilogue":"DEFAULT",
          "lhs_stride":"20000",
          "rhs_stride":"10000",
          "grad_x":false,
          "grad_y":false
        }
      }
    %get-tuple-element.97 = f32[200,100]{1,0} get-tuple-element(%custom-call.16), index=0
    ROOT %address_computation.6 = (f32[100,100]{1,0}, s8[80000]{0}) fusion(%get-tuple-element.97),
      kind=kCustom,
      calls=%address-computation,
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"address_computation"}
        }
      }
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, GetRefModuleConfig(),
                                      GetOptModuleConfig(), error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(AddressComputationFusionTest, SlicedOperandAliasingOutput) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

    ENTRY %main.9 {
      %p0 = (f32[100,100]{1,0}, f32[100,100]{1,0}) parameter(0)
      %get-tuple-element.287 = f32[100,100]{1,0} get-tuple-element(%p0), index=0
      %get-tuple-element.288 = f32[100,100]{1,0} get-tuple-element(%p0), index=1
      %concatenate.12 = f32[200,100]{1,0} concatenate(%get-tuple-element.287, %get-tuple-element.288), dimensions={0}
      %slice.30 = f32[100,100]{1,0} slice(%concatenate.12), slice={[20:120], [0:100]}
      %slice.34 = f32[100,100]{1,0} slice(%concatenate.12), slice={[99:199], [0:100]}
      ROOT %cublas-gemm.15 = (f32[100,100]{1,0}, s8[120000]{0}) custom-call(%get-tuple-element.287, %slice.30, %slice.34),
        custom_call_target="__cublas$gemm",
        output_to_operand_aliasing={{0}: (2, {})},
        backend_config={"gemm_backend_config":{
          "alpha_real":1,
          "beta":1,
          "dot_dimension_numbers":{
            "lhs_contracting_dimensions":["1"],
            "rhs_contracting_dimensions":["0"],
            "lhs_batch_dimensions":[],
            "rhs_batch_dimensions":[]
          },
          "alpha_imag":0,
          "precision_config":{"operand_precision":["HIGHEST","HIGHEST"]},
          "epilogue":"DEFAULT",
          "lhs_stride":"10000",
          "rhs_stride":"10000",
          "grad_x":false,
          "grad_y":false
        }}
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  %address-computation {
    %p0.1 = f32[100,100]{1,0} parameter(0)
    %p2 = f32[200,100]{1,0} parameter(2)
    %slice.0 = f32[100,100]{1,0} slice(f32[200,100]{1,0} %p2), slice={[20:120], [0:100]}
    %p1 = f32[100,100]{1,0} parameter(1)
    %cublas-gemm.0 = (f32[100,100]{1,0}, s8[120000]{0}) custom-call(%p0.1, %slice.0, %p1),
      custom_call_target="__cublas$gemm",
      backend_config={
        "gemm_backend_config":{
          "alpha_real":1,
          "beta":1,
          "dot_dimension_numbers":{
            "lhs_contracting_dimensions":["1"],
            "rhs_contracting_dimensions":["0"],
            "lhs_batch_dimensions":[],
            "rhs_batch_dimensions":[]
          },
          "alpha_imag":0,
          "precision_config":{"operand_precision":["HIGHEST","HIGHEST"]},
          "epilogue":"DEFAULT",
          "lhs_stride":"10000",
          "rhs_stride":"10000",
          "grad_x":false,
          "grad_y":false
        }
      }
    %get-tuple-element = f32[100,100]{1,0} get-tuple-element(%cublas-gemm.0), index=0
    %get-tuple-element.1 = s8[120000]{0} get-tuple-element(%cublas-gemm.0), index=1
    ROOT %tuple = (f32[100,100]{1,0}, s8[120000]{0}) tuple(%get-tuple-element, %get-tuple-element.1)
  }

  ENTRY %main {
    %p0 = (f32[100,100]{1,0}, f32[100,100]{1,0}) parameter(0)
    %get-tuple-element.287 = f32[100,100]{1,0} get-tuple-element(%p0), index=0
    %get-tuple-element.288 = f32[100,100]{1,0} get-tuple-element(%p0), index=1
    %concatenate.12 = f32[200,100]{1,0} concatenate(%get-tuple-element.287, %get-tuple-element.288), dimensions={0}
    %slice.34 = f32[100,100]{1,0} slice(%concatenate.12), slice={[99:199], [0:100]}
    ROOT %address_computation.6 = (f32[100,100]{1,0}, s8[120000]{0}) fusion(%get-tuple-element.287, %slice.34, %concatenate.12),
      kind=kCustom,
      calls=%address-computation,
      output_to_operand_aliasing={{0}: (1, {})},
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"address_computation"}
        }
      }
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, GetRefModuleConfig(),
                                      GetOptModuleConfig(), error_spec,
                                      /*run_hlo_passes=*/false));
}

static absl::Status Memcpy(se::Stream* stream, ffi::BufferBase src,
                           ffi::BufferBase dst) {
  return stream->MemcpyD2D(
      &dst.data, src.data,
      absl::c_accumulate(src.dimensions, 1.0, std::multiplies<int64_t>()) *
          sizeof(float));
}

XLA_FFI_DEFINE_HANDLER(kMemcpy, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<se::Stream>()
                           .Arg<ffi::BufferBase>()  // src
                           .Arg<ffi::BufferBase>()  // dst
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", PLATFORM,
                         kMemcpy);

TEST_F(AddressComputationFusionTest, CustomCallSimple) {
  XlaBuilder b(TestName());
  CustomCall(&b, "__xla_test$$memcpy",
             /*operands=*/
             {Slice(Broadcast(ConstantR0WithType(&b, F32, 42.0), {256}), {0},
                    {128}, {1})},
             ShapeUtil::MakeShape(F32, {128}), /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  xla::HloModuleConfig hlo_config(
      xla::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_address_computation_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_address_computation_fusion(true);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_opt, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec,
                                      /*run_hlo_passes=*/false));
}

static absl::Status SubBuffers(se::Stream* stream, ffi::BufferBase src0,
                               ffi::BufferBase src1, ffi::BufferBase src2,
                               ffi::BufferBase src3, ffi::BufferBase src4,
                               ffi::BufferBase dst0, ffi::BufferBase dst1,
                               ffi::BufferBase dst2, ffi::BufferBase dst3,
                               ffi::BufferBase dst4) {
  //  src0:  param 0 at tuple index {0}, shape f32[128]
  //  src1:  param 0 at tuple index {1}, shape f32[256]
  //  src2:  param 1 at tuple index {0}, shape f32[1024]
  //  src3:  param 1 at tuple index {1}, shape f32[8]
  //  src4:  param 2, shape f32[4,8]
  //
  //  dst0:  result at tuple index {0}, shape f32[8]
  //  dst1:  result at tuple index {1, 0}, shape f32[128]
  //  dst2:  result at tuple index {1, 1}, shape f32[256]
  //  dst3:  result at tuple index {2}, shape f32[1024]
  //  dst4:  result at tuple index {3}, shape f32[4,8]

  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst0.data, src3.data, 8 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst1.data, src0.data, 128 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst2.data, src1.data, 256 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst3.data, src2.data, 1024 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst4.data, src4.data, 4 * 8 * sizeof(float)));
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kSubBuffers, SubBuffers,
                       ffi::Ffi::Bind()
                           .Ctx<se::Stream>()
                           .Arg<ffi::BufferBase>()  // src0
                           .Arg<ffi::BufferBase>()  // src1
                           .Arg<ffi::BufferBase>()  // src2
                           .Arg<ffi::BufferBase>()  // src3
                           .Arg<ffi::BufferBase>()  // src4
                           .Arg<ffi::BufferBase>()  // dst0
                           .Arg<ffi::BufferBase>()  // dst1
                           .Arg<ffi::BufferBase>()  // dst2
                           .Arg<ffi::BufferBase>()  // dst3
                           .Arg<ffi::BufferBase>()  // dst4
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$subbuffers",
                         PLATFORM, kSubBuffers);

TEST_F(AddressComputationFusionTest, CustomCallWithTuple) {
  XlaBuilder b(TestName());
  CustomCall(&b, "__xla_test$$subbuffers", /*operands=*/
             {
                 Tuple(&b,
                       {
                           Broadcast(ConstantR0WithType(&b, F32, 1), {128}),
                           Broadcast(ConstantR0WithType(&b, F32, 2), {256}),
                       }),
                 Tuple(&b,
                       {
                           Broadcast(ConstantR0WithType(&b, F32, 3), {1024}),
                           Broadcast(ConstantR0WithType(&b, F32, 4), {8}),
                       }),
                 Slice(Broadcast(ConstantR0WithType(&b, F32, 5), {8, 8}),
                       {0, 0}, {4, 8}, {1, 1}),
             },
             ShapeUtil::MakeTupleShape({
                 ShapeUtil::MakeShape(F32, {8}),
                 ShapeUtil::MakeTupleShape({
                     ShapeUtil::MakeShape(F32, {128}),
                     ShapeUtil::MakeShape(F32, {256}),
                 }),
                 ShapeUtil::MakeShape(F32, {1024}),
                 ShapeUtil::MakeShape(F32, {4, 8}),
             }),
             /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  xla::HloModuleConfig hlo_config(
      xla::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_address_computation_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_address_computation_fusion(true);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_opt, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec,
                                      /*run_hlo_passes=*/false));
}

static absl::Status NoOp(se::Stream* stream, ffi::BufferBase operand) {
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(
    kNoOp, NoOp,
    ffi::Ffi::Bind().Ctx<se::Stream>().Arg<ffi::BufferBase>()  // operand
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$noop", PLATFORM,
                         kNoOp);

TEST_F(AddressComputationFusionTest, NilTuple) {
  XlaBuilder b(TestName());
  CustomCall(&b, "__xla_test$$noop",
             /*operands=*/
             {Slice(Broadcast(ConstantR0WithType(&b, F32, 42.0), {256}), {0},
                    {128}, {1})},
             ShapeUtil::MakeNil(),
             /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  xla::HloModuleConfig hlo_config(
      xla::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_address_computation_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_address_computation_fusion(true);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_opt, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec,
                                      /*run_hlo_passes=*/false));
}

void Callback_Memcpy(se::gpu::GpuStreamHandle stream, void** buffers,
                     const char* /*opaque*/, size_t /*opaque_len*/) {
  void* src = buffers[0];
  void* dst = buffers[1];
  auto err = gpuMemcpyAsync(dst, src, /*count=*/sizeof(float) * 128,
                            gpuMemcpyDeviceToDevice, stream);
  ASSERT_EQ(err, gpuSuccess);
}

XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_Memcpy, PLATFORM);

TEST_F(AddressComputationFusionTest, CustomCallLegacyAPI) {
  XlaBuilder b(TestName());
  CustomCall(&b, "Callback_Memcpy",
             /*operands=*/
             {Slice(Broadcast(ConstantR0WithType(&b, F32, 42.0), {256}), {0},
                    {128}, {1})},
             ShapeUtil::MakeShape(F32, {128}), /*opaque=*/"");
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  xla::HloModuleConfig hlo_config(
      xla::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_address_computation_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_address_computation_fusion(true);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_opt, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec,
                                      /*run_hlo_passes=*/false));
}

void Callback_Void(se::gpu::GpuStreamHandle /*stream*/, void** /*buffers*/,
                   const char* /*opaque*/, size_t /*opaque_len*/) {}

XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_Void, PLATFORM);

TEST_F(AddressComputationFusionTest, NilTupleLegacyAPI) {
  XlaBuilder b(TestName());
  CustomCall(&b, "Callback_Void", /*operands=*/
             {Slice(Broadcast(ConstantR0WithType(&b, F32, 42.0), {256}), {0},
                    {128}, {1})},
             ShapeUtil::MakeNil(),
             /*opaque=*/"");
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  xla::HloModuleConfig hlo_config(
      xla::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_address_computation_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_address_computation_fusion(true);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_opt, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec,
                                      /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
