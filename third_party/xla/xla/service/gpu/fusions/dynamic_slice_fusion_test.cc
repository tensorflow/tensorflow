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
#include "xla/service/gpu/dynamic_slice_fusion_rewriter.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
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

class DynamicSliceFusionTest : public HloTestBase {
 public:
  HloModuleConfig GetModuleConfigWithoutCommandBuffer() {
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.clear_xla_gpu_enable_command_buffer();
    HloModuleConfig config;
    config.set_debug_options(debug_options);
    return config;
  }
};

TEST_F(DynamicSliceFusionTest, CublasGemmSimple) {
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
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, CublasGemmWithWorkspace) {
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
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, ContiguousSlice) {
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
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, ContiguousSliceNonDefaultLayout) {
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
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, OperandIsSlicedGetTupleElement) {
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
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, ReversedOperandOrder) {
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
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, SingleOperandComputation) {
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
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, SlicedOperandAliasingOutput) {
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
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

static absl::Status Memcpy(se::Stream* stream, ffi::AnyBuffer src,
                           ffi::Result<ffi::AnyBuffer> dst) {
  return stream->MemcpyD2D(
      &dst->data, src.data,
      absl::c_accumulate(src.dimensions, 1.0, std::multiplies<int64_t>()) *
          sizeof(float));
}

XLA_FFI_DEFINE_HANDLER(kMemcpy, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()  // src
                           .Ret<ffi::AnyBuffer>()  // dst
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", PLATFORM,
                         kMemcpy);

TEST_F(DynamicSliceFusionTest, CustomCallSimple) {
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

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_opt, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));
  DynamicSliceFusionRewriter pass(PLATFORM);
  TF_ASSERT_OK_AND_ASSIGN(auto changed, this->RunHloPass(&pass, hlo_opt.get()));
  EXPECT_TRUE(changed);

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec, /*run_hlo_passes=*/false));
}

static absl::Status SubBuffers(
    se::Stream* stream, ffi::AnyBuffer src0, ffi::AnyBuffer src1,
    ffi::AnyBuffer src2, ffi::AnyBuffer src3, ffi::AnyBuffer src4,
    ffi::AnyBuffer src5, ffi::AnyBuffer src6, ffi::AnyBuffer src7,
    ffi::Result<ffi::AnyBuffer> dst0, ffi::Result<ffi::AnyBuffer> dst1,
    ffi::Result<ffi::AnyBuffer> dst2, ffi::Result<ffi::AnyBuffer> dst3,
    ffi::Result<ffi::AnyBuffer> dst4, ffi::Result<ffi::AnyBuffer> dst5,
    ffi::Result<ffi::AnyBuffer> dst6) {
  //  src0:  param 0 at tuple index {0}, shape f32[128]
  //  src1:  param 0 at tuple index {1}, shape f32[256]
  //  src2:  param 1 at tuple index {0}, shape f32[1024]
  //  src3:  param 1 at tuple index {1}, shape f32[8]
  //  src4:  param 2, shape f32[4,8]
  //  src5:  param 3 at tuple index {0, 0}, shape f32[32]
  //  src6:  param 3 at tuple index {0, 1}, shape f32[64]
  //  src7:  param 3 at tuple index {1}, shape f32[3,128]
  //
  //  dst0:  result at tuple index {0}, shape f32[8]
  //  dst1:  result at tuple index {1, 0}, shape f32[128]
  //  dst2:  result at tuple index {1, 1}, shape f32[256]
  //  dst3:  result at tuple index {2}, shape f32[1024]
  //  dst4:  result at tuple index {3}, shape f32[4,8]
  //  dst5:  result at tuple index {4}, shape f32[3,128]
  //  dst6:  result at tuple index {5}, shape f32[96]

  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst0->data, src3.data, 8 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst1->data, src0.data, 128 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst2->data, src1.data, 256 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst3->data, src2.data, 1024 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst4->data, src4.data, 4 * 8 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst5->data, src7.data, 3 * 128 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst6->data, src6.data, 64 * sizeof(float)));
  stream_executor::DeviceMemoryBase slice =
      dst6->data.GetByteSlice(64 * sizeof(float), 32 * sizeof(float));
  TF_RETURN_IF_ERROR(stream->MemcpyD2D(&slice, src6.data, 32 * sizeof(float)));
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kSubBuffers, SubBuffers,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()  // src0
                           .Arg<ffi::AnyBuffer>()  // src1
                           .Arg<ffi::AnyBuffer>()  // src2
                           .Arg<ffi::AnyBuffer>()  // src3
                           .Arg<ffi::AnyBuffer>()  // src4
                           .Arg<ffi::AnyBuffer>()  // src5
                           .Arg<ffi::AnyBuffer>()  // src6
                           .Arg<ffi::AnyBuffer>()  // src7
                           .Ret<ffi::AnyBuffer>()  // dst0
                           .Ret<ffi::AnyBuffer>()  // dst1
                           .Ret<ffi::AnyBuffer>()  // dst2
                           .Ret<ffi::AnyBuffer>()  // dst3
                           .Ret<ffi::AnyBuffer>()  // dst4
                           .Ret<ffi::AnyBuffer>()  // dst5
                           .Ret<ffi::AnyBuffer>()  // dst6
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$subbuffers",
                         PLATFORM, kSubBuffers);

TEST_F(DynamicSliceFusionTest, CustomCallWithTuple) {
  XlaBuilder b(TestName());
  CustomCall(
      &b, "__xla_test$$subbuffers", /*operands=*/
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
          Slice(Broadcast(ConstantR0WithType(&b, F32, 5), {8, 8}), {0, 0},
                {4, 8}, {1, 1}),
          Tuple(&b,
                {
                    Tuple(&b,
                          {
                              Broadcast(ConstantR0WithType(&b, F32, 6), {32}),
                              Broadcast(ConstantR0WithType(&b, F32, 7), {64}),
                          }),
                    Slice(Parameter(&b, 0, ShapeUtil::MakeShape(S32, {4, 128}),
                                    "p0"),
                          {1, 0}, {4, 128}, {1, 1}),
                }),
      },
      ShapeUtil::MakeTupleShape({
          ShapeUtil::MakeShape(F32, {8}),
          ShapeUtil::MakeTupleShape({
              ShapeUtil::MakeShape(F32, {128}),
              ShapeUtil::MakeShape(F32, {256}),
          }),
          ShapeUtil::MakeShape(F32, {1024}),
          ShapeUtil::MakeShape(F32, {4, 8}),
          ShapeUtil::MakeShape(F32, {3, 128}),
          ShapeUtil::MakeShape(F32, {32 + 64}),
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
      /*ignore_layouts=*/true);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_address_computation_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_address_computation_fusion(true);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_opt, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  DynamicSliceFusionRewriter pass(PLATFORM);
  TF_ASSERT_OK_AND_ASSIGN(auto changed, this->RunHloPass(&pass, hlo_opt.get()));
  EXPECT_TRUE(changed);

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec, /*run_hlo_passes=*/false));
}

static absl::Status NoOp(se::Stream* stream, ffi::AnyBuffer operand) {
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kNoOp, NoOp,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()     // stream
                           .Arg<ffi::AnyBuffer>()  // operand
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$noop", PLATFORM,
                         kNoOp);

TEST_F(DynamicSliceFusionTest, NilTuple) {
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

  DynamicSliceFusionRewriter pass(PLATFORM);
  TF_ASSERT_OK_AND_ASSIGN(auto changed, this->RunHloPass(&pass, hlo_opt.get()));
  EXPECT_TRUE(changed);

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec,
                                      /*run_hlo_passes=*/false));
}

void Callback_Memcpy(se::gpu::GpuStreamHandle stream, void** buffers,
                     const char* /*opaque*/, size_t /*opaque_len*/) {
  void* src = buffers[0];
  void* dst = buffers[1];
  auto err = gpuMemcpyAsync(dst, src, /*count=*/sizeof(float) * 3 * 128,
                            gpuMemcpyDeviceToDevice, stream);
  ASSERT_EQ(err, gpuSuccess);
}

XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_Memcpy, PLATFORM);

TEST_F(DynamicSliceFusionTest, CustomCallLegacyAPI) {
  XlaBuilder b(TestName());
  CustomCall(&b, "Callback_Memcpy",
             /*operands=*/
             {Slice(Broadcast(ConstantR0WithType(&b, F32, 42.0), {512}), {128},
                    {4 * 128}, {1})},
             ShapeUtil::MakeShape(F32, {3 * 128}), /*opaque=*/"");
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

  DynamicSliceFusionRewriter pass(PLATFORM);
  TF_ASSERT_OK_AND_ASSIGN(auto changed, this->RunHloPass(&pass, hlo_opt.get()));
  EXPECT_TRUE(changed);

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec,
                                      /*run_hlo_passes=*/false));
}

void Callback_Void(se::gpu::GpuStreamHandle /*stream*/, void** /*buffers*/,
                   const char* /*opaque*/, size_t /*opaque_len*/) {}

XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_Void, PLATFORM);

TEST_F(DynamicSliceFusionTest, NilTupleLegacyAPI) {
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

  DynamicSliceFusionRewriter pass(PLATFORM);
  TF_ASSERT_OK_AND_ASSIGN(auto changed, this->RunHloPass(&pass, hlo_opt.get()));
  EXPECT_TRUE(changed);

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, CublasGemmDynamic) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY main.9 {
    p0 = bf16[2,8,8]{2,1,0} parameter(0)
    p1 = bf16[2,8,8]{2,1,0} parameter(1)
    c1_s32 = s32[] constant(1)
    c0_s32 = s32[] constant(0)
    slice.13 = bf16[1,8,8]{2,1,0} dynamic-slice(p0, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
    bitcast.41 = bf16[8,8]{1,0} bitcast(slice.13)
    slice.14 = bf16[1,8,8]{2,1,0} dynamic-slice(p1, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
    bitcast.42 = bf16[8,8]{1,0} bitcast(slice.14)

    ROOT custom-call.1 = bf16[8,8]{1,0} custom-call(bitcast.41, bitcast.42),
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

  fused_computation {
    p0 = bf16[2,8,8]{2,1,0} parameter(0)
    p1 = bf16[2,8,8]{2,1,0} parameter(1)
    c1_s32 = s32[] parameter(2)
    c0_s32 = s32[] parameter(3)
    slice.13 = bf16[1,8,8]{2,1,0} dynamic-slice(p0, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
    bitcast.41 = bf16[8,8]{1,0} bitcast(slice.13)
    slice.14 = bf16[1,8,8]{2,1,0} dynamic-slice(p1, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
    bitcast.42 = bf16[8,8]{1,0} bitcast(slice.14)

    ROOT custom-call.1 = bf16[8,8]{1,0} custom-call(bitcast.41, bitcast.42),
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

  ENTRY main.9 {
    p0 = bf16[2,8,8]{2,1,0} parameter(0)
    p1 = bf16[2,8,8]{2,1,0} parameter(1)
    c1_s32 = s32[] constant(1)
    c0_s32 = s32[] constant(0)
    ROOT fusion.2 = bf16[8,8]{1,0} fusion(p0, p1, c1_s32, c0_s32), kind=kCustom, calls=fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, CublasGemmDynamicWithWorkspace) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0), sharding={replicated}
    %p1 = f16[2,8,8]{2,1,0} parameter(1), sharding={replicated}
    %c1_s32 = s32[] constant(1)
    %c0_s32 = s32[] constant(0)
    %slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(%p0, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
    %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
    %slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(%p1, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
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
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %c1_s32 = s32[] parameter(2)
    %c0_s32 = s32[] parameter(3)
    %slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(%p0, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
    %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
    %slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(%p1, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
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
    %c1_s32 = s32[] constant(1)
    %c0_s32 = s32[] constant(0)
    ROOT %fusion.2 = (f16[8,8]{1,0}, s8[256]{0}) fusion(%p0, %p1, %c1_s32, %c0_s32), kind=kCustom, calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, DynamicContiguousSlice) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = bf16[2,8,8]{2,1,0} parameter(0), sharding={replicated}
    %p1 = bf16[8,8,10,8]{3,2,1,0} parameter(1), sharding={replicated}
    %c1_s32 = s32[] constant(1)
    %c0_s32 = s32[] constant(0)
    %c2_s32 = s32[] constant(2)
    %c5_s32 = s32[] constant(5)
    %slice.13 = bf16[1,4,8]{2,1,0} dynamic-slice(%p0, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,4,8}
    %bitcast.41 = bf16[4,8]{1,0} bitcast(%slice.13)
    %slice.14 = bf16[1,1,8,8]{3,2,1,0} dynamic-slice(%p1, %c1_s32, %c5_s32, %c2_s32, %c0_s32), dynamic_slice_sizes={1,1,8,8}
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
    %p0 = bf16[2,8,8]{2,1,0} parameter(0)
    %p1 = bf16[8,8,10,8]{3,2,1,0} parameter(1)
    %c1_s32 = s32[] parameter(2)
    %c0_s32 = s32[] parameter(3)
    %c2_s32 = s32[] parameter(4)
    %c5_s32 = s32[] parameter(5)
    %slice.13 = bf16[1,4,8]{2,1,0} dynamic-slice(%p0, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,4,8}
    %bitcast.41 = bf16[4,8]{1,0} bitcast(%slice.13)
    %slice.14 = bf16[1,1,8,8]{3,2,1,0} dynamic-slice(%p1, %c1_s32, %c5_s32, %c2_s32, %c0_s32), dynamic_slice_sizes={1,1,8,8}
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
    %c1_s32 = s32[] constant(1)
    %c0_s32 = s32[] constant(0)
    %c2_s32 = s32[] constant(2)
    %c5_s32 = s32[] constant(5)
    ROOT %fusion.2 = bf16[4,8]{1,0} fusion(%p0, %p1, %c1_s32, %c0_s32, %c2_s32, %c5_s32), kind=kCustom,
    calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, DynamicContiguousSliceNonDefaultLayout) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = bf16[2,8,8]{1,2,0} parameter(0), sharding={replicated}
    %p1 = bf16[8,8,10,8]{1,2,3,0} parameter(1), sharding={replicated}
    %c1_s32 = s32[] constant(1)
    %c0_s32 = s32[] constant(0)
    %c2_s32 = s32[] constant(2)
    %c5_s32 = s32[] constant(5)
    %slice.13 = bf16[1,8,4]{1,2,0} dynamic-slice(%p0, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,4}
    %bitcast.41 = bf16[4,8]{1,0} bitcast(%slice.13)
    %slice.14 = bf16[1,8,8,1]{1,2,3,0} dynamic-slice(%p1, %c0_s32, %c0_s32, %c2_s32, %c5_s32), dynamic_slice_sizes={1,8,8,1}
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
    %p0 = bf16[2,8,8]{1,2,0} parameter(0)
    %p1 = bf16[8,8,10,8]{1,2,3,0} parameter(1)
    %c1_s32 = s32[] parameter(2)
    %c0_s32 = s32[] parameter(3)
    %c2_s32 = s32[] parameter(4)
    %c5_s32 = s32[] parameter(5)
    %slice.13 = bf16[1,8,4]{1,2,0} dynamic-slice(%p0, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,4}
    %bitcast.41 = bf16[4,8]{1,0} bitcast(%slice.13)
    %slice.14 = bf16[1,8,8,1]{1,2,3,0} dynamic-slice(%p1, %c0_s32, %c0_s32, %c2_s32, %c5_s32), dynamic_slice_sizes={1,8,8,1}
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
    %c1_s32 = s32[] constant(1)
    %c0_s32 = s32[] constant(0)
    %c2_s32 = s32[] constant(2)
    %c5_s32 = s32[] constant(5)
    ROOT %fusion.2 = bf16[4,8]{1,0} fusion(%p0, %p1, %c1_s32, %c0_s32, %c2_s32, %c5_s32), kind=kCustom,
    calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, DynamicOperandIsSlicedGetTupleElement) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main {
    %p0 = (f32[100,100]{1,0}, f32[100,100]{1,0}) parameter(0)
    %c0_s32 = s32[] constant(0)
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
    %slice.26 = f32[100,100]{1,0} dynamic-slice(%get-tuple-element.97, %c0_s32, %c0_s32), dynamic_slice_sizes={100,100}
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
    %c0_s32 = s32[] parameter(2)
    %slice.56 = f32[100,100]{1,0} dynamic-slice(%p0.3, %c0_s32, %c0_s32), dynamic_slice_sizes={100,100}
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
    %c0_s32 = s32[] constant(0)
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
    ROOT %address_computation.6 = (f32[100,100]{1,0}, s8[80000]{0}) fusion(%get-tuple-element.97, %get-tuple-element.240, %c0_s32),
      kind=kCustom,
      calls=%address-computation,
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, DynamicReversedOperandOrder) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %c0_s32 = s32[] constant(0)
    %c1_s32 = s32[] constant(1)
    %slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(%p0, %c0_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
    %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(%p1, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
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
    %p1.1 = f16[2,8,8]{2,1,0} parameter(1)
    %c0_s32 = s32[] parameter(2)
    %c1_s32 = s32[] parameter(3)
    %slice.1 = f16[1,8,8]{2,1,0} dynamic-slice(%p0.1, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
    %bitcast.1 = f16[8,8]{1,0} bitcast(%slice.1)
    %slice.0 = f16[1,8,8]{2,1,0} dynamic-slice(%p1.1, %c0_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
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
    %c0_s32 = s32[] constant(0)
    %c1_s32 = s32[] constant(1)
    ROOT %address_computation.6 = f16[8,8]{1,0} fusion(%p1, %p0, %c0_s32, %c1_s32),
      kind=kCustom,
      calls=%address-computation,
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, DynamicSingleOperandComputation) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main {
    %p0 = (f32[100,100]{1,0}, f32[100,100]{1,0}) parameter(0)
    %c0_s32 = s32[] constant(0)
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
    %slice.26 = f32[100,100]{1,0} dynamic-slice(%get-tuple-element.97, %c0_s32, %c0_s32), dynamic_slice_sizes={100,100}
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
    %c0_s32 = s32[] parameter(1)
    %slice.56 = f32[100,100]{1,0} dynamic-slice(%p0.3, %c0_s32, %c0_s32), dynamic_slice_sizes={100,100}
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
    %c0_s32 = s32[] constant(0)
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
    ROOT %address_computation.6 = (f32[100,100]{1,0}, s8[80000]{0}) fusion(%get-tuple-element.97, %c0_s32),
      kind=kCustom,
      calls=%address-computation,
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, DynamicSlicedOperandAliasingOutput) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

    ENTRY %main.9 {
      %p0 = (f32[100,100]{1,0}, f32[100,100]{1,0}) parameter(0)
      %c20_s32 = s32[] constant(20)
      %c99_s32 = s32[] constant(99)
      %c0_s32 = s32[] constant(0)
      %get-tuple-element.287 = f32[100,100]{1,0} get-tuple-element(%p0), index=0
      %get-tuple-element.288 = f32[100,100]{1,0} get-tuple-element(%p0), index=1
      %concatenate.12 = f32[200,100]{1,0} concatenate(%get-tuple-element.287, %get-tuple-element.288), dimensions={0}
      %slice.30 = f32[100,100]{1,0} dynamic-slice(%concatenate.12, %c20_s32, %c0_s32), dynamic_slice_sizes={100,100}
      %slice.34 = f32[100,100]{1,0} dynamic-slice(%concatenate.12, %c99_s32, %c0_s32), dynamic_slice_sizes={100,100}
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
    %p1 = f32[100,100]{1,0} parameter(1)
    %p2 = f32[200,100]{1,0} parameter(2)
    %c0_s32 = s32[] parameter(3)
    %c20_s32 = s32[] parameter(4)
    %slice.0 = f32[100,100]{1,0} dynamic-slice(f32[200,100]{1,0} %p2, %c20_s32, %c0_s32), dynamic_slice_sizes={100,100}
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
    %c20_s32 = s32[] constant(20)
    %c99_s32 = s32[] constant(99)
    %c0_s32 = s32[] constant(0)
    %get-tuple-element.287 = f32[100,100]{1,0} get-tuple-element(%p0), index=0
    %get-tuple-element.288 = f32[100,100]{1,0} get-tuple-element(%p0), index=1
    %concatenate.12 = f32[200,100]{1,0} concatenate(%get-tuple-element.287, %get-tuple-element.288), dimensions={0}
    %slice.34 = f32[100,100]{1,0} dynamic-slice(%concatenate.12, %c99_s32, %c0_s32), dynamic_slice_sizes={100,100}
    ROOT %address_computation.6 = (f32[100,100]{1,0}, s8[120000]{0}) fusion(%get-tuple-element.287, %slice.34, %concatenate.12, %c0_s32, %c20_s32),
      kind=kCustom,
      calls=%address-computation,
      output_to_operand_aliasing={{0}: (1, {})},
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, CublasGemmDUS) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY main.9 {
    p0 = bf16[2,8,8]{2,1,0} parameter(0)
    p1 = bf16[2,8,8]{2,1,0} parameter(1)
    p2 = bf16[4,8,8]{2,1,0} parameter(2)
    c1_s32 = s32[] constant(1)
    c0_s32 = s32[] constant(0)
    slice.13 = bf16[1,8,8]{2,1,0} dynamic-slice(p0, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
    bitcast.41 = bf16[8,8]{1,0} bitcast(slice.13)
    slice.14 = bf16[1,8,8]{2,1,0} dynamic-slice(p1, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
    bitcast.42 = bf16[8,8]{1,0} bitcast(slice.14)

    custom-call.1 = bf16[8,8]{1,0} custom-call(bitcast.41, bitcast.42),
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
    bitcast.43 = bf16[1,8,8]{2,1,0} bitcast(custom-call.1)
    ROOT dus = bf16[4,8,8]{2,1,0} dynamic-update-slice(p2, bitcast.43, c1_s32, c0_s32, c0_s32)
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  fused_computation {
    p0 = bf16[2,8,8]{2,1,0} parameter(0)
    p1 = bf16[2,8,8]{2,1,0} parameter(1)
    p2 = bf16[4,8,8]{2,1,0} parameter(2)
    c1_s32 = s32[] parameter(3)
    c0_s32 = s32[] parameter(4)
    slice.13 = bf16[1,8,8]{2,1,0} dynamic-slice(p0, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
    bitcast.41 = bf16[8,8]{1,0} bitcast(slice.13)
    slice.14 = bf16[1,8,8]{2,1,0} dynamic-slice(p1, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
    bitcast.42 = bf16[8,8]{1,0} bitcast(slice.14)

    custom-call.1 = bf16[8,8]{1,0} custom-call(bitcast.41, bitcast.42),
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
    bitcast.43 = bf16[1,8,8]{2,1,0} bitcast(custom-call.1)
    ROOT dus = bf16[4,8,8]{2,1,0} dynamic-update-slice(p2, bitcast.43, c1_s32, c0_s32, c0_s32)
  }

  ENTRY main.9 {
    p0 = bf16[2,8,8]{2,1,0} parameter(0)
    p1 = bf16[2,8,8]{2,1,0} parameter(1)
    p2 = bf16[4,8,8]{2,1,0} parameter(2)
    c1_s32 = s32[] constant(1)
    c0_s32 = s32[] constant(0)
    ROOT fusion.2 = bf16[4,8,8]{2,1,0} fusion(p0, p1, p2, c1_s32, c0_s32), kind=kCustom, calls=fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  // The GEMM custom call does not have a workspace, shouldn't be run in command
  // buffer.
  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_opt, GetModuleConfigWithoutCommandBuffer(),
      GetModuleConfigWithoutCommandBuffer(), error_spec,
      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, CublasGemmDUSWithWorkspace) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s32[] constant(1)
    %c0_s32 = s32[] constant(0)
    %slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(%p0, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
    %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
    %slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(%p1, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
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
    %bitcast.43 = f16[1,8,8]{2,1,0} bitcast(%get-tuple-element.0)
    %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
    %get-tuple-element.1 = s8[256]{0} get-tuple-element(%custom-call.1), index=1
    ROOT %tuple = (f16[4,8,8]{2,1,0}, s8[256]{0}) tuple(%dus, %get-tuple-element.1)
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  %fused_computation {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s32[] parameter(3)
    %c0_s32 = s32[] parameter(4)
    %slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(%p0, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
    %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
    %slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(%p1, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
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
    %bitcast.43 = f16[1,8,8]{2,1,0} bitcast(%get-tuple-element.0)
    %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
    %get-tuple-element.1 = s8[256]{0} get-tuple-element(%custom-call.1), index=1
    ROOT %tuple = (f16[4,8,8]{2,1,0}, s8[256]{0}) tuple(%dus, %get-tuple-element.1)
  }

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s32[] constant(1)
    %c0_s32 = s32[] constant(0)
    ROOT %fusion.2 = (f16[4,8,8]{2,1,0}, s8[256]{0}) fusion(%p0, %p1, %p2, %c1_s32, %c0_s32), kind=kCustom, calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, CublasGemmDUSWorkspaceIgnored) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = f16[8,8]{1,0} parameter(0)
    %p1 = f16[8,8]{1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s32[] constant(1)
    %c0_s32 = s32[] constant(0)

    %custom-call.1 = (f16[8,8]{1,0}, s8[256]{0}) custom-call(%p0, %p1),
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
    %bitcast.43 = f16[1,8,8]{2,1,0} bitcast(%get-tuple-element.0)
    ROOT %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  %fused_computation {
    %p0 = f16[8,8]{1,0} parameter(0)
    %p1 = f16[8,8]{1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s32[] parameter(3)
    %c0_s32 = s32[] parameter(4)

    %custom-call.1 = (f16[8,8]{1,0}, s8[256]{0}) custom-call(%p0, %p1),
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
    %bitcast.43 = f16[1,8,8]{2,1,0} bitcast(%get-tuple-element.0)
    ROOT %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
  }

  ENTRY %main.9 {
    %p0 = f16[8,8]{1,0} parameter(0)
    %p1 = f16[8,8]{1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s32[] constant(1)
    %c0_s32 = s32[] constant(0)
    ROOT %fusion.2 = f16[4,8,8]{2,1,0} fusion(%p0, %p1, %p2, %c1_s32, %c0_s32), kind=kCustom, calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, CublasGemmDUSOffsetS32NotConstant) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s32[] parameter(3)
    %c0_s32 = s32[] parameter(4)
    %slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(%p0, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
    %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
    %slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(%p1, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
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
    %bitcast.43 = f16[1,8,8]{2,1,0} bitcast(%get-tuple-element.0)
    %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
    %get-tuple-element.1 = s8[256]{0} get-tuple-element(%custom-call.1), index=1
    ROOT %tuple = (f16[4,8,8]{2,1,0}, s8[256]{0}) tuple(%dus, %get-tuple-element.1)
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  %fused_computation {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s32[] parameter(3)
    %c0_s32 = s32[] parameter(4)
    %slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(%p0, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
    %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
    %slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(%p1, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
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
    %bitcast.43 = f16[1,8,8]{2,1,0} bitcast(%get-tuple-element.0)
    %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
    %get-tuple-element.1 = s8[256]{0} get-tuple-element(%custom-call.1), index=1
    ROOT %tuple = (f16[4,8,8]{2,1,0}, s8[256]{0}) tuple(%dus, %get-tuple-element.1)
  }

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s32[] parameter(3)
    %c0_s32 = s32[] parameter(4)
    ROOT %fusion.2 = (f16[4,8,8]{2,1,0}, s8[256]{0}) fusion(%p0, %p1, %p2, %c1_s32, %c0_s32), kind=kCustom, calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, CublasGemmDUSOffsetOOB) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule jit_slice

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s64[] constant(10)
    %c0_s32 = s64[] constant(-1)
    %slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(%p0, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
    %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
    %slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(%p1, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
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
    %bitcast.43 = f16[1,8,8]{2,1,0} bitcast(%get-tuple-element.0)
    %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
    %get-tuple-element.1 = s8[256]{0} get-tuple-element(%custom-call.1), index=1
    ROOT %tuple = (f16[4,8,8]{2,1,0}, s8[256]{0}) tuple(%dus, %get-tuple-element.1)
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  %fused_computation {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s64[] parameter(3)
    %c0_s32 = s64[] parameter(4)
    %slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(%p0, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
    %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
    %slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(%p1, %c1_s32, %c0_s32, %c0_s32), dynamic_slice_sizes={1,8,8}
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
    %bitcast.43 = f16[1,8,8]{2,1,0} bitcast(%get-tuple-element.0)
    %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
    %get-tuple-element.1 = s8[256]{0} get-tuple-element(%custom-call.1), index=1
    ROOT %tuple = (f16[4,8,8]{2,1,0}, s8[256]{0}) tuple(%dus, %get-tuple-element.1)
  }

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s64[] constant(10)
    %c0_s32 = s64[] constant(-1)
    ROOT %fusion.2 = (f16[4,8,8]{2,1,0}, s8[256]{0}) fusion(%p0, %p1, %p2, %c1_s32, %c0_s32), kind=kCustom, calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(hlo_ref, hlo_opt, error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, DynamicCustomCallSimple) {
  XlaBuilder b(TestName());
  CustomCall(
      &b, "__xla_test$$memcpy",
      /*operands=*/
      {DynamicSlice(Parameter(&b, 0, ShapeUtil::MakeShape(S32, {4, 128}), "p0"),
                    {Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}), "start0"),
                     Parameter(&b, 2, ShapeUtil::MakeShape(S32, {}), "start1")},
                    {2, 128})},
      ShapeUtil::MakeShape(F32, {2, 128}), /*opaque=*/"",
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

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_opt, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));
  DynamicSliceFusionRewriter pass(PLATFORM);
  TF_ASSERT_OK_AND_ASSIGN(auto changed, this->RunHloPass(&pass, hlo_opt.get()));
  EXPECT_TRUE(changed);

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec, /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, DynamicCustomCallWithTuple) {
  XlaBuilder b(TestName());
  CustomCall(
      &b, "__xla_test$$subbuffers", /*operands=*/
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
          Slice(Broadcast(ConstantR0WithType(&b, F32, 5), {8, 8}), {0, 0},
                {4, 8}, {1, 1}),
          Tuple(&b,
                {
                    Tuple(&b,
                          {
                              Broadcast(ConstantR0WithType(&b, F32, 6), {32}),
                              Broadcast(ConstantR0WithType(&b, F32, 7), {64}),
                          }),
                    DynamicSlice(
                        Parameter(&b, 0, ShapeUtil::MakeShape(S32, {4, 128}),
                                  "p0"),
                        {Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}),
                                   "start0"),
                         Parameter(&b, 2, ShapeUtil::MakeShape(S32, {}),
                                   "start1")},
                        {3, 128}),
                }),
      },
      ShapeUtil::MakeTupleShape({
          ShapeUtil::MakeShape(F32, {8}),
          ShapeUtil::MakeTupleShape({
              ShapeUtil::MakeShape(F32, {128}),
              ShapeUtil::MakeShape(F32, {256}),
          }),
          ShapeUtil::MakeShape(F32, {1024}),
          ShapeUtil::MakeShape(F32, {4, 8}),
          ShapeUtil::MakeShape(F32, {3, 128}),
          ShapeUtil::MakeShape(F32, {32 + 64}),
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
      /*ignore_layouts=*/true);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_address_computation_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_address_computation_fusion(true);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_opt, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  DynamicSliceFusionRewriter pass(PLATFORM);
  TF_ASSERT_OK_AND_ASSIGN(auto changed, this->RunHloPass(&pass, hlo_opt.get()));
  EXPECT_TRUE(changed);

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec, /*run_hlo_passes=*/false));
}

static absl::Status SubBuffers2(
    se::Stream* stream, ffi::AnyBuffer src0, ffi::AnyBuffer src1,
    ffi::AnyBuffer src2, ffi::AnyBuffer src3, ffi::AnyBuffer src4,
    ffi::AnyBuffer src5, ffi::AnyBuffer src6, ffi::Result<ffi::AnyBuffer> dst0,
    ffi::Result<ffi::AnyBuffer> dst1, ffi::Result<ffi::AnyBuffer> dst2,
    ffi::Result<ffi::AnyBuffer> dst3, ffi::Result<ffi::AnyBuffer> dst4,
    ffi::Result<ffi::AnyBuffer> dst5, ffi::Result<ffi::AnyBuffer> dst6) {
  //  src0:  param 0 at tuple index {0}, shape f32[128]
  //  src1:  param 0 at tuple index {1}, shape f32[256]
  //  src2:  param 1 at tuple index {0}, shape f32[1024]
  //  src3:  param 1 at tuple index {1}, shape f32[8]
  //  src4:  param 2, shape f32[4,8]
  //  src5:  param 3 at tuple index {0, 0}, shape f32[3,128]
  //  src6:  param 3 at tuple index {0, 1}, shape f32[5,128]
  //
  //  dst0:  result at tuple index {0}, shape f32[8]
  //  dst1:  result at tuple index {1, 0}, shape f32[128]
  //  dst2:  result at tuple index {1, 1}, shape f32[256]
  //  dst3:  result at tuple index {2}, shape f32[1024]
  //  dst4:  result at tuple index {3}, shape f32[4,8]
  //  dst5:  result at tuple index {4, 0}, shape f32[5,128]
  //  dst6:  result at tuple index {4, 1}, shape f32[3,128]

  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst0->data, src3.data, 8 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst1->data, src0.data, 128 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst2->data, src1.data, 256 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst3->data, src2.data, 1024 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst4->data, src4.data, 4 * 8 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst5->data, src6.data, 5 * 128 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst6->data, src5.data, 3 * 128 * sizeof(float)));
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kSubBuffers2, SubBuffers2,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()  // src0
                           .Arg<ffi::AnyBuffer>()  // src1
                           .Arg<ffi::AnyBuffer>()  // src2
                           .Arg<ffi::AnyBuffer>()  // src3
                           .Arg<ffi::AnyBuffer>()  // src4
                           .Arg<ffi::AnyBuffer>()  // src5
                           .Arg<ffi::AnyBuffer>()  // src6
                           .Ret<ffi::AnyBuffer>()  // dst0
                           .Ret<ffi::AnyBuffer>()  // dst1
                           .Ret<ffi::AnyBuffer>()  // dst2
                           .Ret<ffi::AnyBuffer>()  // dst3
                           .Ret<ffi::AnyBuffer>()  // dst4
                           .Ret<ffi::AnyBuffer>()  // dst5
                           .Ret<ffi::AnyBuffer>()  // dst6
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$subbuffers2",
                         PLATFORM, kSubBuffers2);

TEST_F(DynamicSliceFusionTest, CustomCallDUS) {
  XlaBuilder b(TestName());
  auto custom_call =
      CustomCall(&b, "Callback_Memcpy",
                 /*operands=*/
                 {Slice(Broadcast(ConstantR0WithType(&b, F32, 42.0), {10, 128}),
                        {2, 0}, {5, 128}, {1, 1})},
                 ShapeUtil::MakeShape(F32, {3, 128}), /*opaque=*/"");

  DynamicUpdateSlice(
      Broadcast(ConstantR0WithType(&b, F32, 92.0), {10, 128}), custom_call,
      {ConstantR0WithType(&b, S32, 4), ConstantR0WithType(&b, S32, 0)});

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

  DynamicSliceFusionRewriter pass(PLATFORM);
  TF_ASSERT_OK_AND_ASSIGN(auto changed, this->RunHloPass(&pass, hlo_opt.get()));
  EXPECT_TRUE(changed);

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec,
                                      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, CustomCallDUSTuple) {
  XlaBuilder b(TestName());
  auto big_buffer1 =
      Parameter(&b, 0, ShapeUtil::MakeShape(F32, {10, 128}), "p0");
  auto big_buffer2 =
      Parameter(&b, 1, ShapeUtil::MakeShape(F32, {10, 256}), "p1");
  auto custom_call = CustomCall(
      &b, "__xla_test$$subbuffers2", /*operands=*/
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
          Slice(Broadcast(ConstantR0WithType(&b, F32, 5), {8, 8}), {0, 0},
                {4, 8}, {1, 1}),
          Tuple(
              &b,
              {
                  Tuple(
                      &b,
                      {
                          Broadcast(ConstantR0WithType(&b, F32, 6), {3, 128}),
                          DynamicSlice(Broadcast(ConstantR0WithType(&b, F32, 7),
                                                 {8, 128}),
                                       {ConstantR0WithType(&b, S32, 2),
                                        ConstantR0WithType(&b, S32, 0)},
                                       {5, 128}),
                      }),
              }),
      },
      ShapeUtil::MakeTupleShape({
          ShapeUtil::MakeShape(F32, {8}),
          ShapeUtil::MakeTupleShape({
              ShapeUtil::MakeShape(F32, {128}),
              ShapeUtil::MakeShape(F32, {256}),
          }),
          ShapeUtil::MakeShape(F32, {1024}),
          ShapeUtil::MakeShape(F32, {4, 8}),
          ShapeUtil::MakeTupleShape({
              ShapeUtil::MakeShape(F32, {5, 128}),
              ShapeUtil::MakeShape(F32, {3, 128}),
          }),
      }),
      /*opaque=*/"",
      /*has_side_effect=*/false,
      /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
      /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  auto tuple_gte = GetTupleElement(custom_call, 4);
  auto dus1 = DynamicUpdateSlice(
      big_buffer1, GetTupleElement(tuple_gte, 0),
      {ConstantR0WithType(&b, S32, 2), ConstantR0WithType(&b, S32, 0)});
  auto dus2 = DynamicUpdateSlice(
      big_buffer1, GetTupleElement(tuple_gte, 1),
      {ConstantR0WithType(&b, S32, 7), ConstantR0WithType(&b, S32, 0)});
  auto dus3 = DynamicUpdateSlice(
      big_buffer2,
      xla::internal::XlaBuilderFriend::BuildBitcast(
          &b, GetTupleElement(custom_call, 2),
          ShapeUtil::MakeShape(F32, {4, 256})),
      {Parameter(&b, 2, ShapeUtil::MakeShape(S32, {}), "start0"),
       Parameter(&b, 3, ShapeUtil::MakeShape(S32, {}), "start1")});
  Tuple(&b, {dus1, dus2, dus3});

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

  DynamicSliceFusionRewriter pass(PLATFORM);
  TF_ASSERT_OK_AND_ASSIGN(auto changed, this->RunHloPass(&pass, hlo_opt.get()));
  EXPECT_TRUE(changed);

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(hlo_ref), std::move(hlo_opt),
                                      error_spec,
                                      /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
