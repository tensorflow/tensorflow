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
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/error_spec.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/service/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/gpu/transforms/dynamic_slice_fusion_rewriter.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/filecheck.h"
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

  HloModuleConfig GetModuleConfigWithDeterministicOps() {
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_exclude_nondeterministic_ops(true);
    HloModuleConfig config;
    config.set_debug_options(debug_options);
    return config;
  }

  std::vector<HloComputation*> GetDynamicSliceFusions(const HloModule& module) {
    std::vector<HloComputation*> computations;
    for (auto computation : module.computations()) {
      if (!computation->IsFusionComputation()) {
        continue;
      }
      auto backend_config = computation->FusionInstruction()
                                ->backend_config<xla::gpu::GpuBackendConfig>();
      if (backend_config.ok()) {
        const FusionBackendConfig& fusion_backend_config =
            backend_config.value().fusion_backend_config();
        const std::string name =
            fusion_backend_config.custom_fusion_config().name();
        if (name == "dynamic_address_computation" ||
            name == "address_computation") {
          computations.push_back(computation);
        }
      }
    }
    return computations;
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
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"address_computation"}}}
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

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_opt, GetModuleConfigWithDeterministicOps(),
      GetModuleConfigWithDeterministicOps(), error_spec,
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

  %dynamic-slice-fusion {
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
    ROOT %dynamic-slice-fusion.6 = (f32[100,100]{1,0}, s8[80000]{0}) fusion(%get-tuple-element.97, %get-tuple-element.240),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
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

  %dynamic-slice-fusion {
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
    ROOT %dynamic-slice-fusion.6 = f16[8,8]{1,0} fusion(%p1, %p0),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
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

  %dynamic-slice-fusion {
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
    ROOT %dynamic-slice-fusion.6 = (f32[100,100]{1,0}, s8[80000]{0}) fusion(%get-tuple-element.97),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
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

  %dynamic-slice-fusion {
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
    ROOT %dynamic-slice-fusion.6 = (f32[100,100]{1,0}, s8[120000]{0}) fusion(%get-tuple-element.287, %slice.34, %concatenate.12),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
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
  se::DeviceMemoryBase dst_mem = dst->device_memory();
  se::DeviceMemoryBase src_mem = src.device_memory();
  return stream->MemcpyD2D(&dst_mem, src_mem, src_mem.size());
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
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
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

  se::DeviceMemoryBase dst0_mem = dst0->device_memory();
  se::DeviceMemoryBase dst1_mem = dst1->device_memory();
  se::DeviceMemoryBase dst2_mem = dst2->device_memory();
  se::DeviceMemoryBase dst3_mem = dst3->device_memory();
  se::DeviceMemoryBase dst4_mem = dst4->device_memory();
  se::DeviceMemoryBase dst5_mem = dst5->device_memory();
  se::DeviceMemoryBase dst6_mem = dst6->device_memory();

  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst0_mem, src3.device_memory(), 8 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst1_mem, src0.device_memory(), 128 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst2_mem, src1.device_memory(), 256 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst3_mem, src2.device_memory(), 1024 * sizeof(float)));
  TF_RETURN_IF_ERROR(stream->MemcpyD2D(&dst4_mem, src4.device_memory(),
                                       4 * 8 * sizeof(float)));
  TF_RETURN_IF_ERROR(stream->MemcpyD2D(&dst5_mem, src7.device_memory(),
                                       3 * 128 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst6_mem, src6.device_memory(), 64 * sizeof(float)));
  stream_executor::DeviceMemoryBase slice =
      dst6_mem.GetByteSlice(64 * sizeof(float), 32 * sizeof(float));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&slice, src6.device_memory(), 32 * sizeof(float)));
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
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(true);
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
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(true);
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
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(true);
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
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(true);
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

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_opt, GetModuleConfigWithDeterministicOps(),
      GetModuleConfigWithDeterministicOps(), error_spec,
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

  %dynamic-slice-fusion {
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
    ROOT %dynamic-slice-fusion.6 = (f32[100,100]{1,0}, s8[80000]{0}) fusion(%get-tuple-element.97, %get-tuple-element.240, %c0_s32),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
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

  %dynamic-slice-fusion {
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
    ROOT %dynamic-slice-fusion.6 = f16[8,8]{1,0} fusion(%p1, %p0, %c0_s32, %c1_s32),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
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

  %dynamic-slice-fusion {
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
    ROOT %dynamic-slice-fusion.6 = (f32[100,100]{1,0}, s8[80000]{0}) fusion(%get-tuple-element.97, %c0_s32),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
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

  %dynamic-slice-fusion {
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
    ROOT %dynamic-slice-fusion.6 = (f32[100,100]{1,0}, s8[120000]{0}) fusion(%get-tuple-element.287, %slice.34, %concatenate.12, %c0_s32, %c20_s32),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
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

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_opt, GetModuleConfigWithDeterministicOps(),
      GetModuleConfigWithDeterministicOps(), error_spec,
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

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_opt, GetModuleConfigWithDeterministicOps(),
      GetModuleConfigWithDeterministicOps(), error_spec,
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

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_opt, GetModuleConfigWithDeterministicOps(),
      GetModuleConfigWithDeterministicOps(), error_spec,
      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, DynamicCustomCallSimple) {
  XlaBuilder b(TestName());
  CustomCall(
      &b, "__xla_test$$memcpy",
      /*operands=*/
      {DynamicSlice(Parameter(&b, 0, ShapeUtil::MakeShape(S32, {4, 128}), "p0"),
                    {ConstantR0(&b, 2), ConstantR0(&b, 0)}, {2, 128})},
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
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
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
                        {ConstantR0(&b, 20), ConstantR0(&b, 0)}, {3, 128}),
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
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(true);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_opt, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  DynamicSliceFusionRewriter pass(PLATFORM);
  TF_ASSERT_OK_AND_ASSIGN(auto changed, this->RunHloPass(&pass, hlo_opt.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(*RunFileCheck(hlo_opt->ToString(), R"(
    // CHECK: %dynamic-slice-fusion{{.+}} {
    // CHECK:   {{.+}} = {{.+}} slice
    // CHECK:   {{.+}} = {{.+}} dynamic-slice
    // CHECK:   {{.+}} = {{.+}} custom-call
    // CHECK: ENTRY {{.+}} {
    // CHECK-NOT: {{.+}} = {{.+}} slice
    // CHECK-NOT: {{.+}} = {{.+}} dynamic-slice
  )"));

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

  se::DeviceMemoryBase dst0_mem = dst0->device_memory();
  se::DeviceMemoryBase dst1_mem = dst1->device_memory();
  se::DeviceMemoryBase dst2_mem = dst2->device_memory();
  se::DeviceMemoryBase dst3_mem = dst3->device_memory();
  se::DeviceMemoryBase dst4_mem = dst4->device_memory();
  se::DeviceMemoryBase dst5_mem = dst5->device_memory();
  se::DeviceMemoryBase dst6_mem = dst6->device_memory();

  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst0_mem, src3.device_memory(), 8 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst1_mem, src0.device_memory(), 128 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst2_mem, src1.device_memory(), 256 * sizeof(float)));
  TF_RETURN_IF_ERROR(
      stream->MemcpyD2D(&dst3_mem, src2.device_memory(), 1024 * sizeof(float)));
  TF_RETURN_IF_ERROR(stream->MemcpyD2D(&dst4_mem, src4.device_memory(),
                                       4 * 8 * sizeof(float)));
  TF_RETURN_IF_ERROR(stream->MemcpyD2D(&dst5_mem, src6.device_memory(),
                                       5 * 128 * sizeof(float)));
  TF_RETURN_IF_ERROR(stream->MemcpyD2D(&dst6_mem, src5.device_memory(),
                                       3 * 128 * sizeof(float)));
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
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(true);
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
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(true);
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

TEST_F(DynamicSliceFusionTest, ReduceScatterDUSConstant) {
  // DUS offset is a constant
  const char* hlo_ref = R"(
  HloModule test, replica_count=2

  add.clone {
    x.1 = f16[] parameter(0)
    y.1 = f16[] parameter(1)
    ROOT add.462 = f16[] add(x.1, y.1)
  }

  ENTRY %main.9 {
    param_0 = f16[128,128]{1,0} parameter(0)
    param_1 = f16[128,128]{1,0} parameter(1)
    constant_20 = u32[] constant(20)
    constant_0 = u32[] constant(0)
    reduce-scatter = f16[64,128]{1,0} reduce-scatter(param_0), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add.clone
    ROOT dynamic-update-slice = f16[128,128]{1,0} dynamic-update-slice(param_1, reduce-scatter, constant_20, constant_0)
  })";

  const char* hlo_opt = R"(
  HloModule test, replica_count=2

  %add {
    %param_0 = f16[] parameter(0)
    %param_1 = f16[] parameter(1)
    ROOT %add.1 = f16[] add(%param_0, %param_1)
  }

  %dynamic-slice-fusion {
    %p1 = f16[128,128]{1,0} parameter(1)
    %p0 = f16[128,128]{1,0} parameter(0)
    %reduce-scatter.1 = f16[64,128]{1,0} reduce-scatter(%p0), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=%add
    %p2 = u32[] parameter(2)
    %p3 = u32[] parameter(3)
    ROOT %loop_dynamic_update_slice_fusion.1 = f16[128,128]{1,0} dynamic-update-slice(%p1, %reduce-scatter.1, %p2, %p3)
  }

  ENTRY %main.9 {
    %param_0.1 = f16[128,128]{1,0} parameter(0)
    %param_1.1 = f16[128,128]{1,0} parameter(1)
    %constant_20 = u32[] constant(20)
    %constant_0 = u32[] constant(0)
    ROOT %dynamic-slice-fusion = f16[128,128]{1,0} fusion(%param_0.1, %param_1.1, %constant_20, %constant_0), kind=kCustom, calls=%dynamic-slice-fusion, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}},"force_earliest_schedule":false}
  })";

  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};
  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(hlo_ref, hlo_opt, true, true,
                                                error_spec));
}

TEST_F(DynamicSliceFusionTest, ReduceScatterDUSParameterOffset) {
  // DUS offset is a parameter. This enforces a d2h copy.
  const char* hlo_ref = R"(
  HloModule test, replica_count=2

  add.clone {
    x.1 = f16[] parameter(0)
    y.1 = f16[] parameter(1)
    ROOT add.462 = f16[] add(x.1, y.1)
  }

  ENTRY %main.9 {
    param_0 = f16[128,128]{1,0} parameter(0)
    param_1 = f16[128,128]{1,0} parameter(1)
    param_2 = u32[] parameter(2)
    constant_0 = u32[] constant(0)
    reduce-scatter = f16[64,128]{1,0} reduce-scatter(param_0), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add.clone
    ROOT dynamic-update-slice = f16[128,128]{1,0} dynamic-update-slice(param_1, reduce-scatter, param_2, constant_0)
  })";

  const char* hlo_opt = R"(
  HloModule test, replica_count=2

  %add {
    %param_0 = f16[] parameter(0)
    %param_1 = f16[] parameter(1)
    ROOT %add.1 = f16[] add(f16[] %param_0, f16[] %param_1)
  }

  %dynamic-slice-fusion {
    %p1 = f16[128,128]{1,0} parameter(1)
    %p0 = f16[128,128]{1,0} parameter(0)
    %reduce-scatter.1 = f16[64,128]{1,0} reduce-scatter(%p0), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=%add
    %p2 = u32[] parameter(2)
    %p3 = u32[] parameter(3)
    ROOT %loop_dynamic_update_slice_fusion.1 = f16[128,128]{1,0} dynamic-update-slice(%p1, %reduce-scatter.1, %p2, %p3)
  }

  ENTRY %main.9 {
    %param_0 = f16[128,128]{1,0} parameter(0)
    %param_1 = f16[128,128]{1,0} parameter(1)
    %param_2 = u32[] parameter(2)
    %constant_0 = u32[] constant(0)
    ROOT %dynamic-slice-fusion = f16[128,128]{1,0} fusion(%param_0, %param_1, %param_2, %constant_0), kind=kCustom, calls=%dynamic-slice-fusion, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}},"force_earliest_schedule":false}
  })";

  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};
  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(hlo_ref, hlo_opt, true, true,
                                                error_spec));
}

TEST_F(DynamicSliceFusionTest, ReduceScatterDUSLoopIterationOffset) {
  const char* hlo_ref = R"(
  HloModule jit_scan, replica_count=2

  %add {
    %param_0 = f32[] parameter(0)
    %param_1 = f32[] parameter(1)
    ROOT %add.1 = f32[] add(%param_0, %param_1)
  }

  %region_0.14 {
    %arg_tuple.15 = (s32[], f32[128,128]{1,0}, f32[128,128,128]{2,1,0}, f32[128,128,128]{2,1,0}, f32[128,128]{1,0}) parameter(0)
    %get-tuple-element.16 = s32[] get-tuple-element(%arg_tuple.15), index=0
    %constant.21 = s32[] constant(1)
    %add.37 = s32[] add(%get-tuple-element.16, %constant.21)
    %get-tuple-element.20 = f32[128,128]{1,0} get-tuple-element(%arg_tuple.15), index=4
    %get-tuple-element.18 = f32[128,128,128]{2,1,0} get-tuple-element(%arg_tuple.15), index=2
    %reduce-scatter.1 = f32[64,128]{1,0} reduce-scatter(%get-tuple-element.20), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=%add
    %reshape.32 = f32[1,64,128]{2,1,0} reshape(%reduce-scatter.1)
    %constant.23 = s32[] constant(0)
    %compare.33 = pred[] compare(%get-tuple-element.16, %constant.23), direction=LT
    %constant.22 = s32[] constant(128)
    %add.34 = s32[] add(%get-tuple-element.16, %constant.22)
    %select.35 = s32[] select(%compare.33, %add.34, %get-tuple-element.16)
    %dynamic-update-slice.36 = f32[128,128,128]{2,1,0} dynamic-update-slice(%get-tuple-element.18, %reshape.32, %select.35, %constant.23, %constant.23)
    %get-tuple-element.19 = f32[128,128,128]{2,1,0} get-tuple-element(%arg_tuple.15), index=3
    ROOT %tuple.38 = tuple(%add.37, %get-tuple-element.20, %dynamic-update-slice.36, %get-tuple-element.19, %get-tuple-element.20)
  }

  %region_1.39 {
    %arg_tuple.40 = (s32[], f32[128,128]{1,0}, f32[128,128,128]{2,1,0}, f32[128,128,128]{2,1,0}, f32[128,128]{1,0}) parameter(0)
    %get-tuple-element.41 = s32[] get-tuple-element(%arg_tuple.40), index=0
    %constant.46 = s32[] constant(128)
    ROOT %compare.47 = pred[] compare(%get-tuple-element.41, %constant.46), direction=LT
  }

  ENTRY %main.55 {
    %constant.4 = s32[] constant(0)
    %Arg_1.2 = f32[128,128]{1,0} parameter(1)
    %constant.5 = f32[] constant(0)
    %broadcast.6 = f32[128,128,128]{2,1,0} broadcast(%constant.5), dimensions={}
    %Arg_2.3 = f32[128,128,128]{2,1,0} parameter(2)
    %Arg_0.1 = f32[128,128]{1,0} parameter(0)
    %tuple.7 = tuple(%constant.4, %Arg_1.2, %broadcast.6, %Arg_2.3, %Arg_0.1)
    %while.48 = while(%tuple.7), condition=%region_1.39, body=%region_0.14
    %get-tuple-element.50 = f32[128,128]{1,0} get-tuple-element(%while.48), index=1
    %get-tuple-element.51 = f32[128,128,128]{2,1,0} get-tuple-element(%while.48), index=2
    ROOT %tuple.54 = tuple(%get-tuple-element.50, %get-tuple-element.51)
  })";
  DebugOptions debugoptions = GetDebugOptionsForTest();

  HloModuleConfig ref_config;
  debugoptions.set_xla_gpu_enable_dynamic_slice_fusion(false);
  ref_config.set_debug_options(debugoptions);
  TF_ASSERT_OK_AND_ASSIGN(auto ref_module,
                          ParseAndReturnVerifiedModule(hlo_ref, ref_config));
  TF_ASSERT_OK_AND_ASSIGN(auto ref_module_opt,
                          GetOptimizedModule(std::move(ref_module)));

  HloModuleConfig opt_config;
  debugoptions.set_xla_gpu_enable_dynamic_slice_fusion(true);
  opt_config.set_debug_options(debugoptions);
  TF_ASSERT_OK_AND_ASSIGN(auto module_with_adddress_computation_flag,
                          ParseAndReturnVerifiedModule(hlo_ref, opt_config));
  TF_ASSERT_OK_AND_ASSIGN(
      auto module_with_adddress_computation,
      GetOptimizedModule(std::move(module_with_adddress_computation_flag)));

  std::vector<HloComputation*> address_computations_opt =
      GetDynamicSliceFusions(*module_with_adddress_computation);
  std::vector<HloComputation*> address_computations_ref =
      GetDynamicSliceFusions(*ref_module_opt);
  EXPECT_EQ(address_computations_ref.size(), 0);
  ASSERT_EQ(address_computations_opt.size(), 1);

  // Check that reduce scatter happens in the fusion in optimized module and not
  // outside the fusion.
  EXPECT_TRUE(*RunFileCheck(address_computations_opt[0]->ToString(), R"(
    // CHECK: {{.+}} = {{.*}}reduce-scatter({{.+}})
    // CHECK: {{.+}} = {{.*}}dynamic-update-slice({{.+}})
  )"));
  EXPECT_TRUE(*RunFileCheck(
      address_computations_opt[0]->FusionInstruction()->parent()->ToString(),
      "// CHECK-NOT: {{.+}} = {{.*}}reduce-scatter"));

  ErrorSpec error{/*aabs=*/1e-3, /*arel=*/1e-3};
  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(
      std::move(ref_module_opt), std::move(module_with_adddress_computation),
      false, true, error));
}

TEST_F(DynamicSliceFusionTest, OffsetArrayTestS32) {
  // When the offset is s32, then this checks that the offset values can be out
  // of bounds on both directions but they should be clamped during execution.
  const char* hlo = R"(
    HloModule test_module, replica_count=2

    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }

    dynamic-slice-fusion {
      dest = s32[16,4] parameter(0)
      src = s32[8,4] parameter(1)
      loop_iter = s32[] parameter(2)
      offset_values = s32[4] constant({-4,4,12,20})
      offset_as_array = s32[1] dynamic-slice(offset_values, loop_iter), dynamic_slice_sizes={1}
      offset = s32[] reshape(offset_as_array)
      rs = s32[4,4] reduce-scatter(src), channel_id=0, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
      zero = s32[] parameter(3)
      ROOT dus = s32[16,4] dynamic-update-slice(dest, rs, offset, zero)
    }

    Body {
      param = (s32[], s32[16, 4], s32[8, 4], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      dest = s32[16,4] get-tuple-element(param), index=1
      src = s32[8,4] get-tuple-element(param), index=2
      loop_iter = s32[] get-tuple-element(param), index=3
      zero = s32[] constant(0)
      fusion = s32[16,4] fusion(dest, src, loop_iter, zero), kind=kCustom, calls=dynamic-slice-fusion, backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      loop_iter_plus_one = s32[] add(loop_iter, one)
      ROOT tuple = tuple(i_plus_one, fusion, src, loop_iter)
    }

    Cond {
      param = (s32[], s32[16,4], s32[8,4], s32[]) parameter(0)
      loop_iter = s32[] get-tuple-element(param), index=0
      four = s32[] constant(4)
      ROOT compare = pred[] compare(loop_iter, four), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      dest = s32[16,4] parameter(0)
      src = s32[8,4] parameter(1)
      tuple = tuple(zero, dest, src, zero)
      ROOT while = (s32[], s32[16,4], s32[8,4], s32[]) while(tuple), body=Body, condition=Cond
    }
  )";

  const char* normal = R"(
    HloModule test_module, replica_count=2

    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }

    Body {
      param = (s32[], s32[16, 4], s32[8, 4], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      dest = s32[16,4] get-tuple-element(param), index=1
      src = s32[8,4] get-tuple-element(param), index=2
      loop_iter = s32[] get-tuple-element(param), index=3
      eight = s32[] constant(8)
      four = s32[] constant(4)
      mul = s32[] multiply(eight, i)
      offset = s32[] subtract(mul, four)
      zero = s32[] constant(0)
      rs = s32[4,4] reduce-scatter(src), channel_id=0, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
      fusion = s32[16,4] dynamic-update-slice(dest, rs, offset, zero)
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      loop_iter_plus_one = s32[] add(loop_iter, one)
      ROOT tuple = tuple(i_plus_one, fusion, src, loop_iter)
    }

    Cond {
      param = (s32[], s32[16,4], s32[8,4], s32[]) parameter(0)
      loop_iter = s32[] get-tuple-element(param), index=0
      four = s32[] constant(4)
      ROOT compare = pred[] compare(loop_iter, four), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      dest = s32[16,4] parameter(0)
      src = s32[8,4] parameter(1)
      tuple = tuple(zero, dest, src, zero)
      ROOT while = (s32[], s32[16,4], s32[8,4], s32[]) while(tuple), body=Body, condition=Cond
    }
  )";
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};
  EXPECT_TRUE(
      RunAndCompareTwoModulesReplicated(hlo, normal, true, true, error_spec));
}

TEST_F(DynamicSliceFusionTest, OffsetArrayTestS64) {
  // This test makes sure that the offsets are not parsed as `int32_t` all the
  // time. If the offset is INT64_MAX, and it is parsed as int32_t, it will be
  // clamped to the zero (parsed as -1) instead of being clamped to the upper
  // bound. The offsets in the example here should be clamped to {0,4,8,12} but
  // if they are parsed as int32_t, then they will be clamped to {12, 4, 8, 0}.
  const char* hlo = R"(
    HloModule test_module, replica_count=2

    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }

    dynamic-slice-fusion {
      dest = s32[16,4] parameter(0)
      src = s32[8,4] parameter(1)
      loop_iter = s32[] parameter(2)
      offset_values = s64[4] constant({-2147483649,4,8,9223372036854775807})
      offset_as_array = s64[1] dynamic-slice(offset_values, loop_iter), dynamic_slice_sizes={1}
      offset = s64[] reshape(offset_as_array)
      rs = s32[4,4] reduce-scatter(src), channel_id=0, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
      zero = s64[] parameter(3)
      ROOT dus = s32[16,4] dynamic-update-slice(dest, rs, offset, zero)
    }

    Body {
      param = (s32[], s32[16, 4], s32[8, 4], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      dest = s32[16,4] get-tuple-element(param), index=1
      src = s32[8,4] get-tuple-element(param), index=2
      loop_iter = s32[] get-tuple-element(param), index=3
      zero = s64[] constant(0)
      fusion = s32[16,4] fusion(dest, src, loop_iter, zero), kind=kCustom, calls=dynamic-slice-fusion, backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      loop_iter_plus_one = s32[] add(loop_iter, one)
      ROOT tuple = tuple(i_plus_one, fusion, src, loop_iter)
    }

    Cond {
      param = (s32[], s32[16,4], s32[8,4], s32[]) parameter(0)
      loop_iter = s32[] get-tuple-element(param), index=0
      four = s32[] constant(4)
      ROOT compare = pred[] compare(loop_iter, four), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      dest = s32[16,4] parameter(0)
      src = s32[8,4] parameter(1)
      tuple = tuple(zero, dest, src, zero)
      ROOT while = (s32[], s32[16,4], s32[8,4], s32[]) while(tuple), body=Body, condition=Cond
    }
  )";

  const char* normal = R"(
    HloModule test_module, replica_count=2

    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }

    Body {
      param = (s32[], s32[16, 4], s32[8, 4], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      dest = s32[16,4] get-tuple-element(param), index=1
      src = s32[8,4] get-tuple-element(param), index=2
      loop_iter = s32[] get-tuple-element(param), index=3
      offset_values = s64[4] constant({-2147483649,4,8,9223372036854775807})
      offset_as_array = s64[1] dynamic-slice(offset_values, i), dynamic_slice_sizes={1}
      offset = s64[] reshape(offset_as_array)
      zero = s64[] constant(0)
      rs = s32[4,4] reduce-scatter(src), channel_id=0, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
      fusion = s32[16,4] dynamic-update-slice(dest, rs, offset, zero)
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      loop_iter_plus_one = s32[] add(loop_iter, one)
      ROOT tuple = tuple(i_plus_one, fusion, src, loop_iter)
    }

    Cond {
      param = (s32[], s32[16,4], s32[8,4], s32[]) parameter(0)
      loop_iter = s32[] get-tuple-element(param), index=0
      four = s32[] constant(4)
      ROOT compare = pred[] compare(loop_iter, four), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      dest = s32[16,4] parameter(0)
      src = s32[8,4] parameter(1)
      tuple = tuple(zero, dest, src, zero)
      ROOT while = (s32[], s32[16,4], s32[8,4], s32[]) while(tuple), body=Body, condition=Cond
    }
  )";
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};
  EXPECT_TRUE(
      RunAndCompareTwoModulesReplicated(hlo, normal, true, true, error_spec));
}

// Same as above for uint32_t
TEST_F(DynamicSliceFusionTest, OffsetArrayTestU32) {
  const char* hlo = R"(
    HloModule test_module, replica_count=2

    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }

    dynamic-slice-fusion {
      dest = s32[16,4] parameter(0)
      src = s32[8,4] parameter(1)
      loop_iter = s32[] parameter(2)
      offset_values = u32[4] constant({0,4,8,4294967295})
      offset_as_array = u32[1] dynamic-slice(offset_values, loop_iter), dynamic_slice_sizes={1}
      offset = u32[] reshape(offset_as_array)
      rs = s32[4,4] reduce-scatter(src), channel_id=0, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
      zero = u32[] parameter(3)
      ROOT dus = s32[16,4] dynamic-update-slice(dest, rs, offset, zero)
    }

    Body {
      param = (s32[], s32[16, 4], s32[8, 4], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      dest = s32[16,4] get-tuple-element(param), index=1
      src = s32[8,4] get-tuple-element(param), index=2
      loop_iter = s32[] get-tuple-element(param), index=3
      zero = u32[] constant(0)
      fusion = s32[16,4] fusion(dest, src, loop_iter, zero), kind=kCustom, calls=dynamic-slice-fusion, backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      loop_iter_plus_one = s32[] add(loop_iter, one)
      ROOT tuple = tuple(i_plus_one, fusion, src, loop_iter)
    }

    Cond {
      param = (s32[], s32[16,4], s32[8,4], s32[]) parameter(0)
      loop_iter = s32[] get-tuple-element(param), index=0
      four = s32[] constant(4)
      ROOT compare = pred[] compare(loop_iter, four), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      dest = s32[16,4] parameter(0)
      src = s32[8,4] parameter(1)
      tuple = tuple(zero, dest, src, zero)
      ROOT while = (s32[], s32[16,4], s32[8,4], s32[]) while(tuple), body=Body, condition=Cond
    }
  )";

  const char* normal = R"(
    HloModule test_module, replica_count=2

    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }

    Body {
      param = (s32[], s32[16, 4], s32[8, 4], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      dest = s32[16,4] get-tuple-element(param), index=1
      src = s32[8,4] get-tuple-element(param), index=2
      loop_iter = s32[] get-tuple-element(param), index=3
      offset_values = u32[4] constant({0,4,8,4294967295})
      offset_as_array = u32[1] dynamic-slice(offset_values, i), dynamic_slice_sizes={1}
      offset = u32[] reshape(offset_as_array)
      zero = u32[] constant(0)
      rs = s32[4,4] reduce-scatter(src), channel_id=0, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
      fusion = s32[16,4] dynamic-update-slice(dest, rs, offset, zero)
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      loop_iter_plus_one = s32[] add(loop_iter, one)
      ROOT tuple = tuple(i_plus_one, fusion, src, loop_iter)
    }

    Cond {
      param = (s32[], s32[16,4], s32[8,4], s32[]) parameter(0)
      loop_iter = s32[] get-tuple-element(param), index=0
      four = s32[] constant(4)
      ROOT compare = pred[] compare(loop_iter, four), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      dest = s32[16,4] parameter(0)
      src = s32[8,4] parameter(1)
      tuple = tuple(zero, dest, src, zero)
      ROOT while = (s32[], s32[16,4], s32[8,4], s32[]) while(tuple), body=Body, condition=Cond
    }
  )";

  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};
  EXPECT_TRUE(
      RunAndCompareTwoModulesReplicated(hlo, normal, true, true, error_spec));
}

// Same as above for uint64_t. It is expected to produce an error if the offset
// is outside the range of int64_t.
TEST_F(DynamicSliceFusionTest, OffsetArrayTestU64) {
  const char* hlo = R"(
    HloModule test_module, replica_count=2

    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }

    dynamic-slice-fusion {
      dest = s32[16,4] parameter(0)
      src = s32[8,4] parameter(1)
      loop_iter = s32[] parameter(2)
      offset_values = u64[4] constant({0,4,8,18446744073709551615})
      offset_as_array = u64[1] dynamic-slice(offset_values, loop_iter), dynamic_slice_sizes={1}
      offset = u64[] reshape(offset_as_array)
      rs = s32[4,4] reduce-scatter(src), channel_id=0, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
      zero = u64[] parameter(3)
      ROOT dus = s32[16,4] dynamic-update-slice(dest, rs, offset, zero)
    }

    Body {
      param = (s32[], s32[16, 4], s32[8, 4], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      dest = s32[16,4] get-tuple-element(param), index=1
      src = s32[8,4] get-tuple-element(param), index=2
      loop_iter = s32[] get-tuple-element(param), index=3
      zero = u64[] constant(0)
      fusion = s32[16,4] fusion(dest, src, loop_iter, zero), kind=kCustom, calls=dynamic-slice-fusion, backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      loop_iter_plus_one = s32[] add(loop_iter, one)
      ROOT tuple = tuple(i_plus_one, fusion, src, loop_iter)
    }

    Cond {
      param = (s32[], s32[16,4], s32[8,4], s32[]) parameter(0)
      loop_iter = s32[] get-tuple-element(param), index=0
      four = s32[] constant(4)
      ROOT compare = pred[] compare(loop_iter, four), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      dest = s32[16,4] parameter(0)
      src = s32[8,4] parameter(1)
      tuple = tuple(zero, dest, src, zero)
      ROOT while = (s32[], s32[16,4], s32[8,4], s32[]) while(tuple), body=Body, condition=Cond
    }
  )";

  const char* normal = R"(
    HloModule test_module, replica_count=2

    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }

    Body {
      param = (s32[], s32[16, 4], s32[8, 4], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      dest = s32[16,4] get-tuple-element(param), index=1
      src = s32[8,4] get-tuple-element(param), index=2
      loop_iter = s32[] get-tuple-element(param), index=3
      offset_values = u64[4] constant({0,4,8,18446744073709551615})
      offset_as_array = u64[1] dynamic-slice(offset_values, i), dynamic_slice_sizes={1}
      offset = u64[] reshape(offset_as_array)
      zero = u64[] constant(0)
      rs = s32[4,4] reduce-scatter(src), channel_id=0, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
      fusion = s32[16,4] dynamic-update-slice(dest, rs, offset, zero)
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      loop_iter_plus_one = s32[] add(loop_iter, one)
      ROOT tuple = tuple(i_plus_one, fusion, src, loop_iter)
    }

    Cond {
      param = (s32[], s32[16,4], s32[8,4], s32[]) parameter(0)
      loop_iter = s32[] get-tuple-element(param), index=0
      four = s32[] constant(4)
      ROOT compare = pred[] compare(loop_iter, four), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      dest = s32[16,4] parameter(0)
      src = s32[8,4] parameter(1)
      tuple = tuple(zero, dest, src, zero)
      ROOT while = (s32[], s32[16,4], s32[8,4], s32[]) while(tuple), body=Body, condition=Cond
    }
  )";

  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(auto status = RunAndCompareTwoModulesReplicated(
                   hlo, normal, true, true, error_spec),
               ".*");
}

TEST_F(DynamicSliceFusionTest, ReduceScatterDegenerateCollective) {
  const char* hlo_ref = R"(
  HloModule test, replica_count=2

  add.clone {
    x.1 = f16[] parameter(0)
    y.1 = f16[] parameter(1)
    ROOT add.462 = f16[] add(x.1, y.1)
  }

  ENTRY %main.9 {
    param_0 = f16[128,128]{1,0} parameter(0)
    param_1 = f16[128,128]{1,0} parameter(1)
    constant_20 = u32[] constant(20)
    constant_0 = u32[] constant(0)
    dynamic-slice = f16[64,128]{1,0} dynamic-slice(param_0, constant_0, constant_0), dynamic_slice_sizes={64,128}
    reduce-scatter = f16[64,128]{1,0} reduce-scatter(dynamic-slice), channel_id=64, replica_groups={{0},{1}}, use_global_device_ids=true, dimensions={0}, to_apply=add.clone
    ROOT dynamic-update-slice = f16[128,128]{1,0} dynamic-update-slice(param_1, reduce-scatter, constant_20, constant_0)
  })";

  const char* hlo_opt = R"(
  HloModule test, replica_count=2

  %add {
    %param_0 = f16[] parameter(0)
    %param_1 = f16[] parameter(1)
    ROOT %add.1 = f16[] add(%param_0, %param_1)
  }

  %address-computation {
    %p0 = f16[128,128]{1,0} parameter(0)
    %p1 = f16[128,128]{1,0} parameter(1)
    %p2 = u32[] parameter(2)
    %p3 = u32[] parameter(3)
    %dynamic-slice = f16[64,128]{1,0} dynamic-slice(%p0, %p3, %p3), dynamic_slice_sizes={64,128}
    %reduce-scatter.1 = f16[64,128]{1,0} reduce-scatter(%dynamic-slice), channel_id=64, replica_groups={{0},{1}}, use_global_device_ids=true, dimensions={0}, to_apply=%add
    ROOT %loop_dynamic_update_slice_fusion.1 = f16[128,128]{1,0} dynamic-update-slice(%p1, %reduce-scatter.1, %p2, %p3)
  }

  ENTRY %main.9 {
    %param_0.1 = f16[128,128]{1,0} parameter(0)
    %param_1.1 = f16[128,128]{1,0} parameter(1)
    %constant_20 = u32[] constant(20)
    %constant_0 = u32[] constant(0)
    ROOT %address_computation = f16[128,128]{1,0} fusion(%param_0.1, %param_1.1, %constant_20, %constant_0), kind=kCustom, calls=%address-computation, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}},"force_earliest_schedule":false}
  })";

  // Dynamic slice fusion is turned on by default. So, we need to turn that off
  // while parsing.
  HloModuleConfig ref_config;
  DebugOptions debug_options;
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  ref_config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto ref_module,
                          ParseAndReturnVerifiedModule(hlo_ref, ref_config));
  TF_ASSERT_OK_AND_ASSIGN(auto ref_module_opt,
                          GetOptimizedModule(std::move(ref_module)));

  TF_ASSERT_OK_AND_ASSIGN(auto module_with_fusion,
                          ParseAndReturnVerifiedModule(hlo_opt, ref_config));
  TF_ASSERT_OK_AND_ASSIGN(auto module_with_fusion_opt,
                          GetOptimizedModule(std::move(module_with_fusion)));

  // Check that the thunk is a d2d copy thunk because the collective is
  // degenerate.
  auto module_with_fusion_opt_clone = module_with_fusion_opt->Clone();
  TF_ASSERT_OK_AND_ASSIGN(
      auto exec,
      CreateExecutable(std::move(module_with_fusion_opt_clone), false));
  GpuExecutable* gpu_exec = dynamic_cast<GpuExecutable*>(exec.get());
  auto& child_thunk = gpu_exec->GetThunk().thunks()[1];
  ASSERT_EQ(child_thunk->kind(), Thunk::kDynamicSlice);
  auto* ds_thunk = dynamic_cast<const DynamicSliceThunk*>(child_thunk.get());
  ASSERT_EQ(ds_thunk->embedded_thunk()->kind(), Thunk::kSequential);
  auto* embedded_thunk =
      dynamic_cast<const SequentialThunk*>(ds_thunk->embedded_thunk());
  ASSERT_EQ(embedded_thunk->thunks().size(), 1ul);
  ASSERT_EQ(embedded_thunk->thunks()[0]->kind(), Thunk::kCopy);
  ErrorSpec error{/*aabs=*/1e-3, /*arel=*/1e-3};

  // Comparing the outputs.
  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(
      std::move(ref_module_opt), std::move(module_with_fusion_opt),
      /*run_hlo_passes=*/false, /*use_threads=*/true, error));
}

TEST_F(DynamicSliceFusionTest, ReduceScatterSlice) {
  const char* hlo_ref = R"(
  HloModule jit_slice, replica_count=2

  add {
    a = s32[] parameter(0)
    b = s32[] parameter(1)
    ROOT add = add(a,b)
  }

  ENTRY %main.9 {
    %p0 = s32[2,8,32]{2,1,0} parameter(0)
    %slice = s32[1,8,32]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:32]}
    %bc1 = s32[8,32]{1,0} reshape(%slice)
    ROOT rs = s32[4,32] reduce-scatter(bc1), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
  }
  )";

  HloModuleConfig config;
  DebugOptions options;
  options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  options.clear_xla_gpu_enable_command_buffer();
  config.set_debug_options(options);
  TF_ASSERT_OK_AND_ASSIGN(auto module_ref,
                          ParseAndReturnVerifiedModule(hlo_ref, config));

  options.set_xla_gpu_enable_dynamic_slice_fusion(true);
  options.clear_xla_gpu_enable_command_buffer();
  config.set_debug_options(options);
  TF_ASSERT_OK_AND_ASSIGN(auto module_new,
                          ParseAndReturnVerifiedModule(hlo_ref, config));

  TF_ASSERT_OK_AND_ASSIGN(auto module_ref_opt,
                          GetOptimizedModule(std::move(module_ref)));
  TF_ASSERT_OK_AND_ASSIGN(auto module_new_opt,
                          GetOptimizedModule(std::move(module_new)));

  ASSERT_TRUE(GetDynamicSliceFusions(*module_ref_opt).empty());
  ASSERT_FALSE(GetDynamicSliceFusions(*module_new_opt).empty());

  auto module_new_opt_clone = module_new_opt->Clone();
  TF_ASSERT_OK_AND_ASSIGN(
      auto exec, CreateExecutable(std::move(module_new_opt_clone), false));
  GpuExecutable* gpu_exec = dynamic_cast<GpuExecutable*>(exec.get());
  ASSERT_EQ(gpu_exec->GetThunk().thunks().size(), 2ul);
  auto& rs_start_thunk = gpu_exec->GetThunk().thunks()[0];
  auto& rs_done_thunk = gpu_exec->GetThunk().thunks()[1];
  ASSERT_EQ(rs_start_thunk->kind(), Thunk::kNcclReduceScatterStart);
  ASSERT_EQ(rs_done_thunk->kind(), Thunk::kNcclReduceScatterDone);

  ErrorSpec error{/*aabs=*/1e-3, /*arel=*/1e-3};
  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(std::move(module_ref_opt),
                                                std::move(module_new_opt),
                                                false, true, error));
}

TEST_F(DynamicSliceFusionTest, ReduceScatterDynamicSlice) {
  const char* hlo_ref = R"(
  HloModule jit_slice, replica_count=2

  add {
    a = s32[] parameter(0)
    b = s32[] parameter(1)
    ROOT add = add(a,b)
  }

  ENTRY %main.9 {
    p0 = s32[2,8,32]{2,1,0} parameter(0)
    c0 = s32[] constant(0)
    c1 = s32[] constant(1)
    slice = s32[1,8,32]{2,1,0} dynamic-slice(p0, c1, c0, c0), dynamic_slice_sizes={1,8,32}
    bc1 = s32[8,32]{1,0} reshape(slice)
    ROOT rs = s32[4,32] reduce-scatter(bc1), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
  })";

  HloModuleConfig config;
  DebugOptions options;
  options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  options.clear_xla_gpu_enable_command_buffer();
  config.set_debug_options(options);
  TF_ASSERT_OK_AND_ASSIGN(auto module_ref,
                          ParseAndReturnVerifiedModule(hlo_ref, config));

  options.set_xla_gpu_enable_dynamic_slice_fusion(true);
  options.clear_xla_gpu_enable_command_buffer();
  config.set_debug_options(options);
  TF_ASSERT_OK_AND_ASSIGN(auto module_new,
                          ParseAndReturnVerifiedModule(hlo_ref, config));

  TF_ASSERT_OK_AND_ASSIGN(auto module_ref_opt,
                          GetOptimizedModule(std::move(module_ref)));
  TF_ASSERT_OK_AND_ASSIGN(auto module_new_opt,
                          GetOptimizedModule(std::move(module_new)));

  ASSERT_TRUE(GetDynamicSliceFusions(*module_ref_opt).empty());
  ASSERT_FALSE(GetDynamicSliceFusions(*module_new_opt).empty());

  ErrorSpec error{/*aabs=*/1e-3, /*arel=*/1e-3};
  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(std::move(module_ref_opt),
                                                std::move(module_new_opt),
                                                false, true, error));
}

TEST_F(DynamicSliceFusionTest, ReduceScatterDegenerateSlice) {
  const char* hlo_ref = R"(
    HloModule test_module, replica_count=2

    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }

    ENTRY main {
      p0 = s32[2,4,8] parameter(0)
      slice = s32[1,4,8] slice(p0), slice={[1:2], [0:4], [0:8]}
      bc = s32[4,8] reshape(slice)
      ROOT rs = s32[4,8] reduce-scatter(bc), channel_id=64, replica_groups={{0},{1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
    }
  )";
  HloModuleConfig config;
  DebugOptions options;
  options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  options.clear_xla_gpu_enable_command_buffer();
  config.set_debug_options(options);
  TF_ASSERT_OK_AND_ASSIGN(auto module_ref,
                          ParseAndReturnVerifiedModule(hlo_ref, config));

  options.set_xla_gpu_enable_dynamic_slice_fusion(true);
  options.clear_xla_gpu_enable_command_buffer();
  config.set_debug_options(options);
  TF_ASSERT_OK_AND_ASSIGN(auto module_new,
                          ParseAndReturnVerifiedModule(hlo_ref, config));

  TF_ASSERT_OK_AND_ASSIGN(auto module_ref_opt,
                          GetOptimizedModule(std::move(module_ref)));
  TF_ASSERT_OK_AND_ASSIGN(auto module_new_opt,
                          GetOptimizedModule(std::move(module_new)));

  ASSERT_TRUE(GetDynamicSliceFusions(*module_ref_opt).empty());
  ASSERT_FALSE(GetDynamicSliceFusions(*module_new_opt).empty());

  auto module_new_opt_clone = module_new_opt->Clone();
  TF_ASSERT_OK_AND_ASSIGN(
      auto exec, CreateExecutable(std::move(module_new_opt_clone), false));
  GpuExecutable* gpu_exec = dynamic_cast<GpuExecutable*>(exec.get());
  ASSERT_EQ(gpu_exec->GetThunk().thunks()[0]->kind(), Thunk::kCopy);

  ErrorSpec error{/*aabs=*/1e-3, /*arel=*/1e-3};
  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(std::move(module_ref_opt),
                                                std::move(module_new_opt),
                                                false, true, error));
}

TEST_F(DynamicSliceFusionTest, TestWithRewriter) {
  const char* hlo = R"(
    HloModule test_module, replica_count=2

    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }

    Body {
      param = (s32[], s32[16, 32], s32[8, 32]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      dest = s32[16,32] get-tuple-element(param), index=1
      src = s32[8,32] get-tuple-element(param), index=2
      eight = s32[] constant(8)
      zero = s32[] constant(0)
      thirty_two = s32[] constant(32)
      add = s32[] add(eight, i)
      add.2 = s32[] subtract(add, thirty_two)
      compare = pred[] compare(add, thirty_two), direction=LT
      offset = s32[] select(compare, add, add.2)
      rs = s32[4,32] reduce-scatter(src), channel_id=0, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
      fusion = s32[16,32] dynamic-update-slice(dest, rs, offset, zero)
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = tuple(i_plus_one, fusion, src)
    }

    Cond {
      param = (s32[], s32[16,32], s32[8,32]) parameter(0)
      loop_iter = s32[] get-tuple-element(param), index=0
      c32 = s32[] constant(32)
      ROOT compare = pred[] compare(loop_iter, c32), direction=LT
    }

    ENTRY main {
      zero = s32[] constant(0)
      dest = s32[16,32] parameter(0)
      src = s32[8,32] parameter(1)
      tuple = tuple(zero, dest, src)
      ROOT while = while(tuple), body=Body, condition=Cond
    }
  )";

  HloModuleConfig config;
  DebugOptions dboptions;
  dboptions.set_xla_gpu_enable_dynamic_slice_fusion(false);
  config.set_debug_options(dboptions);
  TF_ASSERT_OK_AND_ASSIGN(auto module0,
                          ParseAndReturnVerifiedModule(hlo, config));

  TF_ASSERT_OK_AND_ASSIGN(auto module_without_fusion,
                          GetOptimizedModule(std::move(module0)));
  dboptions.set_xla_gpu_enable_dynamic_slice_fusion(true);
  config.set_debug_options(dboptions);
  TF_ASSERT_OK_AND_ASSIGN(auto module1,
                          ParseAndReturnVerifiedModule(hlo, config));
  TF_ASSERT_OK_AND_ASSIGN(auto module_with_fusion,
                          GetOptimizedModule(std::move(module1)));

  ASSERT_EQ(GetDynamicSliceFusions(*module_without_fusion).size(), 0);
  auto fusions = GetDynamicSliceFusions(*module_with_fusion);
  ASSERT_EQ(fusions.size(), 1);
  HloPrintOptions options;
  options.set_print_large_constants(true)
      .set_print_result_shape(false)
      .set_print_operand_shape(false);
  TF_ASSERT_OK_AND_ASSIGN(auto filecheck_fusion,
                          RunFileCheck(fusions[0]->ToString(options),
                                       R"(
      // CHECK-DAG: %[[rs:.+]] = reduce-scatter({{.+}})
      // CHECK-DAG: %[[offset_vals:.+]] = constant({8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7})
      // CHECK-DAG: %[[offset_as_arr:.+]] = dynamic-slice(%[[offset_vals]], {{.+}}), dynamic_slice_sizes={1}
      // CHECK-DAG: %[[offset:.+]] = reshape(%[[offset_as_arr]])
      // CHECK-DAG: ROOT %{{.+}} = dynamic-update-slice({{.+}}, %[[rs]], %[[offset]], {{.+}})
    )"));
  EXPECT_TRUE(filecheck_fusion);
  TF_ASSERT_OK_AND_ASSIGN(
      auto filecheck_while_loop,
      RunFileCheck(fusions[0]->FusionInstruction()->parent()->ToString(options),
                   R"(
      // CHECK-DAG: %[[p:.+]] = parameter(0)
      // CHECK-DAG: %[[loop_counter:.+]] = get-tuple-element(%[[p]]), index=3
      // CHECK-DAG: %[[address_computation:.+]] = fusion({{.+}}, %[[loop_counter]]), kind=kCustom
      // CHECK-DAG: %[[updated_loop_counter:.+]] = add(%[[loop_counter]], {{.+}})
      // CHECK-DAG: ROOT {{.+}} = tuple({{.+}}, %[[address_computation]], {{.+}}, %[[updated_loop_counter]])
    )"));
  EXPECT_TRUE(filecheck_while_loop);
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};
  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(
      std::move(module_without_fusion), std::move(module_with_fusion), false,
      true, error_spec));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
