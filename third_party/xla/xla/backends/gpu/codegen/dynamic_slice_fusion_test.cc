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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/error_spec.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/transforms/dynamic_slice_fusion_rewriter.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

std::string GetPlatformName() {
  return absl::AsciiStrToUpper(
      PlatformUtil::CanonicalPlatformName("gpu").value());
}

using ::testing::ElementsAre;
using ::testing::Optional;
using ::testing::VariantWith;

MATCHER_P(ThunkKindIs, kind, "") {
  return ExplainMatchResult(::testing::Eq(kind), arg->kind(), result_listener);
}

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
      if (IsDynamicSliceFusion(computation->FusionInstruction())) {
        computations.push_back(computation);
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
    ROOT %get-tuple-element.0 = f16[8,8]{1,0} get-tuple-element(%custom-call.1), index=0
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
    %fusion.2 = (f16[8,8]{1,0}, s8[256]{0}) fusion(%p0, %p1), kind=kCustom, calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
    ROOT %get-tuple-element.0 = f16[8,8]{1,0} get-tuple-element(%fusion.2), index=0
  })";

  EXPECT_TRUE(RunAndCompareTwoModules(
      hlo_ref, hlo_opt, GetModuleConfigWithDeterministicOps(),
      GetModuleConfigWithDeterministicOps(), error_spec,
      /*run_hlo_passes=*/false));
}

TEST_F(DynamicSliceFusionTest, NestedTupleOutputForCublasGemmWithWorkspace) {
  ErrorSpec error_spec{/*aabs=*/1e-3, /*arel=*/1e-3};

  const char* hlo_ref = R"(
  HloModule nested_tuple

  ENTRY main {
    p0 = f16[2,8,8]{2,1,0} parameter(0)
    p1 = f16[2,8,8]{2,1,0} parameter(1)
    slice_1 = f16[1,8,8]{2,1,0} slice(p0), slice={[1:2], [0:8], [0:8]}
    bitcast_1 = f16[8,8]{1,0} bitcast(slice_1)
    slice_2 = f16[1,8,8]{2,1,0} slice(p1), slice={[1:2], [0:8], [0:8]}
    bitcast_2 = f16[8,8]{1,0} bitcast(slice_2)

    custom-call = (f16[8,8]{1,0}, s8[256]{0}) custom-call(bitcast_1, bitcast_2),
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
    result = f16[8,8]{1,0} get-tuple-element(custom-call), index=0
    workspace = s8[256]{0} get-tuple-element(custom-call), index=1
    nested_tuple = (s8[256]{0}) tuple(workspace)
    ROOT tuple = (f16[8,8]{1,0}, (s8[256]{0})) tuple(result, nested_tuple)
  })";

  const char* hlo_opt = R"(
  HloModule jit_slice

  fused_computation {
    p0 = f16[2,8,8]{2,1,0} parameter(0)
    p1 = f16[2,8,8]{2,1,0} parameter(1)
    slice_1 = f16[1,8,8]{2,1,0} slice(p0), slice={[1:2], [0:8], [0:8]}
    bitcast_1 = f16[8,8]{1,0} bitcast(slice_1)
    slice_2 = f16[1,8,8]{2,1,0} slice(p1), slice={[1:2], [0:8], [0:8]}
    bitcast_2 = f16[8,8]{1,0} bitcast(slice_2)

    custom-call = (f16[8,8]{1,0}, s8[256]{0}) custom-call(bitcast_1, bitcast_2),
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
    result = f16[8,8]{1,0} get-tuple-element(custom-call), index=0
    workspace = s8[256]{0} get-tuple-element(custom-call), index=1
    nested_tuple = (s8[256]{0}) tuple(workspace)
    ROOT tuple = (f16[8,8]{1,0}, (s8[256]{0})) tuple(result, nested_tuple)
  }

  ENTRY main.9 {
    p0 = f16[2,8,8]{2,1,0} parameter(0)
    p1 = f16[2,8,8]{2,1,0} parameter(1)
    ROOT fusion = (f16[8,8]{1,0}, (s8[256]{0})) fusion(p0, p1), kind=kCustom, calls=fused_computation,
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
    %custom-call.17 = (f32[100,100]{1,0}, s8[80000]{0}) custom-call(%slice.26, %get-tuple-element.240),
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
    ROOT %get-tuple-element.221 = f32[100,100]{1,0} get-tuple-element(%custom-call.17), index=0
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
    %dynamic-slice-fusion.6 = (f32[100,100]{1,0}, s8[80000]{0}) fusion(%get-tuple-element.97, %get-tuple-element.240),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
    ROOT %get-tuple-element.221 = f32[100,100]{1,0} get-tuple-element(%dynamic-slice-fusion.6), index=0
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
    %custom-call.17 = (f32[100,100]{1,0}, s8[80000]{0}) custom-call(%slice.26, %slice.26),
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
    ROOT %get-tuple-element.221 = f32[100,100]{1,0} get-tuple-element(%custom-call.17), index=0
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
    %dynamic-slice-fusion.6 = (f32[100,100]{1,0}, s8[80000]{0}) fusion(%get-tuple-element.97),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
    ROOT %get-tuple-element.221 = f32[100,100]{1,0} get-tuple-element(%dynamic-slice-fusion.6), index=0
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
      %cublas-gemm.15 = (f32[100,100]{1,0}, s8[120000]{0}) custom-call(%get-tuple-element.287, %slice.30, %slice.34),
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
      ROOT %get-tuple-element.289 = f32[100,100]{1,0} get-tuple-element(%cublas-gemm.15), index=0
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
    %dynamic-slice-fusion.6 = (f32[100,100]{1,0}, s8[120000]{0}) fusion(%get-tuple-element.287, %slice.34, %concatenate.12),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
      output_to_operand_aliasing={{0}: (1, {})},
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
    ROOT %get-tuple-element.289 = f32[100,100]{1,0} get-tuple-element(%dynamic-slice-fusion.6), index=0
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
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", "CUDA",
                         kMemcpy);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", "ROCM",
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto program_shape,
      xla::ProgramShape::FromProto(computation.proto().host_program_shape()));
  xla::HloModuleConfig hlo_config(program_shape,
                                  /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_opt, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));
  DynamicSliceFusionRewriter pass(GetPlatformName());
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
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$subbuffers", "CUDA",
                         kSubBuffers);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$subbuffers", "ROCM",
                         kSubBuffers);

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
  TF_ASSERT_OK_AND_ASSIGN(
      auto program_shape,
      xla::ProgramShape::FromProto(computation.proto().host_program_shape()));
  xla::HloModuleConfig hlo_config(program_shape,
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

  DynamicSliceFusionRewriter pass(GetPlatformName());
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
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$noop", "CUDA",
                         kNoOp);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$noop", "ROCM",
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto program_shape,
      xla::ProgramShape::FromProto(computation.proto().host_program_shape()));
  xla::HloModuleConfig hlo_config(program_shape,
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

  DynamicSliceFusionRewriter pass(GetPlatformName());
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
    ROOT %gte = f16[8,8]{1,0} get-tuple-element(%custom-call.1), index=0
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
    %fusion.2 = (f16[8,8]{1,0}, s8[256]{0}) fusion(%p0, %p1, %c1_s32, %c0_s32), kind=kCustom, calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
    ROOT %gte = f16[8,8]{1,0} get-tuple-element(%fusion.2), index=0
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
    %custom-call.17 = (f32[100,100]{1,0}, s8[80000]{0}) custom-call(%slice.26, %get-tuple-element.240),
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
    ROOT %get-tuple-element.221 = f32[100,100]{1,0} get-tuple-element(%custom-call.17), index=0
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
    %dynamic-slice-fusion.6 = (f32[100,100]{1,0}, s8[80000]{0}) fusion(%get-tuple-element.97, %get-tuple-element.240, %c0_s32),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
    ROOT %get-tuple-element.221 = f32[100,100]{1,0} get-tuple-element(%dynamic-slice-fusion.6), index=0
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
    %custom-call.17 = (f32[100,100]{1,0}, s8[80000]{0}) custom-call(%slice.26, %slice.26),
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
    ROOT %get-tuple-element.221 = f32[100,100]{1,0} get-tuple-element(%custom-call.17), index=0
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
    %dynamic-slice-fusion.6 = (f32[100,100]{1,0}, s8[80000]{0}) fusion(%get-tuple-element.97, %c0_s32),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
    ROOT %get-tuple-element.221 = f32[100,100]{1,0} get-tuple-element(%dynamic-slice-fusion.6), index=0
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
      %cublas-gemm.15 = (f32[100,100]{1,0}, s8[120000]{0}) custom-call(%get-tuple-element.287, %slice.30, %slice.34),
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
      ROOT %get-tuple-element.289 = f32[100,100]{1,0} get-tuple-element(%cublas-gemm.15), index=0
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
    %dynamic-slice-fusion.6 = (f32[100,100]{1,0}, s8[120000]{0}) fusion(%get-tuple-element.287, %slice.34, %concatenate.12, %c0_s32, %c20_s32),
      kind=kCustom,
      calls=%dynamic-slice-fusion,
      output_to_operand_aliasing={{0}: (1, {})},
      backend_config={
        "fusion_backend_config":{
          "kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}
        }
      }
    ROOT %get-tuple-element.289 = f32[100,100]{1,0} get-tuple-element(%dynamic-slice-fusion.6), index=0
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
    ROOT %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
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
    %fusion.2 = (f16[4,8,8]{2,1,0}, s8[256]{0}) fusion(%p0, %p1, %p2, %c1_s32, %c0_s32), kind=kCustom, calls=%fused_computation,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
    ROOT %gte = f16[4,8,8]{2,1,0} get-tuple-element(%fusion.2), index=0
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
    ROOT %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
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
    ROOT %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
  }

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s32[] parameter(3)
    %c0_s32 = s32[] parameter(4)
    ROOT %fusion.2 = f16[4,8,8]{2,1,0} fusion(%p0, %p1, %p2, %c1_s32, %c0_s32), kind=kCustom, calls=%fused_computation,
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
    ROOT %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
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
    ROOT %dus = f16[4,8,8]{2,1,0} dynamic-update-slice(%p2, %bitcast.43, %c1_s32, %c0_s32, %c0_s32)
  }

  ENTRY %main.9 {
    %p0 = f16[2,8,8]{2,1,0} parameter(0)
    %p1 = f16[2,8,8]{2,1,0} parameter(1)
    %p2 = f16[4,8,8]{2,1,0} parameter(2)
    %c1_s32 = s64[] constant(10)
    %c0_s32 = s64[] constant(-1)
    ROOT %fusion.2 = f16[4,8,8]{2,1,0} fusion(%p0, %p1, %p2, %c1_s32, %c0_s32), kind=kCustom, calls=%fused_computation,
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto program_shape,
      xla::ProgramShape::FromProto(computation.proto().host_program_shape()));
  xla::HloModuleConfig hlo_config(program_shape,
                                  /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_ref, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_opt, xla::HloModule::CreateFromProto(
                                            computation.proto(), hlo_config));
  DynamicSliceFusionRewriter pass(GetPlatformName());
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
  TF_ASSERT_OK_AND_ASSIGN(
      auto program_shape,
      xla::ProgramShape::FromProto(computation.proto().host_program_shape()));
  xla::HloModuleConfig hlo_config(program_shape,
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

  DynamicSliceFusionRewriter pass(GetPlatformName());
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
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$subbuffers2", "CUDA",
                         kSubBuffers2);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$subbuffers2", "ROCM",
                         kSubBuffers2);

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
  TF_ASSERT_OK_AND_ASSIGN(
      auto program_shape,
      xla::ProgramShape::FromProto(computation.proto().host_program_shape()));
  xla::HloModuleConfig hlo_config(program_shape,
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

  DynamicSliceFusionRewriter pass(GetPlatformName());
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
  debugoptions.set_xla_gpu_enable_pipelined_reduce_scatter(false);
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
      std::unique_ptr<OpaqueExecutable> wrapped_executable,
      CreateExecutable(std::move(module_new_opt_clone), false));
  TF_ASSERT_OK_AND_ASSIGN(Executable* const exec,
                          test_runner_as_hlo_runner().ExecutableFromWrapped(
                              wrapped_executable.get()));
  GpuExecutable* gpu_exec = dynamic_cast<GpuExecutable*>(exec);

  // The pattern we have here is a static slice along with reduce-scatter
  // operation. With this pattern, we can compute the offset at compile time and
  // we do not need to emit a dynamic slice thunk to compute the offset at
  // runtime. So, we expect to see kNcclReduceScatterStart and
  // kNcclReduceScatterDone thunks. We also expect to see surrounding
  // kWaitsForStreams thunks because dynamic slice fusion with a collective hero
  // is converted into an async operation. The kWaitForStreams thunks are
  // expected because of the async operation.
  ASSERT_EQ(gpu_exec->GetThunk().thunks().size(), 4ul);
  EXPECT_THAT(gpu_exec->GetThunk().thunks(),
              ::testing::ElementsAre(ThunkKindIs(Thunk::kWaitForStreams),
                                     ThunkKindIs(Thunk::kReduceScatterStart),
                                     ThunkKindIs(Thunk::kReduceScatterDone),
                                     ThunkKindIs(Thunk::kWaitForStreams)));

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

TEST_F(DynamicSliceFusionTest,
       OffsetsThatCanBeEvaluatedSuccessfullyAreCorrectlyEmbeddedIntoThunks) {
  const char* hlo_opt = R"(
    HloModule test, replica_count=2
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a,b)
    }
    dynamic-slice-fusion {
      src = s32[32,32] parameter(0)
      dest = s32[32,32] parameter(1)
      offset1 = s32[] parameter(2)
      offset2 = s32[] parameter(3)
      rs = s32[16,32] reduce-scatter(src), dimensions={0}, replica_groups={{0,1}}, to_apply=add
      ROOT dus = s32[32,32] dynamic-update-slice(dest, rs, offset1, offset2)
    }
    ENTRY main {
      src = s32[32,32] parameter(0)
      dest = s32[32,32] parameter(1)
      c0 = s32[] constant(0)
      c5 = s32[] constant(5)
      add = s32[] add(c5, c5)
      ROOT fusion = s32[32,32] fusion(src, dest, add, c0), kind=kCustom, calls=dynamic-slice-fusion,
        backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
    }
  )";

  const char* hlo_ref = R"(
    HloModule test, replica_count=2
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a,b)
    }
    ENTRY main {
      src = s32[32,32] parameter(0)
      dest = s32[32,32] parameter(1)
      c0 = s32[] constant(0)
      c5 = s32[] constant(5)
      add = s32[] add(c5, c5)
      rs.1 = ((s32[32,32]), s32[16,32]) reduce-scatter-start(src), dimensions={0}, replica_groups={{0,1}}, to_apply=add
      rs = s32[16,32] reduce-scatter-done(rs.1)
      ROOT dus = s32[32,32] dynamic-update-slice(dest, rs, add, c0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_ref,
                          ParseAndReturnVerifiedModule(hlo_ref));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_opt,
                          ParseAndReturnVerifiedModule(hlo_opt));

  // Check that the offset value in the thunk is an evaluated constant even if
  // no simplification passes are executed.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> wrapped_executable,
                          CreateExecutable(/*module=*/module_opt->Clone(),
                                           /*run_hlo_passes=*/false));
  TF_ASSERT_OK_AND_ASSIGN(Executable* const exec,
                          test_runner_as_hlo_runner().ExecutableFromWrapped(
                              wrapped_executable.get()));
  GpuExecutable* gpu_exec = dynamic_cast<GpuExecutable*>(exec);
  ASSERT_NE(gpu_exec, nullptr);
  const SequentialThunk& thunk = gpu_exec->GetThunk();
  auto dynamic_slice_thunk =
      absl::c_find_if(thunk.thunks(), [](const std::unique_ptr<Thunk>& thunk) {
        return thunk->kind() == Thunk::kDynamicSlice;
      });
  ASSERT_NE(dynamic_slice_thunk, thunk.thunks().end());
  std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets =
      dynamic_cast<DynamicSliceThunk*>(dynamic_slice_thunk->get())
          ->get_offsets();
  ASSERT_EQ(offsets.size(), 2);
  ASSERT_TRUE(offsets[1].has_value());
  ASSERT_EQ(offsets[1].value()[0], DynamicSliceThunk::Offset(10l));
  ASSERT_EQ(offsets[1].value()[1], DynamicSliceThunk::Offset(0l));

  ErrorSpec error{1e-3, 1e-3};
  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(
      /*module_0=*/std::move(module_ref), /*module_1=*/std::move(module_opt),
      /*run_hlo_passes=*/false, /*use_threads=*/true, error));
}

TEST_F(DynamicSliceFusionTest,
       AsyncDynamicSliceFusionWithCollectiveOverlapsWithComputeThunk) {
  const char* hlo = R"(
    HloModule test-clone, replica_count=2

    add {
      x = s32[] parameter(0)
      y = s32[] parameter(1)
      ROOT add = s32[] add(x, y)
    }

    dynamic-slice-fusion {
      p1 = s32[2,2,32]{2,1,0} parameter(1)
      p0 = s32[8,32]{1,0} parameter(0)
      slice = s32[4,32]{1,0} slice(p0), slice={[4:8], [0:32]}
      rs = s32[2,32]{1,0} reduce-scatter(slice), replica_groups={{0,1}}, dimensions={0}, to_apply=add
      bitcast = s32[1,2,32]{2,1,0} bitcast(rs)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT dynamic-update-slice = s32[2,2,32]{2,1,0} dynamic-update-slice(p1, bitcast, p2, p3, p3)
    }

    ENTRY main {
      source = s32[8,32]{1,0} parameter(1)
      destination = s32[2,2,32]{2,1,0} parameter(0)
      copy = s32[2,2,32]{2,1,0} copy(destination)
      c1 = s32[] constant(1)
      c0 = s32[] constant(0)
      fusion-start = ((s32[8,32]{1,0}, s32[2,2,32]{2,1,0}, s32[], s32[]), s32[2,2,32]{2,1,0}, u32[]) fusion-start(source, copy, c1, c0), kind=kCustom, calls=dynamic-slice-fusion, backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
      fusion-done = s32[2,2,32]{2,1,0} fusion-done(fusion-start), backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
      a = s32[1024,1024]{1,0} parameter(2)
      b = s32[1024,1024]{1,0} parameter(3)
      dot = s32[1024,1024]{1,0} dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT tuple = (s32[2,2,32]{2,1,0}, s32[1024,1024]{1,0}) tuple(fusion-done, dot)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> wrapped_exec,
      CreateExecutable(hlo_module->Clone(), /*run_hlo_passes=*/false));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> exec,
                          test_runner_as_hlo_runner().ExecutableFromWrapped(
                              std::move(wrapped_exec)));
  GpuExecutable* gpu_exec = dynamic_cast<GpuExecutable*>(exec.get());
  const ThunkSequence& thunks = gpu_exec->GetThunk().thunks();

  // This is only needed to ensure that the next checks don't fail.
  ASSERT_EQ(thunks.size(), 6);

  // In the following checks, only the order of the thunks matter.
  EXPECT_THAT(thunks,
              ::testing::ElementsAre(ThunkKindIs(Thunk::kCopy),
                                     ThunkKindIs(Thunk::kWaitForStreams),
                                     ThunkKindIs(Thunk::kDynamicSlice),
                                     ThunkKindIs(Thunk::kKernel),
                                     ThunkKindIs(Thunk::kReduceScatterDone),
                                     ThunkKindIs(Thunk::kWaitForStreams)));

  // Check that the dynamic slice thunk only produces a start thunk, and not a
  // done thunk.
  DynamicSliceThunk* dynamic_slice_thunk =
      dynamic_cast<DynamicSliceThunk*>(thunks[2].get());
  ASSERT_NE(dynamic_slice_thunk, nullptr);
  const SequentialThunk* embedded_thunk = dynamic_cast<const SequentialThunk*>(
      dynamic_slice_thunk->embedded_thunk());
  ASSERT_NE(embedded_thunk, nullptr);
  EXPECT_THAT(embedded_thunk->thunks(),
              ::testing::ElementsAre(ThunkKindIs(Thunk::kReduceScatterStart)));

  // Check that the offsets were propagated as constants, and not as device
  // allocated buffers.
  auto offsets = dynamic_slice_thunk->get_offsets();
  EXPECT_THAT(offsets,
              ElementsAre(std::nullopt,
                          Optional(ElementsAre(VariantWith<int64_t>(1),
                                               VariantWith<int64_t>(0),
                                               VariantWith<int64_t>(0)))));
}

TEST_F(DynamicSliceFusionTest,
       OffsetAsFunctionOfInductionVariableShouldUseOffsetModules) {
  const char* hlo_fused = R"(
    HloModule test, replica_count=2
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }
    dynamic-slice-fusion {
      p1 = s32[32,32] parameter(1)
      p0 = s32[32,32] parameter(0)
      rs = s32[16,32] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0}, to_apply=add
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT dus = s32[32,32] dynamic-update-slice(p1, rs, p2, p3)
    }
    body {
      param = (s32[], s32[32,32], s32[32,32]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      c1 = s32[] constant(1)
      add = s32[] add(iter, c1)
      src = s32[32,32] get-tuple-element(param), index=1
      dest = s32[32,32] get-tuple-element(param), index=2

      // Offset calculation as a function of the induction variable.
      add.1 = s32[] add(iter, iter)
      c3 = s32[] constant(3)
      multiply = s32[] multiply(add.1, c3)
      c16 = s32[] constant(16)
      offset = s32[] subtract(multiply, c16)

      c0 = s32[] constant(0)
      address_computation = s32[32,32] fusion(src, dest, offset, c0), kind=kCustom, calls=dynamic-slice-fusion, backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
      ROOT tuple = (s32[], s32[32,32], s32[32,32]) tuple(add, src, address_computation)
    }
    condition {
      param = (s32[], s32[32,32], s32[32,32]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      c16 = s32[] constant(16)
      ROOT compare = pred[] compare(iter, c16), direction=LT
    }
    ENTRY main {
      c0 = s32[] constant(0)
      src = s32[32,32] parameter(0)
      dest = s32[32,32] parameter(1)
      tuple = (s32[], s32[32,32], s32[32,32]) tuple(c0, src, dest)
      ROOT while = (s32[], s32[32,32], s32[32,32]) while(tuple), condition=condition, body=body
    })";
  const char* hlo_unfused = R"(
    HloModule test, replica_count=2

    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }

    body {
      param = (s32[], s32[32,32], s32[32,32]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      src = s32[32,32] get-tuple-element(param), index=1
      dest = s32[32,32] get-tuple-element(param), index=2

      // Offset calculation as a function of the induction variable.
      add = s32[] add(iter, iter)
      c3 = s32[] constant(3)
      multiply = s32[] multiply(add, c3)
      c16 = s32[] constant(16)
      offset = s32[] subtract(multiply, c16)

      c0 = s32[] constant(0)
      rs_start = ((s32[32,32]), s32[16,32]) reduce-scatter-start(src), dimensions={0}, replica_groups={{0,1}}, to_apply=add
      rs = s32[16,32] reduce-scatter-done(rs_start)
      dus = s32[32,32] dynamic-update-slice(dest, rs, offset, c0)
      c1 = s32[] constant(1)
      add.1 = s32[] add(iter, c1)
      ROOT tuple = tuple(add.1, src, dus)
    }

    condition {
      param = (s32[], s32[32,32], s32[32,32]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      c16 = s32[] constant(16)
      ROOT compare = pred[] compare(iter, c16), direction=LT
    }

    ENTRY main {
      src = s32[32,32] parameter(0)
      dest = s32[32,32] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[32,32], s32[32,32]) tuple(c0, src, dest)
      ROOT while = (s32[], s32[32,32], s32[32,32]) while(tuple), body=body, condition=condition
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> fused_module,
                          ParseAndReturnVerifiedModule(hlo_fused));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> wrapped_exec,
                          CreateExecutable(fused_module->Clone(), false));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> exec,
                          test_runner_as_hlo_runner().ExecutableFromWrapped(
                              std::move(wrapped_exec)));
  GpuExecutable* gpu_exec = dynamic_cast<GpuExecutable*>(exec.get());
  ASSERT_NE(gpu_exec, nullptr);

  auto while_thunk = absl::c_find_if(gpu_exec->GetThunk().thunks(),
                                     [](const std::unique_ptr<Thunk>& thunk) {
                                       return thunk->kind() == Thunk::kWhile;
                                     });
  ASSERT_NE(while_thunk, gpu_exec->GetThunk().thunks().end());
  WhileThunk* while_thunk_ptr = dynamic_cast<WhileThunk*>(while_thunk->get());

  auto ds_thunk =
      absl::c_find_if(while_thunk_ptr->body_thunk_sequence()->thunks(),
                      [](const std::unique_ptr<Thunk>& thunk) {
                        return thunk->kind() == Thunk::kDynamicSlice;
                      });
  ASSERT_NE(ds_thunk, while_thunk_ptr->body_thunk_sequence()->thunks().end());
  DynamicSliceThunk* ds_thunk_ptr =
      dynamic_cast<DynamicSliceThunk*>(ds_thunk->get());
  std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets =
      ds_thunk_ptr->get_offsets();

  // Expect two offsets: one for the input, and one for the outputs.
  ASSERT_EQ(offsets.size(), 2);
  ASSERT_TRUE(offsets[1].has_value());
  std::vector<DynamicSliceThunk::Offset> output_offsets = *offsets[1];
  ASSERT_EQ(output_offsets.size(), 2);

  // The first value of offset must be an HloModule
  HloModule** offset_0 = std::get_if<HloModule*>(&output_offsets[0]);
  ASSERT_NE(offset_0, nullptr);
  ASSERT_NE(*offset_0, nullptr);

  // The second offset must be a constant value
  ASSERT_EQ(output_offsets[1], DynamicSliceThunk::Offset(0l));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> unfused_module,
                          ParseAndReturnVerifiedModule(hlo_unfused));

  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(
      std::move(fused_module), std::move(unfused_module),
      /*run_hlo_passes=*/false, /*use_threads=*/true, std::nullopt));
}

TEST_F(DynamicSliceFusionTest, MultipleOffsetsAsFunctionOfInductionVariable) {
  const char* hlo_fused = R"(
    HloModule test, replica_count=2
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }
    dynamic-slice-fusion {
      p0 = s32[16,32,32] parameter(0)
      p1 = s32[32,32] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      p4 = s32[] parameter(4)
      ds = s32[1,32,32] dynamic-slice(p0, p2, p4, p4), dynamic_slice_sizes={1,32,32}
      bitcast = s32[32,32] bitcast(ds)
      rs = s32[16,32] reduce-scatter(bitcast), replica_groups={{0,1}}, dimensions={0}, to_apply=add
      ROOT dus = s32[32,32] dynamic-update-slice(p1, rs, p3, p4)
    }
    body {
      param = (s32[], s32[16,32,32], s32[32,32]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      c1 = s32[] constant(1)
      add = s32[] add(iter, c1)
      src = s32[16,32,32] get-tuple-element(param), index=1
      dest = s32[32,32] get-tuple-element(param), index=2

      // Offset calculation as a function of the induction variable.
      // offset.1 = 5i-32
      c5 = s32[] constant(5)
      c32 = s32[] constant(32)
      multiply.1 = s32[] multiply(c5, iter)
      offset.1 = s32[] subtract(multiply.1, c32)
      // offset.2 = 6i-16
      add.1 = s32[] add(iter, iter)
      c3 = s32[] constant(3)
      multiply.2 = s32[] multiply(add.1, c3)
      c16 = s32[] constant(16)
      offset.2 = s32[] subtract(multiply.2, c16)

      c0 = s32[] constant(0)
      address_computation = s32[32,32] fusion(src, dest, offset.1, offset.2, c0), kind=kCustom, calls=dynamic-slice-fusion, backend_config={"fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"}}}
      ROOT tuple = (s32[], s32[16,32,32], s32[32,32]) tuple(add, src, address_computation)
    }
    condition {
      param = (s32[], s32[16,32,32], s32[32,32]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      c16 = s32[] constant(16)
      ROOT compare = pred[] compare(iter, c16), direction=LT
    }
    ENTRY main {
      c0 = s32[] constant(0)
      src = s32[16,32,32] parameter(0)
      dest = s32[32,32] parameter(1)
      tuple = (s32[], s32[16,32,32], s32[32,32]) tuple(c0, src, dest)
      ROOT while = (s32[], s32[16,32,32], s32[32,32]) while(tuple), condition=condition, body=body
    })";
  const char* hlo_unfused = R"(
    HloModule test, replica_count=2

    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }

    body {
      param = (s32[], s32[16,32,32], s32[32,32]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      src = s32[16,32,32] get-tuple-element(param), index=1
      dest = s32[32,32] get-tuple-element(param), index=2

      // Offset calculation as a function of the induction variable.
      // offset.1 = 5i-32
      c5 = s32[] constant(5)
      c32 = s32[] constant(32)
      multiply.1 = s32[] multiply(c5, iter)
      offset.1 = s32[] subtract(multiply.1, c32)
      // offset.2 = 6i-16
      add = s32[] add(iter, iter)
      c3 = s32[] constant(3)
      multiply.2 = s32[] multiply(add, c3)
      c16 = s32[] constant(16)
      offset.2 = s32[] subtract(multiply.2, c16)

      c0 = s32[] constant(0)
      ds = s32[1,32,32] dynamic-slice(src, offset.1, c0, c0), dynamic_slice_sizes={1,32,32}
      reshape = s32[32,32] reshape(ds)
      rs_start = ((s32[32,32]), s32[16,32]) reduce-scatter-start(reshape), dimensions={0}, replica_groups={{0,1}}, to_apply=add
      rs = s32[16,32] reduce-scatter-done(rs_start)
      dus = s32[32,32] dynamic-update-slice(dest, rs, offset.2, c0)
      c1 = s32[] constant(1)
      add.1 = s32[] add(iter, c1)
      ROOT tuple = tuple(add.1, src, dus)
    }

    condition {
      param = (s32[], s32[16,32,32], s32[32,32]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      c16 = s32[] constant(16)
      ROOT compare = pred[] compare(iter, c16), direction=LT
    }

    ENTRY main {
      src = s32[16,32,32] parameter(0)
      dest = s32[32,32] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[16,32,32], s32[32,32]) tuple(c0, src, dest)
      ROOT while = (s32[], s32[16,32,32], s32[32,32]) while(tuple), body=body, condition=condition
    }
  )";

  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(
      /*module_0_str=*/hlo_unfused, /*module_1_str=*/hlo_fused,
      /*run_hlo_passes=*/false, /*use_threads=*/true, std::nullopt));
}

TEST_F(DynamicSliceFusionTest,
       ReduceScatterDynamicSliceMultipleBuffersShouldFuseAndExecuteCorrectly) {
  const char* hlo = R"(
    HloModule test, replica_count=2
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }
    body {
      param.1 = (s32[], s32[8,8,8], s32[8,8,8], s32[8,4,8], s32[8,4,8]) parameter(0)
      iter.1 = s32[] get-tuple-element(param.1), index=0
      c1 = s32[] constant(1)
      c0 = s32[] constant(0)
      src1 = s32[8,8,8] get-tuple-element(param.1), index=1
      src2 = s32[8,8,8] get-tuple-element(param.1), index=2
      dst1 = s32[8,4,8] get-tuple-element(param.1), index=3
      dst2 = s32[8,4,8] get-tuple-element(param.1), index=4
      ds1 = s32[1,8,8]{2,1,0} dynamic-slice(src1, iter.1, c0, c0), dynamic_slice_sizes={1,8,8}
      ds2 = s32[1,8,8]{2,1,0} dynamic-slice(src2, iter.1, c0, c0), dynamic_slice_sizes={1,8,8}
      rs1 = s32[8,8] reshape(ds1)
      rs2 = s32[8,8] reshape(ds2)
      rs = (s32[4,8], s32[4,8]) reduce-scatter(rs1, rs2), dimensions={0}, replica_groups={{0,1}}, to_apply=add
      reduce-scatter1 = s32[4,8] get-tuple-element(rs), index=0
      reduce-scatter2 = s32[4,8] get-tuple-element(rs), index=1
      reshape1 = s32[1,4,8] reshape(reduce-scatter1)
      reshape2 = s32[1,4,8] reshape(reduce-scatter2)
      dus1 = s32[8,4,8] dynamic-update-slice(dst1, reshape1, iter.1, c0, c0)
      dus2 = s32[8,4,8] dynamic-update-slice(dst2, reshape2, iter.1, c0, c0)
      add = s32[] add(iter.1, c1)
      ROOT tuple = tuple(add, src1, src2, dus1, dus2)
    }
    condition {
      param.2 = (s32[], s32[8,8,8], s32[8,8,8], s32[8,4,8], s32[8,4,8]) parameter(0)
      iter.2 = s32[] get-tuple-element(param.2), index=0
      c8 = s32[] constant(8)
      ROOT compare = pred[] compare(iter.2, c8), direction=LT
    }
    ENTRY main {
      c0 = s32[] constant(0)
      p1 = s32[8,8,8] parameter(0)
      p2 = s32[8,8,8] parameter(1)
      p3 = s32[8,4,8] parameter(2)
      p4 = s32[8,4,8] parameter(3)
      tuple = (s32[], s32[8,8,8], s32[8,8,8], s32[8,4,8], s32[8,4,8]) tuple(c0, p1, p2, p3, p4)
      ROOT while = (s32[], s32[8,8,8], s32[8,8,8], s32[8,4,8], s32[8,4,8]) while(tuple), body=body, condition=condition
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
  std::unique_ptr<HloModule> m_fused = m->Clone();
  m_fused->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_dynamic_slice_fusion(true);
  TF_ASSERT_OK_AND_ASSIGN(m_fused, GetOptimizedModule(std::move(m_fused)));
  TF_ASSERT_OK_AND_ASSIGN(m, GetOptimizedModule(std::move(m)));

  // Check that the fused module has a dynamic address computation.
  std::vector<HloComputation*> fused_dynamic_slice_fusions =
      GetDynamicSliceFusions(*m_fused);
  ASSERT_EQ(fused_dynamic_slice_fusions.size(), 1);
  // Check that the unfused module does not have a dynamic address computation.
  std::vector<HloComputation*> unfused_dynamic_slice_fusions =
      GetDynamicSliceFusions(*m);
  ASSERT_EQ(unfused_dynamic_slice_fusions.size(), 0);

  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(
      std::move(m_fused), std::move(m),
      /*run_hlo_passes=*/false, /*use_threads=*/true, std::nullopt));
}

TEST_F(DynamicSliceFusionTest, WhileLoopSliceWithNoInductionVariable) {
  const char* hlo = R"(
  HloModule test, replica_count=2

  add {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT add = f32[] add(a, b)
  }

  body {
    param = (s32[], s32[], f32[128,128], f32[1024,128]) parameter(0)
    iter0 = s32[] get-tuple-element(param), index=0
    iter1 = s32[] get-tuple-element(param), index=1
    c0 = s32[] constant(0)
    c1 = s32[] constant(1)
    add0 = s32[] add(iter0, iter0)
    add1 = s32[] add(iter1, c1)
    a = f32[128,128] get-tuple-element(param), index=2
    b = f32[1024,128] get-tuple-element(param), index=3
    slice = f32[256,128] slice(b), slice={[0:256], [0:128]}
    rs = f32[128,128] reduce-scatter(slice), replica_groups={{0,1}}, dimensions={0}, to_apply=add
    ROOT tuple = tuple(add0, add1, rs, b)
  }

  condition {
    param = (s32[], s32[], f32[128,128], f32[1024,128]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    iter1 = s32[] get-tuple-element(param), index=1
    c8 = s32[] constant(8)
    compare1 = pred[] compare(iter, c8), direction=LT
    compare2 = pred[] compare(iter1, c8), direction=LT
    ROOT compare = pred[] and(compare1, compare2)
  }

  ENTRY main {
    c1 = s32[] constant(1)
    a = f32[128,128] parameter(0)
    b = f32[1024,128] parameter(1)
    tuple = tuple(c1, c1, a, b)
    while = while(tuple), body=body, condition=condition
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
  m->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_dynamic_slice_fusion(false);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_ref,
                          GetOptimizedModule(m->Clone()));
  m->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_dynamic_slice_fusion(true);
  TF_ASSERT_OK_AND_ASSIGN(m, GetOptimizedModule(std::move(m)));
  ErrorSpec error_spec(1e-5, 1e-5);
  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(std::move(m), std::move(m_ref),
                                                /*run_hlo_passes=*/false,
                                                /*use_threads=*/true,
                                                error_spec));
}
}  // namespace
}  // namespace gpu
}  // namespace xla
