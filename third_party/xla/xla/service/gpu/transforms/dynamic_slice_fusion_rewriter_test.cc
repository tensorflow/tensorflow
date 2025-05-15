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

#include "xla/service/gpu/transforms/dynamic_slice_fusion_rewriter.h"

#include <cstddef>
#include <optional>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu {

class DynamicSliceFusionRewriterTest : public HloHardwareIndependentTestBase {};

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemm) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main.9 {
      %p0 = f16[2,8,8]{2,1,0} parameter(0)
      %p1 = f16[2,8,8]{2,1,0} parameter(1)
      %slice.13 = f16[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
      %slice.14 = f16[1,8,8]{2,1,0} slice(%p1), slice={[1:2], [0:8], [0:8]}
      %bitcast.42 = f16[8,8]{1,0} bitcast(%slice.14)

      ROOT %custom-call.1 = f16[8,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
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
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P1]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = f16[8,8]{1,0} fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmWithWorkspace) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main.9 {
      %p0 = f16[2,8,8]{2,1,0} parameter(0)
      %p1 = f16[2,8,8]{2,1,0} parameter(1)
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
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P1]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       [[CC:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0}) custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:       [[DOT:%[^ ]+]] = f16[8,8]{1,0} get-tuple-element([[CC]]), index=0
    ; CHECK:       [[WORKSPACE:%[^ ]+]] = s8[256]{0} get-tuple-element([[CC]]), index=1
    ; CHECK:       ROOT [[TUPLE:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0})
    ; CHECK:              tuple([[DOT]], [[WORKSPACE]])
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0}) fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmWorkspaceIgnored) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main.9 {
      %p0 = f16[2,8,8]{2,1,0} parameter(0)
      %p1 = f16[2,8,8]{2,1,0} parameter(1)
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
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P1]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       [[CC:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0}) custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:       [[DOT:%[^ ]+]] = f16[8,8]{1,0} get-tuple-element([[CC]]), index=0
    ; CHECK:       [[WORKSPACE:%[^ ]+]] = s8[256]{0} get-tuple-element([[CC]]), index=1
    ; CHECK:       ROOT [[TUPLE:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0})
    ; CHECK:              tuple([[DOT]], [[WORKSPACE]])
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       [[FUSION:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0}) fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:       ROOT [[DOT_MAIN:%[^ ]+]] = f16[8,8]{1,0} get-tuple-element([[FUSION]]), index=0
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmNotRoot) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main.9 {
      %p0 = f16[2,8,8]{2,1,0} parameter(0)
      %p1 = f16[2,8,8]{2,1,0} parameter(1)
      %slice.13 = f16[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
      %slice.14 = f16[1,8,8]{2,1,0} slice(%p1), slice={[1:2], [0:8], [0:8]}
      %bitcast.42 = f16[8,8]{1,0} bitcast(%slice.14)

      %custom-call.1 = f16[8,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
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
      ROOT %res = f16[8,8]{1,0} add(%custom-call.1, %custom-call.1)
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P1]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       [[FUSION:%[^ ]+]] = f16[8,8]{1,0} fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:       ROOT {{.*}} = f16[8,8]{1,0} add([[FUSION]], [[FUSION]])
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmOperandHasMultipleUsers) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main.9 {
      %p0 = f16[2,8,8]{2,1,0} parameter(0)
      %p1 = f16[4,8,8]{2,1,0} parameter(1)
      %slice.13 = f16[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
      %slice.14 = f16[1,8,8]{2,1,0} slice(%p1), slice={[2:3], [0:8], [0:8]}
      %bitcast.42 = f16[8,8]{1,0} bitcast(%slice.14)

      %custom-call.1 = f16[8,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
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
      ROOT %res = f16[8,8]{1,0} add(%custom-call.1, %bitcast.41)
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[4,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P1]]), slice={[2:3], [0:8], [0:8]}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[4,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[FUSION:%[^ ]+]] = f16[8,8]{1,0} fusion([[P0]], [[P1]])
    ; CHECK-DAG:     kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK-DAG:     backend_config={
    ; CHECK-DAG:       "kind":"__custom_fusion",
    ; CHECK-DAG:       "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK-DAG:     }
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK:       ROOT {{.*}} = f16[8,8]{1,0} add([[FUSION]], [[B0]])
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmOperandsHaveMultipleUsers) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main.9 {
      %p0 = f16[2,8,8]{2,1,0} parameter(0)
      %p1 = f16[2,8,8]{2,1,0} parameter(1)
      %slice.13 = f16[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
      %slice.14 = f16[1,8,8]{2,1,0} slice(%p1), slice={[1:2], [0:8], [0:8]}
      %bitcast.42 = f16[8,8]{1,0} bitcast(%slice.14)

      %custom-call.0 = f16[8,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
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
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P1]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:     }
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P1]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmSlicingNotParameter) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main.9 {
      %p0 = f16[4,8,8]{2,1,0} parameter(0)
      %p1 = f16[2,8,8]{2,1,0} parameter(1)
      %slice.12 = f16[2,8,8]{2,1,0} slice(%p0), slice={[0:2], [0:8], [0:8]}
      %slice.13 = f16[1,8,8]{2,1,0} slice(%slice.12), slice={[1:2], [0:8], [0:8]}
      %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
      %slice.14 = f16[1,8,8]{2,1,0} slice(%p1), slice={[1:2], [0:8], [0:8]}
      %bitcast.42 = f16[8,8]{1,0} bitcast(%slice.14)

      %custom-call.1 = f16[8,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
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
      ROOT %res = f16[8,8]{1,0} add(%custom-call.1, %custom-call.1)
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P1]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[4,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[2,8,8]{2,1,0} slice([[P0]]), slice={[0:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK:       [[FUSION:%[^ ]+]] = f16[8,8]{1,0} fusion([[S0]], [[P1]])
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:       ROOT {{.*}} = f16[8,8]{1,0} add([[FUSION]], [[FUSION]])
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmNotContiguousSlice) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main.9 {
      %p0 = f16[2,8,8]{2,1,0} parameter(0)
      %p1 = f16[2,8,8]{2,1,0} parameter(1)
      %slice.13 = f16[1,4,6]{2,1,0} slice(%p0), slice={[1:2], [0:4], [0:6]}
      %bitcast.41 = f16[4,6]{1,0} bitcast(%slice.13)
      %slice.14 = f16[1,6,4]{2,1,0} slice(%p1), slice={[1:2], [0:6], [0:4]}
      %bitcast.42 = f16[6,4]{1,0} bitcast(%slice.14)

      ROOT %custom-call.1 = f16[4,4]{1,0} custom-call(%bitcast.41, %bitcast.42),
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
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"),
                            std::nullopt);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmNonNoOpInSliceChain) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main.9 {
      %p0 = f16[2,8,8]{2,1,0} parameter(0)
      %p1 = f16[2,8,8]{2,1,0} parameter(1)
      %slice.13 = f16[1,8,8]{2,1,0} slice(%p0), slice={[0:1], [0:8], [0:8]}
      %slice.14 = f16[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %add.0 = f16[1,8,8]{2,1,0} add(%slice.13, %slice.14)
      %bitcast.41 = f16[8,8]{1,0} bitcast(%add.0)
      %slice.15 = f16[1,8,8]{2,1,0} slice(%p1), slice={[0:1], [0:8], [0:8]}
      %slice.16 = f16[1,8,8]{2,1,0} slice(%p1), slice={[1:2], [0:8], [0:8]}
      %add.1 = f16[1,8,8]{2,1,0} add(%slice.15, %slice.16)
      %bitcast.42 = f16[8,8]{1,0} bitcast(%add.1)

      ROOT %custom-call.1 = f16[8,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
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
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"),
                            std::nullopt);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmDuplicateOperand) {
  const char* hlo = R"(
    HloModule test

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

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[P0:%[^ ]+]] = f32[200,100]{1,0} parameter(0)
    ; CHECK:       [[S0:%[^ ]+]] = f32[100,100]{1,0} slice([[P0]]), slice={[0:100], [0:100]}
    ; CHECK-NOT:   slice
    ; CHECK:       [[CC:%[^ ]+]] = (f32[100,100]{1,0}, s8[80000]{0}) custom-call([[S0]], [[S0]]),
    ; CHECK:         custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = (f32[100,100]{1,0}, s8[80000]{0}) fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmReverseOperandOrder) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main.9 {
      %p0 = f16[2,8,8]{2,1,0} parameter(1)
      %slice.13 = f16[1,8,8]{2,1,0} slice(%p0), slice={[0:1], [0:8], [0:8]}
      %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
      %p1 = f16[2,8,8]{2,1,0} parameter(0)
      %slice.14 = f16[1,8,8]{2,1,0} slice(%p1), slice={[1:2], [0:8], [0:8]}
      %bitcast.42 = f16[8,8]{1,0} bitcast(%slice.14)

      ROOT %custom-call.1 = f16[8,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
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
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[0:1], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P1]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK-DAG:   [[A0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[A1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = f16[8,8]{1,0} fusion([[A0]], [[A1]])
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmReverseOperandOrder2) {
  const char* hlo = R"(
    HloModule test

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
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P1]]), slice={[0:1], [0:8], [0:8]}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK-DAG:   [[A0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[A1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = f16[8,8]{1,0} fusion([[A0]], [[A1]])
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmOperandAliasingOutput) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main.9 {
      %p0 = (f32[100,100]{1,0}, f32[100,100]{1,0}) parameter(0)
      %get-tuple-element.287 = f32[100,100]{1,0} get-tuple-element(%p0), index=0
      %get-tuple-element.288 = f32[100,100]{1,0} get-tuple-element(%p0), index=1
      %concatenate.12 = f32[200,100]{1,0} concatenate(%get-tuple-element.287, %get-tuple-element.288), dimensions={0}
      %slice.30 = f32[100,100]{1,0} slice(%concatenate.12), slice={[16:116], [0:100]}
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
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P2:%[^ ]+]] = f32[100,100]{1,0} parameter(2)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f32[100,100]{1,0} parameter(1)
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f32[200,100]{1,0} parameter(0)
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f32[100,100]{1,0} slice([[P0]]), slice={[16:116], [0:100]}
    ; CHECK:       [[CC:%[^ ]+]] = (f32[100,100]{1,0}, s8[120000]{0}) custom-call([[P1]], [[S1]], [[P2]]),
    ; CHECK:         custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       [[P:%[^ ]+]] = (f32[100,100]{1,0}, f32[100,100]{1,0}) parameter(0)
    ; CHECK:       [[GTE0:%[^ ]+]] = f32[100,100]{1,0} get-tuple-element([[P]]), index=0
    ; CHECK:       [[GTE1:%[^ ]+]] = f32[100,100]{1,0} get-tuple-element([[P]]), index=1
    ; CHECK:       [[CONCAT:%[^ ]+]] = f32[200,100]{1,0} concatenate([[GTE0]], [[GTE1]]), dimensions={0}
    ; CHECK:       [[S:%[^ ]+]] = f32[100,100]{1,0} slice([[CONCAT]]), slice={[99:199], [0:100]}
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = (f32[100,100]{1,0}, s8[120000]{0}) fusion([[CONCAT]], [[GTE0]], [[S]])
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, SimpleGemmOperandsFromSameSlice) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main.9 {
      %p0 = f16[2,8,8]{2,1,0} parameter(0)
      %slice.13 = f16[1,8,8]{2,1,0} slice(%p0), slice={[0:1], [0:8], [0:8]}
      %bitcast.41 = f16[8,8]{1,0} bitcast(%slice.13)
      %bitcast.42 = f16[8,8]{0,1} bitcast(%slice.13)

      ROOT %custom-call.1 = f16[8,8]{1,0} custom-call(%bitcast.41, %bitcast.42),
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
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[0:1], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{0,1} bitcast([[S0]])
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK-DAG:   [[A0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = f16[8,8]{1,0} fusion([[A0]])
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

static absl::Status Memcpy(se::Stream* stream, ffi::AnyBuffer src,
                           ffi::AnyBuffer dst) {
  se::DeviceMemoryBase dst_mem = dst.device_memory();
  se::DeviceMemoryBase src_mem = src.device_memory();
  return stream->MemcpyD2D(&dst_mem, src_mem, src_mem.size());
}

XLA_FFI_DEFINE_HANDLER(kMemcpy, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()  // src
                           .Arg<ffi::AnyBuffer>()  // dst
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", "gpu",
                         kMemcpy);

TEST_F(DynamicSliceFusionRewriterTest, SimpleCustomCall) {
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
  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      ProgramShape program_shape,
      ProgramShape::FromProto(computation.proto().host_program_shape()));
  xla::HloModuleConfig hlo_config(program_shape,
                                  /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, xla::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[P0:%[^ ]+]] = f32[256]{0} parameter(0)
    ; CHECK:       [[S0:%[^ ]+]] = f32[128]{0} slice([[P0]]), slice={[0:128]}
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f32[128]{0} custom-call([[S0]]),
    ; CHECK:              custom_call_target="__xla_test$$memcpy",
    ; CHECK:              api_version=API_VERSION_TYPED_FFI
    ; CHECK:     }

    ; CHECK:     ENTRY %{{.*}} {
    ; CHECK:       [[C0:%[^ ]+]] = f32[] constant(42)
    ; CHECK:       [[BC:%[^ ]+]] = f32[256]{0} broadcast([[C0]])
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = f32[128]{0} fusion([[BC]])
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo->ToString(), DynamicSliceFusionRewriter("gpu"),
                            expected);
}

void Callback_Void(void* stream, void** buffers, const char* /*opaque*/,
                   size_t /*opaque_len*/) {}

XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_Void, "gpu");

TEST_F(DynamicSliceFusionRewriterTest, SimpleCustomCallLegacy) {
  XlaBuilder b(TestName());
  CustomCall(&b, "Callback_Void",
             /*operands=*/
             {Slice(Broadcast(ConstantR0WithType(&b, F32, 42.0), {256}), {0},
                    {128}, {1})},
             ShapeUtil::MakeShape(F32, {128}), /*opaque=*/"");
  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      ProgramShape program_shape,
      ProgramShape::FromProto(computation.proto().host_program_shape()));
  xla::HloModuleConfig hlo_config(program_shape,
                                  /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, xla::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));
  // TF_ASSERT_OK_AND_ASSIGN(
  //     HloSchedule schedule,
  //     ScheduleModule(hlo.get(), [](const BufferValue& buffer) {
  //       return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  //     }));
  // TF_CHECK_OK(hlo->set_schedule(std::move(schedule)));

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[P0:%[^ ]+]] = f32[256]{0} parameter(0)
    ; CHECK:       [[S0:%[^ ]+]] = f32[128]{0} slice([[P0]]), slice={[0:128]}
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f32[128]{0} custom-call([[S0]]),
    ; CHECK:              custom_call_target="Callback_Void"
    ; CHECK:     }

    ; CHECK:     ENTRY %{{.*}} {
    ; CHECK:       [[C0:%[^ ]+]] = f32[] constant(42)
    ; CHECK:       [[BC:%[^ ]+]] = f32[256]{0} broadcast([[C0]])
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = f32[128]{0} fusion([[BC]])
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo->ToString(), DynamicSliceFusionRewriter("gpu"),
                            expected);
}

TEST_F(DynamicSliceFusionRewriterTest, TupleSliceCustomCallLegacy) {
  XlaBuilder b(TestName());
  CustomCall(
      &b, "Callback_Void",
      /*operands=*/
      {
          Tuple(&b,
                {
                    Slice(Broadcast(ConstantR0WithType(&b, F32, 5), {8, 8}),
                          {0, 0}, {4, 8}, {1, 1}),
                    Broadcast(ConstantR0WithType(&b, F32, 2), {256}),
                }),
          Tuple(&b,
                {
                    Broadcast(ConstantR0WithType(&b, F32, 3), {1024}),
                    Broadcast(ConstantR0WithType(&b, F32, 4), {8}),
                }),
      },
      ShapeUtil::MakeShape(F32, {128}), /*opaque=*/"");
  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      ProgramShape program_shape,
      ProgramShape::FromProto(computation.proto().host_program_shape()));
  xla::HloModuleConfig hlo_config(program_shape,
                                  /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, xla::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));
  // TF_ASSERT_OK_AND_ASSIGN(
  //     HloSchedule schedule,
  //     ScheduleModule(hlo.get(), [](const BufferValue& buffer) {
  //       return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  //     }));
  // TF_CHECK_OK(hlo->set_schedule(std::move(schedule)));

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f32[8,8]{1,0} parameter(0)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f32[4,8]{1,0} slice([[P0]]), slice={[0:4], [0:8]}
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f32[256]{0} parameter(1)
    ; CHECK-DAG:   [[T0:%[^ ]+]] = (f32[4,8]{1,0}, f32[256]{0}) tuple([[S0]], [[P1]])
    ; CHECK-DAG:   [[P2:%[^ ]+]] = (f32[1024]{0}, f32[8]{0}) parameter(2)
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f32[128]{0} custom-call([[T0]], [[P2]]),
    ; CHECK:              custom_call_target="Callback_Void"
    ; CHECK:     }

    ; CHECK:     ENTRY %{{.*}} {
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = f32[128]{0} fusion(
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo->ToString(), DynamicSliceFusionRewriter("gpu"),
                            expected);
}

TEST_F(DynamicSliceFusionRewriterTest, TupledOutputCustomCallLegacy) {
  XlaBuilder b(TestName());
  auto custom_call = CustomCall(
      &b, "Callback_Void",
      /*operands=*/
      {
          Tuple(&b,
                {
                    Slice(Broadcast(ConstantR0WithType(&b, F32, 5), {8, 8}),
                          {0, 0}, {4, 8}, {1, 1}),
                    Broadcast(ConstantR0WithType(&b, F32, 2), {256}),
                }),
          Tuple(&b,
                {
                    Broadcast(ConstantR0WithType(&b, F32, 3), {1024}),
                    Broadcast(ConstantR0WithType(&b, F32, 4), {8}),
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
      }),
      /*opaque=*/"");
  Tuple(&b, {GetTupleElement(GetTupleElement(custom_call, 1), 0),
             GetTupleElement(custom_call, 2)});
  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      ProgramShape program_shape,
      ProgramShape::FromProto(computation.proto().host_program_shape()));
  xla::HloModuleConfig hlo_config(program_shape,
                                  /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, xla::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));
  // TF_ASSERT_OK_AND_ASSIGN(
  //     HloSchedule schedule,
  //     ScheduleModule(hlo.get(), [](const BufferValue& buffer) {
  //       return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  //     }));
  // TF_CHECK_OK(hlo->set_schedule(std::move(schedule)));

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P2:%[^ ]+]] = (f32[1024]{0}, f32[8]{0}) parameter(2)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f32[256]{0} parameter(1)
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f32[8,8]{1,0} parameter(0)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f32[4,8]{1,0} slice([[P0]]), slice={[0:4], [0:8]}
    ; CHECK-DAG:   [[T0:%[^ ]+]] = (f32[4,8]{1,0}, f32[256]{0}) tuple([[S0]], [[P1]])
    ; CHECK:       [[CC:%[^ ]+]] = (f32[8]{0}, (f32[128]{0}, f32[256]{0}), f32[1024]{0}, f32[4,8]{1,0}) custom-call([[T0]], [[P2]]),
    ; CHECK:              custom_call_target="Callback_Void"
    ; CHECK-DAG:   [[GTE0:%[^ ]+]] = f32[8]{0} get-tuple-element([[CC]]), index=0
    ; CHECK-DAG:   [[GTE1:%[^ ]+]] = (f32[128]{0}, f32[256]{0}) get-tuple-element([[CC]]), index=1
    ; CHECK-DAG:   [[GTE2:%[^ ]+]] = f32[128]{0} get-tuple-element([[GTE1]]), index=0
    ; CHECK-DAG:   [[GTE3:%[^ ]+]] = f32[256]{0} get-tuple-element([[GTE1]]), index=1
    ; CHECK-DAG:   [[T1:%[^ ]+]] = (f32[128]{0}, f32[256]{0}) tuple([[GTE2]], [[GTE3]])
    ; CHECK-DAG:   [[GTE4:%[^ ]+]] = f32[1024]{0} get-tuple-element([[CC]]), index=2
    ; CHECK-DAG:   [[GTE5:%[^ ]+]] = f32[4,8]{1,0} get-tuple-element([[CC]]), index=3
    ; CHECK:       ROOT {{.*}} = (f32[8]{0}, (f32[128]{0}, f32[256]{0}), f32[1024]{0}, f32[4,8]{1,0}) tuple([[GTE0]], [[T1]], [[GTE4]], [[GTE5]])
    ; CHECK:     }

    ; CHECK:     ENTRY %{{.*}} {
    ; CHECK:       [[FUSION:%[^ ]+]] = (f32[8]{0}, (f32[128]{0}, f32[256]{0}), f32[1024]{0}, f32[4,8]{1,0}) fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK-DAG:   [[GTE6:%[^ ]+]] = f32[1024]{0} get-tuple-element([[FUSION]]), index=2
    ; CHECK-DAG:   [[GTE7:%[^ ]+]] = (f32[128]{0}, f32[256]{0}) get-tuple-element([[FUSION]]), index=1
    ; CHECK-DAG:   [[GTE8:%[^ ]+]] = f32[128]{0} get-tuple-element([[GTE7]]), index=0
    ; CHECK:       ROOT {{.*}} = (f32[128]{0}, f32[1024]{0}) tuple([[GTE8]], [[GTE6]])
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo->ToString(), DynamicSliceFusionRewriter("gpu"),
                            expected);
}

TEST_F(DynamicSliceFusionRewriterTest, UnalignedSlice) {
  XlaBuilder b(TestName());
  CustomCall(
      &b, "Callback_Void",
      /*operands=*/
      {Slice(Broadcast(ConstantR0WithType(&b, S32, 42), {17}), {1}, {17}, {1})},
      ShapeUtil::MakeShape(S32, {16}), /*opaque=*/"");
  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      ProgramShape program_shape,
      ProgramShape::FromProto(computation.proto().host_program_shape()));
  xla::HloModuleConfig hlo_config(program_shape,
                                  /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, xla::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));
  // TF_ASSERT_OK_AND_ASSIGN(
  //     HloSchedule schedule,
  //     ScheduleModule(hlo.get(), [](const BufferValue& buffer) {
  //       return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  //     }));
  // TF_CHECK_OK(hlo->set_schedule(std::move(schedule)));

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo->ToString(), DynamicSliceFusionRewriter("gpu"),
                            std::nullopt);
}

TEST_F(DynamicSliceFusionRewriterTest, DynamicSimpleGemm) {
  const char* hlo = R"(
    HloModule test

    ENTRY main.9 {
      p0 = f16[2,8,8]{2,1,0} parameter(0)
      p1 = f16[2,8,8]{2,1,0} parameter(1)
      c1_s32 = s32[] constant(1)
      c0_s32 = s32[] constant(0)
      slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(p0, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.41 = f16[8,8]{1,0} bitcast(slice.13)
      slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(p1, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.42 = f16[8,8]{1,0} bitcast(slice.14)

      ROOT custom-call.1 = f16[8,8]{1,0} custom-call(bitcast.41, bitcast.42),
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
  )";

  const char* expected = R"(
    ; CHECK:     dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(3)
    ; CHECK-DAG:   [[C1:%[^ ]+]] = s32[] parameter(1)
    ; CHECK-DAG:   [[C0:%[^ ]+]] = s32[] parameter(2)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} dynamic-slice([[P0]], [[C1]], [[C0]], [[C0]]), dynamic_slice_sizes={1,8,8}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} dynamic-slice([[P1]], [[C1]], [[C0]], [[C0]]), dynamic_slice_sizes={1,8,8}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = f16[8,8]{1,0} fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"dynamic_address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, DynamicSimpleGemmWithWorkspace) {
  const char* hlo = R"(
    HloModule test

    ENTRY main.9 {
      p0 = f16[2,8,8]{2,1,0} parameter(0)
      p1 = f16[2,8,8]{2,1,0} parameter(1)
      c1_s32 = s32[] constant(1)
      c0_s32 = s32[] constant(0)
      slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(p0, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.41 = f16[8,8]{1,0} bitcast(slice.13)
      slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(p1, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.42 = f16[8,8]{1,0} bitcast(slice.14)

      ROOT custom-call.1 = (f16[8,8]{1,0}, s8[256]{0}) custom-call(bitcast.41, bitcast.42),
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
  )";

  const char* expected = R"(
    ; CHECK:     dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(3)
    ; CHECK-DAG:   [[C1:%[^ ]+]] = s32[] parameter(1)
    ; CHECK-DAG:   [[C0:%[^ ]+]] = s32[] parameter(2)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} dynamic-slice([[P0]], [[C1]], [[C0]], [[C0]]), dynamic_slice_sizes={1,8,8}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} dynamic-slice([[P1]], [[C1]], [[C0]], [[C0]]), dynamic_slice_sizes={1,8,8}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       [[CC:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0}) custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:       [[DOT:%[^ ]+]] = f16[8,8]{1,0} get-tuple-element([[CC]]), index=0
    ; CHECK:       [[WORKSPACE:%[^ ]+]] = s8[256]{0} get-tuple-element([[CC]]), index=1
    ; CHECK:       ROOT [[TUPLE:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0})
    ; CHECK:              tuple([[DOT]], [[WORKSPACE]])
    ; CHECK:     }


    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0}) fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"dynamic_address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, DynamicSimpleGemmWorkspaceIgnored) {
  const char* hlo = R"(
    HloModule test

    ENTRY main.9 {
      p0 = f16[2,8,8]{2,1,0} parameter(0)
      p1 = f16[2,8,8]{2,1,0} parameter(1)
      c1_s32 = s32[] constant(1)
      c0_s32 = s32[] constant(0)
      slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(p0, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.41 = f16[8,8]{1,0} bitcast(slice.13)
      slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(p1, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.42 = f16[8,8]{1,0} bitcast(slice.14)

      custom-call.1 = (f16[8,8]{1,0}, s8[256]{0}) custom-call(bitcast.41, bitcast.42),
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
      ROOT get-tuple-element.0 = f16[8,8]{1,0} get-tuple-element(custom-call.1), index=0
    }
  )";

  const char* expected = R"(
    ; CHECK:     dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(3)
    ; CHECK-DAG:   [[C1:%[^ ]+]] = s32[] parameter(1)
    ; CHECK-DAG:   [[C0:%[^ ]+]] = s32[] parameter(2)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} dynamic-slice([[P0]], [[C1]], [[C0]], [[C0]]), dynamic_slice_sizes={1,8,8}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} dynamic-slice([[P1]], [[C1]], [[C0]], [[C0]]), dynamic_slice_sizes={1,8,8}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       [[CC:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0}) custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:       [[DOT:%[^ ]+]] = f16[8,8]{1,0} get-tuple-element([[CC]]), index=0
    ; CHECK:       [[WORKSPACE:%[^ ]+]] = s8[256]{0} get-tuple-element([[CC]]), index=1
    ; CHECK:       ROOT [[TUPLE:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0})
    ; CHECK:              tuple([[DOT]], [[WORKSPACE]])
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       [[FUSION:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0}) fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"dynamic_address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:       ROOT [[DOT_MAIN:%[^ ]+]] = f16[8,8]{1,0} get-tuple-element([[FUSION]]), index=0
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, DynamicSimpleGemmNotRoot) {
  const char* hlo = R"(
    HloModule test

    ENTRY main.9 {
      p0 = f16[2,8,8]{2,1,0} parameter(0)
      p1 = f16[2,8,8]{2,1,0} parameter(1)
      c1_s32 = s32[] constant(1)
      c0_s32 = s32[] constant(0)
      slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(p0, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.41 = f16[8,8]{1,0} bitcast(slice.13)
      slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(p1, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.42 = f16[8,8]{1,0} bitcast(slice.14)

      custom-call.1 = f16[8,8]{1,0} custom-call(bitcast.41, bitcast.42),
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
      ROOT res = f16[8,8]{1,0} add(custom-call.1, custom-call.1)
    }
  )";

  const char* expected = R"(
    ; CHECK:     dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(3)
    ; CHECK-DAG:   [[C1:%[^ ]+]] = s32[] parameter(1)
    ; CHECK-DAG:   [[C0:%[^ ]+]] = s32[] parameter(2)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} dynamic-slice([[P0]], [[C1]], [[C0]], [[C0]]), dynamic_slice_sizes={1,8,8}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} dynamic-slice([[P1]], [[C1]], [[C0]], [[C0]]), dynamic_slice_sizes={1,8,8}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       [[FUSION:%[^ ]+]] = f16[8,8]{1,0} fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"dynamic_address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:       ROOT {{.*}} = f16[8,8]{1,0} add([[FUSION]], [[FUSION]])
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, DUSSimpleGemm) {
  const char* hlo = R"(
    HloModule test

    ENTRY main.9 {
      p0 = f16[1,8,8]{2,1,0} parameter(0)
      p1 = f16[1,8,8]{2,1,0} parameter(1)
      p2 = f16[4,8,8]{2,1,0} parameter(2)
      c1_s32 = s32[] constant(1)
      c0_s32 = s32[] constant(0)
      bitcast.41 = f16[8,8]{1,0} bitcast(p0)
      bitcast.42 = f16[8,8]{1,0} bitcast(p1)

      custom-call.1 = f16[8,8]{1,0} custom-call(bitcast.41, bitcast.42),
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
      bitcast.43 = f16[1,8,8]{2,1,0} bitcast(custom-call.1)
      ROOT dus = f16[4,8,8]{2,1,0} dynamic-update-slice(p2, bitcast.43, c1_s32, c0_s32, c0_s32)
    }
  )";

  const char* expected = R"(
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[8,8]{1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[8,8]{1,0} parameter(1)
    ; CHECK-DAG:   [[P2:%[^ ]+]] = f16[4,8,8]{2,1,0} parameter(2)
    ; CHECK-DAG:   [[C1:%[^ ]+]] = s32[] parameter(3)
    ; CHECK-DAG:   [[C0:%[^ ]+]] = s32[] parameter(4)
    ; CHECK-DAG:   [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[P0]], [[P1]]),
    ; CHECK-DAG:          custom_call_target="__cublas$gemm"
    ; CHECK-DAG:   [[BC:%[^ ]+]] = f16[1,8,8]{2,1,0} bitcast([[CC]])
    ; CHECK:       ROOT {{.*}} = f16[4,8,8]{2,1,0} dynamic-update-slice([[P2]], [[BC]], [[C1]], [[C0]], [[C0]])
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = f16[4,8,8]{2,1,0} fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"dynamic_address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, DUSSimpleGemmNotRoot) {
  const char* hlo = R"(
    HloModule test

    ENTRY main.9 {
      p0 = f16[2,8,8]{2,1,0} parameter(0)
      p1 = f16[2,8,8]{2,1,0} parameter(1)
      p2 = f16[4,8,8]{2,1,0} parameter(2)
      c1_s32 = s32[] constant(1)
      c0_s32 = s32[] constant(0)
      slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(p0, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.41 = f16[8,8]{1,0} bitcast(slice.13)
      slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(p1, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.42 = f16[8,8]{1,0} bitcast(slice.14)

      custom-call.1 = f16[8,8]{1,0} custom-call(bitcast.41, bitcast.42),
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
      bitcast.43 = f16[1,8,8]{2,1,0} bitcast(custom-call.1)
      dus = f16[4,8,8]{2,1,0} dynamic-update-slice(p2, bitcast.43, c1_s32, c0_s32, c0_s32)
      ROOT res = f16[4,8,8]{2,1,0} log(dus)
    }
  )";

  const char* expected = R"(
    ; CHECK:     dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(3)
    ; CHECK-DAG:   [[P2:%[^ ]+]] = f16[4,8,8]{2,1,0} parameter(4)
    ; CHECK-DAG:   [[C1:%[^ ]+]] = s32[] parameter(1)
    ; CHECK-DAG:   [[C0:%[^ ]+]] = s32[] parameter(2)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} dynamic-slice([[P0]], [[C1]], [[C0]], [[C0]]), dynamic_slice_sizes={1,8,8}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} dynamic-slice([[P1]], [[C1]], [[C0]], [[C0]]), dynamic_slice_sizes={1,8,8}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK-DAG:   [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[B0]], [[B1]]),
    ; CHECK-DAG:          custom_call_target="__cublas$gemm"
    ; CHECK-DAG:   [[BC:%[^ ]+]] = f16[1,8,8]{2,1,0} bitcast([[CC]])
    ; CHECK:       ROOT {{.*}} = f16[4,8,8]{2,1,0} dynamic-update-slice([[P2]], [[BC]], [[C1]], [[C0]], [[C0]])
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       [[FUSION:%[^ ]+]] = f16[4,8,8]{2,1,0} fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"dynamic_address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:       ROOT {{.*}} = f16[4,8,8]{2,1,0} log([[FUSION]])
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, DUSSimpleGemmWithWorkspace) {
  const char* hlo = R"(
    HloModule test

    ENTRY main.9 {
      p0 = f16[2,8,8]{2,1,0} parameter(0)
      p1 = f16[2,8,8]{2,1,0} parameter(1)
      p2 = f16[4,8,8]{2,1,0} parameter(2)
      c1_s32 = s32[] constant(1)
      c0_s32 = s32[] constant(0)
      slice.13 = f16[1,8,8]{2,1,0} dynamic-slice(p0, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.41 = f16[8,8]{1,0} bitcast(slice.13)
      slice.14 = f16[1,8,8]{2,1,0} dynamic-slice(p1, c1_s32, c0_s32, c0_s32), dynamic_slice_sizes={1,8,8}
      bitcast.42 = f16[8,8]{1,0} bitcast(slice.14)

      custom-call.1 = (f16[8,8]{1,0}, s8[256]{0}) custom-call(bitcast.41, bitcast.42),
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

    get-tuple-element.0 = f16[8,8]{1,0} get-tuple-element(custom-call.1), index=0
    bitcast.43 = f16[1,8,8]{2,1,0} bitcast(get-tuple-element.0)
    dus = f16[4,8,8]{2,1,0} dynamic-update-slice(p2, bitcast.43, c1_s32, c0_s32, c0_s32)
    get-tuple-element.1 = s8[256]{0} get-tuple-element(custom-call.1), index=1
    ROOT tuple = (f16[4,8,8]{2,1,0}, s8[256]{0}) tuple(dus, get-tuple-element.1)
    }
  )";

  const char* expected = R"(
    ; CHECK:     dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(3)
    ; CHECK-DAG:   [[P2:%[^ ]+]] = f16[4,8,8]{2,1,0} parameter(4)
    ; CHECK-DAG:   [[C1:%[^ ]+]] = s32[] parameter(1)
    ; CHECK-DAG:   [[C0:%[^ ]+]] = s32[] parameter(2)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} dynamic-slice([[P0]], [[C1]], [[C0]], [[C0]]), dynamic_slice_sizes={1,8,8}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} dynamic-slice([[P1]], [[C1]], [[C0]], [[C0]]), dynamic_slice_sizes={1,8,8}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       [[CC:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0}) custom-call([[B0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:       [[DOT:%[^ ]+]] = f16[8,8]{1,0} get-tuple-element([[CC]]), index=0
    ; CHECK:       [[BC:%[^ ]+]] = f16[1,8,8]{2,1,0} bitcast([[DOT]])
    ; CHECK:       [[DUS:%[^ ]+]] = f16[4,8,8]{2,1,0} dynamic-update-slice([[P2]], [[BC]], [[C1]], [[C0]], [[C0]])
    ; CHECK:       [[WORKSPACE:%[^ ]+]] = s8[256]{0} get-tuple-element([[CC]]), index=1
    ; CHECK:       ROOT [[TUPLE:%[^ ]+]] = (f16[4,8,8]{2,1,0}, s8[256]{0})
    ; CHECK:              tuple([[DUS]], [[WORKSPACE]])
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       [[FUSION:%[^ ]+]] = (f16[4,8,8]{2,1,0}, s8[256]{0}) fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"dynamic_address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:       [[DUS_MAIN:%[^ ]+]] = f16[4,8,8]{2,1,0} get-tuple-element([[FUSION]]), index=0
    ; CHECK:       [[WORKSPACE_MAIN:%[^ ]+]] = s8[256]{0} get-tuple-element([[FUSION]]), index=1
    ; CHECK:       ROOT {{.*}} = (f16[4,8,8]{2,1,0}, s8[256]{0})
    ; CHECK:              tuple([[DUS_MAIN]], [[WORKSPACE_MAIN]])
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, DUSSimpleGemmWorkspaceIgnored) {
  const char* hlo = R"(
    HloModule test

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

  const char* expected = R"(
    ; CHECK:     dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[8,8]{1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[8,8]{1,0} parameter(1)
    ; CHECK-DAG:   [[P2:%[^ ]+]] = f16[4,8,8]{2,1,0} parameter(2)
    ; CHECK-DAG:   [[C1:%[^ ]+]] = s32[] parameter(3)
    ; CHECK-DAG:   [[C0:%[^ ]+]] = s32[] parameter(4)
    ; CHECK-DAG:   [[CC:%[^ ]+]] = (f16[8,8]{1,0}, s8[256]{0}) custom-call([[P0]], [[P1]]),
    ; CHECK-DAG:          custom_call_target="__cublas$gemm"
    ; CHECK-DAG:   [[DOT:%[^ ]+]] = f16[8,8]{1,0} get-tuple-element([[CC]]), index=0
    ; CHECK-DAG:   [[BC:%[^ ]+]] = f16[1,8,8]{2,1,0} bitcast([[DOT]])
    ; CHECK-DAG:   [[DUS:%[^ ]+]] = f16[4,8,8]{2,1,0} dynamic-update-slice([[P2]], [[BC]], [[C1]], [[C0]], [[C0]])
    ; CHECK-DAG:   [[WORKSPACE:%[^ ]+]] = s8[256]{0} get-tuple-element([[CC]]), index=1
    ; CHECK:       ROOT [[TUPLE:%[^ ]+]] = (f16[4,8,8]{2,1,0}, s8[256]{0})
    ; CHECK:              tuple([[DUS]], [[WORKSPACE]])
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       [[FUSION:%[^ ]+]] = (f16[4,8,8]{2,1,0}, s8[256]{0}) fusion
    ; CHECK:         kind=kCustom, calls=%dynamic-slice-fusion,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"dynamic_address_computation","kernel_index":0}
    ; CHECK:         }
    ; CHECK:       ROOT [[DOT_MAIN:%[^ ]+]] = f16[4,8,8]{2,1,0} get-tuple-element([[FUSION]]), index=0
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, ReduceScatterDUSConstantOffset) {
  const char* hlo = R"(
  HloModule test, replica_count=2

  add {
    param_0 = f16[] parameter(0)
    param_1 = f16[] parameter(1)
    ROOT add.1 = f16[] add(param_0, param_1)
  }

  ENTRY main.9 {
    param_0 = f16[128,128]{1,0} parameter(0)
    param_1 = f16[128,128]{1,0} parameter(1)
    constant_20 = u32[] constant(20)
    constant_0 = u32[] constant(0)
    reduce-scatter = f16[64,128]{1,0} reduce-scatter(param_0), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
    ROOT loop_dynamic_update_slice_fusion = f16[128,128]{1,0} dynamic-update-slice(param_1, reduce-scatter, constant_20, constant_0)
  }
  )";

  const char* expected = R"(
  // CHECK: %dynamic-slice-fusion{{.+}} {
  // CHECK:   %[[RS:.+]] = f16[64,128]{1,0} reduce-scatter({{.+}})
  // CHECK:   ROOT %{{.+}} = f16[128,128]{1,0} dynamic-update-slice(%{{.+}}, %[[RS]], %{{.+}}, %{{.+}})
  // CHECK: }
  // CHECK: ENTRY {{.+}} {
  // CHECK-NOT: reduce-scatter
  // CHECK:   ROOT %{{.+}} = {{.+}} fusion(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}), kind=kCustom, calls=%dynamic-slice-fusion, {{.+}}"name":"dynamic_address_computation"
  // CHECK: }
  )";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, ReduceScatterDUSParameterOffset) {
  const char* hlo = R"(
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
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"),
                            std::nullopt);
}

TEST_F(DynamicSliceFusionRewriterTest, ReduceScatterDUSLoopIterationOffset) {
  const char* hlo = R"(
  HloModule jit_scan, replica_count=2

  add {
    param_0 = f32[] parameter(0)
    param_1 = f32[] parameter(1)
    ROOT add.6 = f32[] add(param_0, param_1)
  }

  Body {
    arg_tuple.1 = (s32[], f32[128,128]{1,0}, f32[128,128,128]{2,1,0}, f32[128,128]{1,0}) parameter(0)
    get-tuple-element.5 = s32[] get-tuple-element(arg_tuple.1), index=0
    constant.1 = s32[] constant(1)
    add.7 = s32[] add(get-tuple-element.5, constant.1)
    get-tuple-element.6 = f32[128,128]{1,0} get-tuple-element(arg_tuple.1), index=3
    get-tuple-element.7 = f32[128,128,128]{2,1,0} get-tuple-element(arg_tuple.1), index=2
    reduce-scatter.0 = f32[64,128]{1,0} reduce-scatter(get-tuple-element.6), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
    bitcast.63 = f32[1,64,128]{2,1,0} bitcast(reduce-scatter.0)
    constant.2 = s32[] constant(0)
    compare.4 = pred[] compare(get-tuple-element.5, constant.2), direction=LT
    constant.3 = s32[] constant(128)
    add.8 = s32[] add(get-tuple-element.5, constant.3)
    select.2 = s32[] select(compare.4, add.8, get-tuple-element.5)
    dynamic-update-slice.2 = f32[128,128,128]{2,1,0} dynamic-update-slice(get-tuple-element.7, bitcast.63, select.2, constant.2, constant.2)
    ROOT tuple.1 = tuple(add.7, get-tuple-element.6, dynamic-update-slice.2, get-tuple-element.6)
  } // Body

  Cond {
    arg_tuple.0 = (s32[], f32[128,128]{1,0}, f32[128,128,128]{2,1,0}, f32[128,128]{1,0}) parameter(0)
    get-tuple-element.4 = s32[] get-tuple-element(arg_tuple.0), index=0
    constant.0 = s32[] constant(128)
    ROOT compare.5 = pred[] compare(get-tuple-element.4, constant.0), direction=LT
  }

  ENTRY main.55 {
    Arg_2.3 = f32[128,128,128]{2,1,0} parameter(2)
    constant.4 = s32[] constant(0)
    Arg_1.2 = f32[128,128]{1,0} parameter(1)
    constant.5 = f32[] constant(0)
    broadcast.1 = f32[128,128,128]{2,1,0} broadcast(constant.5), dimensions={}
    Arg_0.1 = f32[128,128]{1,0} parameter(0)
    tuple = tuple(constant.4, Arg_1.2, broadcast.1, Arg_0.1)
    while = while(tuple), condition=Cond, body=Body, backend_config={"known_trip_count":{"n":"128"}}
    get-tuple-element.50 = f32[128,128]{1,0} get-tuple-element(while), index=1
    get-tuple-element.51 = f32[128,128,128]{2,1,0} get-tuple-element(while), index=2
    ROOT tuple.54 = (f32[128,128]{1,0}, f32[128,128,128]{2,1,0}) tuple(get-tuple-element.50, get-tuple-element.51)
  })";
  const char* expected = R"(
  // CHECK: %dynamic-slice-fusion{{.*}}{
  // CHECK:   {{.+}} = {{.*}}reduce-scatter({{.+}})
  // CHECK:   {{.+}} = {{.*}}dynamic-update-slice({{.+}})
  // CHECK: }
  // CHECK: Body{{.+}}{
  // CHECK-NOT: {{.+}} = {{.*}}reduce-scatter({{.+}})
  // CHECK:   {{.+}} = {{.+}}fusion({{.+}}), kind=kCustom, calls=%dynamic-slice-fusion{{.*}}"name":"dynamic_address_computation"
  // CHECK: }
  )";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, DUSSimpleGemmLoopIteration) {
  const char* hlo = R"(
  HloModule test

  %Body {
    param = (f16[1,8,8]{2,1,0}, f16[1,8,8]{2,1,0}, f16[4,8,8]{2,1,0}, u32[]) parameter(0)
    p0 = get-tuple-element(param), index=0
    p1 = get-tuple-element(param), index=1
    p2 = get-tuple-element(param), index=2
    loop_iter = get-tuple-element(param), index=3

    bitcast.41 = f16[8,8]{1,0} bitcast(p0)
    bitcast.42 = f16[8,8]{1,0} bitcast(p1)
    custom-call.1 = f16[8,8]{1,0} custom-call(bitcast.41, bitcast.42), custom_call_target="__cublas$gemm", backend_config={"gemm_backend_config":{
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
    bitcast.43 = f16[1,8,8]{2,1,0} bitcast(custom-call.1)
    c0 = u32[] constant(0)
    c_trip_count = u32[] constant(11)
    compare = pred[] compare(loop_iter, c0), direction=LT
    add = u32[] add(loop_iter, c_trip_count)
    offset = u32[] select(compare, add, loop_iter)
    dus = f16[4,8,8]{2,1,0} dynamic-update-slice(p2, bitcast.43, offset, c0, c0)
    c1 = u32[] constant(1)
    add2 = u32[] add(loop_iter, c1)
    ROOT tuple = tuple(p0, p1, dus, u32[] add2)
  }

  %Cond {
    %param.1 = (f16[1,8,8]{2,1,0}, f16[1,8,8]{2,1,0}, f16[4,8,8]{2,1,0}, u32[]) parameter(0)
    %i.1 = u32[] get-tuple-element(%param.1), index=3
    %trip_count = u32[] constant(11)
    ROOT %done = pred[] compare(u32[] %i.1, u32[] %trip_count), direction=LT
  }

  ENTRY %test {
    %p0.1 = f16[1,8,8]{2,1,0} parameter(0)
    %p1.1 = f16[1,8,8]{2,1,0} parameter(1)
    %p2.1 = f16[4,8,8]{2,1,0} parameter(2)
    %c0.1 = u32[] constant(0)
    %initial_tuple = tuple(%p0.1, %p1.1, %p2.1, u32[] %c0.1)
    ROOT %while = while(%initial_tuple), condition=%Cond, body=%Body, backend_config={"known_trip_count":{"n":"11"}}
  })";

  const char* expected = R"(
  // CHECK: %Body{{.+}}{
  // CHECK:   %[[PARAM:.+]] = {{.+}} parameter(0)
  // CHECK:   %[[LOOP_ITER:.+]] = u32[] get-tuple-element(%[[PARAM]]), index=3
  // CHECK:   %[[OFFSET:.+]] = u32[] select({{.+}})
  // CHECK:   %[[ADDRESS_COMPUTATION:.+]] = {{.+}} fusion({{.+}}, {{.+}}, {{.+}}, %[[OFFSET]], %{{.+}}), kind=kCustom, calls=%dynamic-slice-fusion, {{.+}}"name":"dynamic_address_computation"
  // CHECK:   ROOT %tuple = {{.+}} tuple(%{{.+}}, %{{.+}}, %[[ADDRESS_COMPUTATION]], %{{.+}})
  // CHECK: }
  // CHECK: ENTRY %test{{.+}}{
  // CHECK:   ROOT %{{.+}} = {{.+}} while(%{{.+}}), condition=%{{.+}}, body=%Body{{.*}}, backend_config={"known_trip_count":{"n":"11"}}
  }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, DUSSimpleGemmParameterOffset) {
  const char* hlo = R"(
      HloModule test

    ENTRY main.9 {
      p0 = f16[1,8,8]{2,1,0} parameter(0)
      p1 = f16[1,8,8]{2,1,0} parameter(1)
      p2 = f16[4,8,8]{2,1,0} parameter(2)
      p3 = s32[] parameter(3)
      c1_s32 = s32[] constant(1)
      c0_s32 = s32[] constant(0)
      bitcast.41 = f16[8,8]{1,0} bitcast(p0)
      bitcast.42 = f16[8,8]{1,0} bitcast(p1)

      custom-call.1 = f16[8,8]{1,0} custom-call(bitcast.41, bitcast.42),
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
      bitcast.43 = f16[1,8,8]{2,1,0} bitcast(custom-call.1)
      ROOT dus = f16[4,8,8]{2,1,0} dynamic-update-slice(p2, bitcast.43, p3, c0_s32, c0_s32)
    })";

  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"),
                            std::nullopt);
}

TEST_F(DynamicSliceFusionRewriterTest, DUSSimpleGemmLaxScan) {
  const char* hlo = R"(
  HloModule lax_scan

  // This is the HLO generated for the following:
  //
  // inp = jax.random.uniform(jax.random.key(128), (128, 128, 128))
  // init = jnp.identity(128)
  // ans = jax.lax.scan(lambda carry, x : (init, x@carry), init, inp)

  Body {
    arg_tuple.15 = (s32[], f32[128,128]{1,0}, f32[128,128,128]{2,1,0}, f32[128,128,128]{2,1,0}, f32[128,128]{1,0}) parameter(0)
    get-tuple-element.16 = s32[] get-tuple-element(arg_tuple.15), index=0
    constant.21 = s32[] constant(1)
    add.2 = s32[] add(get-tuple-element.16, constant.21)
    get-tuple-element.30 = f32[128,128]{1,0} get-tuple-element(arg_tuple.15), index=4
    get-tuple-element.18 = f32[128,128,128]{2,1,0} get-tuple-element(arg_tuple.15), index=2
    get-tuple-element.19 = f32[128,128,128]{2,1,0} get-tuple-element(arg_tuple.15), index=3
    constant.23 = s32[] constant(0)
    compare.2 = pred[] compare(get-tuple-element.16, constant.23), direction=LT
    constant.22 = s32[] constant(128)
    add.3 = s32[] add(get-tuple-element.16, constant.22)
    select.1 = s32[] select(compare.2, add.3, get-tuple-element.16)
    dynamic-slice.1 = f32[1,128,128]{2,1,0} dynamic-slice(get-tuple-element.19, select.1, constant.23, constant.23), dynamic_slice_sizes={1,128,128}
    bitcast.72 = f32[128,128]{1,0} bitcast(dynamic-slice.1)
    get-tuple-element.17 = f32[128,128]{1,0} get-tuple-element(arg_tuple.15), index=1
    custom-call.1 = (f32[128,128]{1,0}, s8[131072]{0}) custom-call(bitcast.72, get-tuple-element.17), custom_call_target="__cublas$gemm"
    get-tuple-element = f32[128,128]{1,0} get-tuple-element(custom-call.1), index=0
    bitcast.77 = f32[1,128,128]{2,1,0} bitcast(get-tuple-element)
    dynamic-update-slice.1 = f32[128,128,128]{2,1,0} dynamic-update-slice(get-tuple-element.18, bitcast.77, select.1, constant.23, constant.23)
    ROOT tuple.38 = tuple(add.2, get-tuple-element.30, dynamic-update-slice.1, get-tuple-element.19, get-tuple-element.30)
  } // Body

  Cond {
    arg_tuple.40 = (s32[], f32[128,128]{1,0}, f32[128,128,128]{2,1,0}, f32[128,128,128]{2,1,0}, f32[128,128]{1,0}) parameter(0)
    get-tuple-element.41 = s32[] get-tuple-element(arg_tuple.40), index=0
    constant.46 = s32[] constant(128)
    ROOT compare.3 = pred[] compare(get-tuple-element.41, constant.46), direction=LT
  }

  ENTRY main {
    constant.4 = s32[] constant(0)
    Arg_1.2 = f32[128,128]{1,0} parameter(1)
    constant.5 = f32[] constant(0)
    broadcast.1 = f32[128,128,128]{2,1,0} broadcast(constant.5), dimensions={}
    Arg_2.3 = f32[128,128,128]{2,1,0} parameter(2)
    Arg_0.1 = f32[128,128]{1,0} parameter(0)
    tuple.7 = tuple(constant.4, Arg_1.2, broadcast.1, Arg_2.3, Arg_0.1)
    while.48 = while(tuple.7), condition=Cond, body=Body, backend_config={"known_trip_count":{"n":"128"}}
    get-tuple-element.50 = f32[128,128]{1,0} get-tuple-element(while.48), index=1
    get-tuple-element.51 = f32[128,128,128]{2,1,0} get-tuple-element(while.48), index=2
    ROOT tuple.54 = (f32[128,128]{1,0}, f32[128,128,128]{2,1,0}) tuple(get-tuple-element.50, get-tuple-element.51)
  } // main.55

)";
  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  const char* expected = R"(
  // CHECK: %dynamic-slice-fusion{{.*}} {{.+}} {
  // CHECK:   {{.+}} = {{.+}}dynamic-slice
  // CHECK:   {{.+}} = {{.+}}custom-call
  // CHECK:   {{.+}} = {{.+}}dynamic-update-slice
  // CHECK: }
  // CHECK: %Body{{.+}}{
  // CHECK:   %[[PARAM:.+]] = {{.+}} parameter(0)
  // CHECK:   %[[LOOP_ITER:.+]] = s32[] get-tuple-element(%[[PARAM]]), index=0
  // CHECK:   %[[OFFSET:.+]] = s32[] select({{.+}})
  // CHECK:   %[[ADDRESS_COMPUTATION:.+]] = {{.+}} fusion({{.+}}, %[[OFFSET]], %{{.+}}), kind=kCustom, calls=%dynamic-slice-fusion{{.+}}"name":"dynamic_address_computation"
  // CHECK:   %[[GTE:.+]] = {{.+}} get-tuple-element(%[[ADDRESS_COMPUTATION]]), index=0
  // CHECK:   ROOT %{{.+}} = {{.+}} tuple(%{{.+}}, %[[GTE]], %{{.+}})
  // CHECK: }
  // CHECK: ENTRY %main{{.+}}{
  // CHECK:   %{{.+}} = {{.+}} while(%{{.+}}), condition=%{{.+}}, body=%Body{{.*}}, backend_config={"known_trip_count":{"n":"128"}}
  // CHECK: }
  )";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, ReduceScatterSlice) {
  const char* hlo = R"(
  HloModule jit_slice, replica_count=2

  add {
    a = s32[] parameter(0)
    b = s32[] parameter(1)
    ROOT add = add(a,b)
  }

  ENTRY %main.9 {
    p0 = s32[2,8,32]{2,1,0} parameter(0)
    slice = s32[1,8,32]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:32]}
    bc = s32[8,32]{1,0} bitcast(%slice)
    ROOT rs = s32[4,32] reduce-scatter(bc), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
  })";
  const char* expected = R"(
  // CHECK: dynamic-slice-fusion{{.*}} {
  // CHECK:   %[[p0:.+]] = {{.+}} parameter(0)
  // CHECK:   %[[slice:.+]] = {{.+}} slice(%[[p0]]), slice={[1:2], [0:8], [0:32]}
  // CHECK:   %[[bc:.+]] = {{.+}} bitcast(%[[slice]])
  // CHECK:   ROOT {{.+}} = {{.+}} reduce-scatter(%[[bc]])
  // CHECK: }
  )";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest, ReduceScatterDynamicSlice) {
  const char* hlo = R"(
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
    bc = s32[8,32]{1,0} bitcast(%slice)
    ROOT rs = s32[4,32] reduce-scatter(bc), channel_id=64, replica_groups={{0,1}}, use_global_device_ids=true, dimensions={0}, to_apply=add
  })";
  const char* expected = R"(
  // CHECK: add
  // CHECK: dynamic-slice-fusion{{.*}} {
  // CHECK:   %[[p0:.+]] = {{.+}} parameter(0)
  // CHECK:   %[[slice:.+]] = {{.+}} dynamic-slice(%[[p0]], {{.+}}), dynamic_slice_sizes={1,8,32}
  // CHECK:   %[[bc:.+]] = {{.+}} bitcast(%[[slice]])
  // CHECK:   ROOT {{.+}} = {{.+}} reduce-scatter(%[[bc]])
  // CHECK: }
  // CHECK: ENTRY
  )";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), expected);
}

TEST_F(DynamicSliceFusionRewriterTest,
       OffsetAsFunctionOfInductionVariableShouldFuse) {
  const char* hlo = R"(
    HloModule test, replica_count=2
    add {
      a = s32[] parameter(0)
      b = s32[] parameter(1)
      ROOT add = s32[] add(a, b)
    }
    body {
      param.1 = (s32[], s32[32,32], s32[32,32]) parameter(0)
      iter.1 = s32[] get-tuple-element(param.1), index=0
      src = s32[32,32] get-tuple-element(param.1), index=1
      dest = s32[32,32] get-tuple-element(param.1), index=2

      // offset as a function of only the loop induction variable.
      add.1 = s32[] add(iter.1, iter.1)
      c3 = s32[] constant(3)
      multiply.1 = s32[] multiply(add.1, c3)
      c16 = s32[] constant(16)
      offset.1 = s32[] subtract(multiply.1, c16)

      c0 = s32[] constant(0)
      rs = s32[16,32] reduce-scatter(src), dimensions={0}, replica_groups={{0,1}}, to_apply=add
      dus = s32[32,32] dynamic-update-slice(dest, rs, offset.1, c0)
      c1 = s32[] constant(1)
      add.2 = s32[] add(iter.1, c1)
      ROOT tuple = tuple(add.2, src, dus)
    }
    condition {
      param.2 = (s32[], s32[32,32], s32[32,32]) parameter(0)
      iter.2 = s32[] get-tuple-element(param.2), index=0
      c16 = s32[] constant(16)
      ROOT compare = pred[] compare(iter.2, c16), direction=LT
    }
    ENTRY main {
      src = s32[32,32] parameter(0)
      dest = s32[32,32] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[32,32], s32[32,32]) tuple(c0, src, dest)
      ROOT while = (s32[], s32[32,32], s32[32,32]) while(tuple), body=body, condition=condition
    }
  )";
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), R"(
    // CHECK: dynamic-slice-fusion
    // CHECK:    %[[rs:.+]] = {{.+}} reduce-scatter({{.+}})
    // CHECK:    ROOT %[[dus:.+]] = {{.+}} dynamic-update-slice({{.+}})
    // CHECK: body
    // CHECK:   %[[fusion:.+]] = {{.+}} fusion({{.+}}), kind=kCustom, calls=%dynamic-slice-fusion,
    // CHECK-SAME:  "fusion_backend_config":{"kind":"__custom_fusion","custom_fusion_config":{"name":"dynamic_address_computation"
  )");
}

TEST_F(DynamicSliceFusionRewriterTest,
       ReduceScatterDynamicSliceAndDUSMultipleBuffersGetsFused) {
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
      rs1 = s32[8,8] bitcast(ds1)
      rs2 = s32[8,8] bitcast(ds2)
      rs = (s32[4,8], s32[4,8]) reduce-scatter(rs1, rs2), dimensions={0}, replica_groups={{0,1}}, to_apply=add
      reduce-scatter1 = s32[4,8] get-tuple-element(rs), index=0
      reduce-scatter2 = s32[4,8] get-tuple-element(rs), index=1
      bitcast1 = s32[1,4,8] bitcast(reduce-scatter1)
      bitcast2 = s32[1,4,8] bitcast(reduce-scatter2)
      dus1 = s32[8,4,8] dynamic-update-slice(dst1, bitcast1, iter.1, c0, c0)
      dus2 = s32[8,4,8] dynamic-update-slice(dst2, bitcast2, iter.1, c0, c0)
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

  // Checking for 2 dynamic-slices, their uses in reduce-scatter and their
  // update via dus inside the fusion.
  RunAndFilecheckHloRewrite(hlo, DynamicSliceFusionRewriter("gpu"), R"(
    // CHECK: dynamic-slice-fusion
    // CHECK-DAG:   %[[ds1:.+]] = {{.+}} dynamic-slice({{.+}})
    // CHECK-DAG:   %[[ds2:.+]] = {{.+}} dynamic-slice({{.+}})
    // CHECK-DAG:   %[[bc1:.+]] = {{.+}} bitcast(%[[ds1]])
    // CHECK-DAG:   %[[bc2:.+]] = {{.+}} bitcast(%[[ds2]])
    // CHECK-DAG:   %[[rs:.+]] = {{.+}} reduce-scatter(%[[bc1]], %[[bc2]])
    // CHECK-DAG:   %[[rs1:.+]] = {{.+}} get-tuple-element(%[[rs]]), index=0
    // CHECK-DAG:   %[[rs2:.+]] = {{.+}} get-tuple-element(%[[rs]]), index=1
    // CHECK-DAG:   %[[bc3:.+]] = {{.+}} bitcast(%[[rs1]])
    // CHECK-DAG:   %[[bc4:.+]] = {{.+}} bitcast(%[[rs2]])
    // CHECK-DAG:   %[[dus1:.+]] = {{.+}} dynamic-update-slice({{.+}}, %[[bc3]], {{.+}})
    // CHECK-DAG:   %[[dus2:.+]] = {{.+}} dynamic-update-slice({{.+}}, %[[bc4]], {{.+}})

    // CHECK: body
  )");
}

}  // namespace xla::gpu
