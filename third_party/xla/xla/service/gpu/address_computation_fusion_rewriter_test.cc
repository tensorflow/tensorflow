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

#include "xla/service/gpu/address_computation_fusion_rewriter.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/buffer_value.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/hlo_memory_scheduler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

#define PLATFORM "GPU"
namespace xla::gpu {

class AddressComputationFusionRewriterTest : public HloTestBase {};

TEST_F(AddressComputationFusionRewriterTest, SimpleGemm) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
    ; CHECK:     %address-computation {{.*}} {
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
    ; CHECK:         kind=kCustom, calls=%address-computation,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation"}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            expected, [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(AddressComputationFusionRewriterTest, SimpleGemmWithWorkspace) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
    ; CHECK:     %address-computation {{.*}} {
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
    ; CHECK:         kind=kCustom, calls=%address-computation,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation"}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            expected, [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(AddressComputationFusionRewriterTest, SimpleGemmNotRoot) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
    ; CHECK:     %address-computation {{.*}} {
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
    ; CHECK:         kind=kCustom, calls=%address-computation,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation"}
    ; CHECK:         }
    ; CHECK:       ROOT {{.*}} = f16[8,8]{1,0} add([[FUSION]], [[FUSION]])
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            expected, [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(AddressComputationFusionRewriterTest,
       SimpleGemmOperandHasMultipleUsers) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
      ROOT %res = f16[8,8]{1,0} add(%custom-call.1, %bitcast.41)
    }
  )";

  const char* expected = R"(
    ; CHECK:     %address-computation {{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[8,8]{1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P1]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B1:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S1]])
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f16[8,8]{1,0} custom-call([[P0]], [[B1]]),
    ; CHECK:              custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   [[S0:%[^ ]+]] = f16[1,8,8]{2,1,0} slice([[P0]]), slice={[1:2], [0:8], [0:8]}
    ; CHECK-DAG:   [[B0:%[^ ]+]] = f16[8,8]{1,0} bitcast([[S0]])
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f16[2,8,8]{2,1,0} parameter(1)
    ; CHECK:       [[FUSION:%[^ ]+]] = f16[8,8]{1,0} fusion([[B0]], [[P1]])
    ; CHECK:         kind=kCustom, calls=%address-computation,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation"}
    ; CHECK:         }
    ; CHECK:       ROOT {{.*}} = f16[8,8]{1,0} add([[FUSION]], [[B0]])
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            expected, [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(AddressComputationFusionRewriterTest,
       SimpleGemmOperandsHaveMultipleUsers) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            std::nullopt);
}

TEST_F(AddressComputationFusionRewriterTest, SimpleGemmSlicingNotParameter) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
    ; CHECK:     %address-computation {{.*}} {
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
    ; CHECK:         kind=kCustom, calls=%address-computation,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation"}
    ; CHECK:         }
    ; CHECK:       ROOT {{.*}} = f16[8,8]{1,0} add([[FUSION]], [[FUSION]])
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            expected, [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(AddressComputationFusionRewriterTest, SimpleGemmNotContiguousSlice) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            std::nullopt);
}

TEST_F(AddressComputationFusionRewriterTest, SimpleGemmNonNoOpInSliceChain) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            std::nullopt);
}

TEST_F(AddressComputationFusionRewriterTest, SimpleGemmDuplicateOperand) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
    ; CHECK:     %address-computation {{.*}} {
    ; CHECK:       [[P0:%[^ ]+]] = f32[200,100]{1,0} parameter(0)
    ; CHECK:       [[S0:%[^ ]+]] = f32[100,100]{1,0} slice([[P0]]), slice={[0:100], [0:100]}
    ; CHECK-NOT:   slice
    ; CHECK:       [[CC:%[^ ]+]] = (f32[100,100]{1,0}, s8[80000]{0}) custom-call([[S0]], [[S0]]),
    ; CHECK:         custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = (f32[100,100]{1,0}, s8[80000]{0}) fusion
    ; CHECK:         kind=kCustom, calls=%address-computation,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation"}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            expected, [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(AddressComputationFusionRewriterTest, SimpleGemmReverseOperandOrder) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
    ; CHECK:     %address-computation {{.*}} {
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
    ; CHECK:         kind=kCustom, calls=%address-computation,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation"}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            expected, [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(AddressComputationFusionRewriterTest, SimpleGemmReverseOperandOrder2) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
    ; CHECK:     %address-computation {{.*}} {
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
    ; CHECK:         kind=kCustom, calls=%address-computation,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation"}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            expected, [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(AddressComputationFusionRewriterTest, SimpleGemmOperandAliasingOutput) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
    ; CHECK:     %address-computation {{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f32[100,100]{1,0} parameter(0)
    ; CHECK-DAG:   [[P1:%[^ ]+]] = f32[100,100]{1,0} parameter(1)
    ; CHECK-DAG:   [[P2:%[^ ]+]] = f32[200,100]{1,0} parameter(2)
    ; CHECK-DAG:   [[S1:%[^ ]+]] = f32[100,100]{1,0} slice([[P2]]), slice={[16:116], [0:100]}
    ; CHECK:       [[CC:%[^ ]+]] = (f32[100,100]{1,0}, s8[120000]{0}) custom-call([[P0]], [[S1]], [[P1]]),
    ; CHECK:         custom_call_target="__cublas$gemm"
    ; CHECK:     }

    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       [[P:%[^ ]+]] = (f32[100,100]{1,0}, f32[100,100]{1,0}) parameter(0)
    ; CHECK:       [[GTE0:%[^ ]+]] = f32[100,100]{1,0} get-tuple-element([[P]]), index=0
    ; CHECK:       [[GTE1:%[^ ]+]] = f32[100,100]{1,0} get-tuple-element([[P]]), index=1
    ; CHECK:       [[CONCAT:%[^ ]+]] = f32[200,100]{1,0} concatenate([[GTE0]], [[GTE1]]), dimensions={0}
    ; CHECK:       [[S:%[^ ]+]] = f32[100,100]{1,0} slice([[CONCAT]]), slice={[99:199], [0:100]}
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = (f32[100,100]{1,0}, s8[120000]{0}) fusion([[GTE0]], [[S]], [[CONCAT]])
    ; CHECK:         kind=kCustom, calls=%address-computation,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation"}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            expected, [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(AddressComputationFusionRewriterTest, SimpleGemmOperandsFromSameSlice) {
  const char* hlo = R"(
    HloModule test, is_scheduled=true

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
    ; CHECK:     %address-computation {{.*}} {
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
    ; CHECK:         kind=kCustom, calls=%address-computation,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation"}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo, AddressComputationFusionRewriter(PLATFORM),
                            expected, [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

static absl::Status Memcpy(const ServiceExecutableRunOptions* run_options,
                           ffi::BufferBase src, ffi::BufferBase dst) {
  return run_options->stream()->MemcpyD2D(
      &dst.data, src.data,
      absl::c_accumulate(src.dimensions, 1.0, std::multiplies<int64_t>()) *
          sizeof(float));
}

XLA_FFI_DEFINE_HANDLER(kMemcpy, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ServiceExecutableRunOptions>()
                           .Arg<ffi::BufferBase>()  // src
                           .Arg<ffi::BufferBase>()  // dst
);
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", PLATFORM,
                         kMemcpy);

TEST_F(AddressComputationFusionRewriterTest, SimpleCustomCall) {
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
  xla::HloModuleConfig hlo_config(
      xla::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_address_computation_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, xla::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(hlo.get(), [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
      }));
  TF_CHECK_OK(hlo->set_schedule(std::move(schedule)));

  const char* expected = R"(
    ; CHECK:     %address-computation {{.*}} {
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
    ; CHECK:         kind=kCustom, calls=%address-computation,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation"}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo->ToString(),
                            AddressComputationFusionRewriter(PLATFORM),
                            expected, [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

void Callback_Void(se::gpu::GpuStreamHandle stream, void** buffers,
                   const char* /*opaque*/, size_t /*opaque_len*/) {}

XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_Void, PLATFORM);

TEST_F(AddressComputationFusionRewriterTest, SimpleCustomCallLegacy) {
  XlaBuilder b(TestName());
  CustomCall(&b, "Callback_Void",
             /*operands=*/
             {Slice(Broadcast(ConstantR0WithType(&b, F32, 42.0), {256}), {0},
                    {128}, {1})},
             ShapeUtil::MakeShape(F32, {128}), /*opaque=*/"");
  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  xla::HloModuleConfig hlo_config(
      xla::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_address_computation_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, xla::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(hlo.get(), [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
      }));
  TF_CHECK_OK(hlo->set_schedule(std::move(schedule)));

  const char* expected = R"(
    ; CHECK:     %address-computation {{.*}} {
    ; CHECK:       [[P0:%[^ ]+]] = f32[256]{0} parameter(0)
    ; CHECK:       [[S0:%[^ ]+]] = f32[128]{0} slice([[P0]]), slice={[0:128]}
    ; CHECK:       ROOT [[CC:%[^ ]+]] = f32[128]{0} custom-call([[S0]]),
    ; CHECK:              custom_call_target="Callback_Void"
    ; CHECK:     }

    ; CHECK:     ENTRY %{{.*}} {
    ; CHECK:       [[C0:%[^ ]+]] = f32[] constant(42)
    ; CHECK:       [[BC:%[^ ]+]] = f32[256]{0} broadcast([[C0]])
    ; CHECK:       ROOT [[FUSION:%[^ ]+]] = f32[128]{0} fusion([[BC]])
    ; CHECK:         kind=kCustom, calls=%address-computation,
    ; CHECK:         backend_config={
    ; CHECK:           "kind":"__custom_fusion",
    ; CHECK:           "custom_fusion_config":{"name":"address_computation"}
    ; CHECK:         }
    ; CHECK:     }
  )";

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo->ToString(),
                            AddressComputationFusionRewriter(PLATFORM),
                            expected, [](HloModule* module) {
                              EXPECT_TRUE(module->has_schedule());
                              TF_CHECK_OK(module->schedule().Verify());
                            });
}

TEST_F(AddressComputationFusionRewriterTest, UnalignedSlice) {
  XlaBuilder b(TestName());
  CustomCall(
      &b, "Callback_Void",
      /*operands=*/
      {Slice(Broadcast(ConstantR0WithType(&b, S32, 42), {17}), {1}, {17}, {1})},
      ShapeUtil::MakeShape(S32, {16}), /*opaque=*/"");
  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  xla::HloModuleConfig hlo_config(
      xla::ProgramShape(computation.proto().host_program_shape()),
      /*ignore_layouts=*/false);
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_address_computation_fusion(false);
  hlo_config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto hlo, xla::HloModule::CreateFromProto(
                                        computation.proto(), hlo_config));
  TF_ASSERT_OK_AND_ASSIGN(
      HloSchedule schedule,
      ScheduleModule(hlo.get(), [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
      }));
  TF_CHECK_OK(hlo->set_schedule(std::move(schedule)));

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  RunAndFilecheckHloRewrite(hlo->ToString(),
                            AddressComputationFusionRewriter(PLATFORM),
                            std::nullopt);
}

}  // namespace xla::gpu
