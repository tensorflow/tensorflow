/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/model/matmul_ptable_stats_collection.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

constexpr const char* kFile = "profiles.pbtxt";

using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::Property;
using ::testing::Test;

DeviceHloInstructionProfiles TestProfiles(
    const se::DeviceDescription& device_info) {
  constexpr char perf_table[] = R"pb(
    entries {
      key: "sm_89"
      value {
        entries {
          instruction {
            opcode: "dot"
            shape {
              element_type: BF16
              dimensions: 1
              dimensions: 1024
              dimensions: 1024
            }
            dot_dimension_numbers {
              lhs_contracting_dimensions: 2
              rhs_contracting_dimensions: 1
              lhs_batch_dimensions: 0
              rhs_batch_dimensions: 0
            }
            id: 2
            operand_ids: 0
            operand_ids: 1
          }
          operands {
            name: "lhs"
            opcode: "parameter"
            shape {
              element_type: BF16
              dimensions: 1
              dimensions: 1024
              dimensions: 1024
            }
          }
          operands {
            name: "rhs"
            opcode: "parameter"
            shape {
              element_type: BF16
              dimensions: 1
              dimensions: 1024
              dimensions: 1024
            }
            parameter_number: 1
            id: 1
          }
          clock_cycles: 1410000000
        }
      }
    }
  )pb";
  DeviceHloInstructionProfiles profiles;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(perf_table, &profiles));
  return profiles;
}

class MatmulStatsCollectionTest : public Test {
 public:
  explicit MatmulStatsCollectionTest()
      : device_info_(TestGpuDeviceInfo::RTXA6000DeviceInfo()),
        profiles_path_(tsl::io::JoinPath(tsl::testing::TmpDir(), kFile)) {}

  void SetUp() override {
    CHECK_OK(tsl::WriteTextProto(tsl::Env::Default(), profiles_path_,
                                 TestProfiles(device_info_)));
  }

 protected:
  const se::DeviceDescription device_info_;
  const std::string profiles_path_;
};

TEST_F(MatmulStatsCollectionTest,
       CollectsMatmulPerfTableDataForGemmCustomCalls) {
  absl::string_view hlo = R"(
    HloModule m

    ENTRY e {
      p0 = bf16[1024,1024] parameter(0)
      p1 = bf16[1024,1024] parameter(1)
      ROOT dot =  (bf16[1024,1024], s8[2097152]{0}) custom-call(p0,p1),
        custom_call_target="__cublas$gemm",
        backend_config={
          "operation_queue_id":"0",
          "wait_on_operation_queues":[],
          "gemm_backend_config":{
            "alpha_real":1,
            "beta":1,
            "dot_dimension_numbers": {
              "lhs_contracting_dimensions":["1"],
              "rhs_contracting_dimensions":["1"],
              "lhs_batch_dimensions":[],
              "rhs_batch_dimensions":[]
            }
          }
        }
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, MatmulPerfTableStatsCollection(profiles_path_, device_info_)
                        .Run(module.get()));

  VLOG(1) << module->ToString();

  EXPECT_FALSE(changed);
  EXPECT_THAT(
      module->entry_computation()
          ->root_instruction()
          ->backend_config<GpuBackendConfig>()
          ->reification_cost(),
      ElementsAre(Property(&ReificationCost::exec_time_us,
                           DoubleNear(1000000, /*max_abs_error=*/0.01))));
}

TEST_F(MatmulStatsCollectionTest,
       CollectsMatmulPerfTableDataForTritonFusionConfig) {
  absl::string_view hlo = R"(
    HloModule m

    comp {
      p0 = bf16[1024,1024] parameter(0)
      p1 = bf16[1024,1024] parameter(1)
      ROOT _ = bf16[1024,1024] dot(p0,p1),
        lhs_contracting_dims={0},
        rhs_contracting_dims={1}
    }

    ENTRY e {
      p0 = bf16[1024,1024] parameter(0)
      p1 = bf16[1024,1024] parameter(1)
      ROOT triton_gemm =  bf16[1024,1024] fusion(p0,p1),
        kind=kCustom,
        calls=comp,
        backend_config={
          "operation_queue_id":"0",
          "wait_on_operation_queues":[],
          "fusion_backend_config": {
            "kind":"__triton_gemm",
            "triton_gemm_config":{
              "block_m":"128",
              "block_n":"128",
              "block_k":"64",
              "split_k":"1",
              "num_stages":"1",
              "num_warps":"8",
              "num_ctas":"1"
            }
          },
        }
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, MatmulPerfTableStatsCollection(profiles_path_, device_info_)
                        .Run(module.get()));

  VLOG(1) << module->ToString();

  EXPECT_FALSE(changed);
  EXPECT_EQ(module->entry_computation()
                ->root_instruction()
                ->backend_config<GpuBackendConfig>()
                ->reification_cost_size(),
            0);
}

TEST_F(MatmulStatsCollectionTest,
       CollectsMatmulGEMMCostModelDataForTritonFusionConfig) {
  absl::string_view hlo = R"(
    HloModule m

    comp {
      p0 = bf16[1024,1024] parameter(0)
      p1 = bf16[1024,1024] parameter(1)
      ROOT _ = bf16[1024,1024] dot(p0,p1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }

    ENTRY e {
      p0 = bf16[1024,1024] parameter(0)
      p1 = bf16[1024,1024] parameter(1)
      ROOT triton_gemm =  bf16[1024,1024] fusion(p0,p1),
        kind=kCustom,
        calls=comp,
        backend_config={
          "operation_queue_id":"0",
          "wait_on_operation_queues":[],
          "fusion_backend_config": {
            "kind":"__triton_gemm",
            "triton_gemm_config":{
              "block_m":"128",
              "block_n":"128",
              "block_k":"64",
              "split_k":"1",
              "num_stages":"1",
              "num_warps":"8",
              "num_ctas":"1"
            }
          },
        }
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, MatmulPerfTableStatsCollection(profiles_path_, device_info_)
                        .Run(module.get()));

  VLOG(1) << module->ToString();

  EXPECT_FALSE(changed);
  EXPECT_THAT(module->entry_computation()
                  ->root_instruction()
                  ->backend_config<GpuBackendConfig>()
                  ->reification_cost(),
              ElementsAre(Property(&ReificationCost::exec_time_us,
                                   DoubleNear(199, /*max_abs_error=*/1))));
}

}  // namespace
}  // namespace xla::gpu
