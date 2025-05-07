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

#include "xla/service/gpu/model/sol_latency_estimator.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/sol_gpu_cost_model.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

struct EstimatorTestCase {
  std::string test_name;
  std::string module_string;
  HloOpcode opcode;
  absl::Duration expected_latency;
};

class SolLatencyEstimatorTest : public HloHardwareIndependentTestBase,
                                public WithParamInterface<EstimatorTestCase> {
 protected:
  SolLatencyEstimatorTest()
      : shape_size_fn_(HloCostAnalysis::DefaultShapeSize),
        gpu_device_info_(TestGpuDeviceInfo::RTXA6000DeviceInfo(
            se::CudaComputeCapability(9, 0))),
        sol_flags_({
            /*nccl_op_launch_time=*/absl::Microseconds(100),
            /*nic_speed_gbps=*/100,
            /*chunk_prep_time=*/absl::Microseconds(100),
            /*rtt=*/absl::Microseconds(100),
            /*gpus_per_node=*/8,
            /*chunk_size_bytes=*/4 * 1024 * 1024,
        }) {}

  absl::Duration ComputeCollectiveTime(const HloInstruction& instr) {
    return SolLatencyEstimator::ComputeCollectiveTime(
        instr, gpu_device_info_, shape_size_fn_, sol_flags_);
  }

  HloCostAnalysis::ShapeSizeFunction shape_size_fn_;
  const se::DeviceDescription gpu_device_info_;
  const SolGPUCostModel::Config sol_flags_;
};

TEST_P(SolLatencyEstimatorTest, TestLatencyEstimation) {
  EstimatorTestCase test_case = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(test_case.module_string));
  HloInstruction* instr =
      hlo_query::FindInstruction(module->entry_computation(), test_case.opcode);
  absl::Duration actual_time_us =
      absl::Trunc(ComputeCollectiveTime(*instr), absl::Microseconds(1));
  EXPECT_EQ(actual_time_us, test_case.expected_latency);
}

std::vector<EstimatorTestCase> GetSolLatencyEstimatorTestCases() {
  EstimatorTestCase all_gather_intra_host = {
      /*test_name=*/"all_gather_intra_host",
      /*module_string=*/R"(
HloModule m, num_partitions=16

ENTRY main {
  p = bf16[16000,1000] parameter(0)
  ag-start = (bf16[16000,1000], bf16[16000,8000]) all-gather-start(p),
    replica_groups={{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}},
    channel_id=1,
    use_global_device_ids=true,
    dimensions={1}
  ROOT ag-done = bf16[16000,8000] all-gather-done(ag-start)
})",
      /*opcode=*/HloOpcode::kAllGatherStart,
      /*expected_latency=*/GpuPerformanceModelBase::kNcclKernelLaunchOverhead,
  };

  EstimatorTestCase all_gather_inter_host_pairwise = {
      /*test_name=*/"all_gather_intra_host_pairwise",
      /*module_string=*/R"(
HloModule m, num_partitions=16

ENTRY main {
  p = bf16[16000,4000] parameter(0)
  ag-start = (bf16[16000,4000], bf16[16000,8000]) all-gather-start(p),
    replica_groups={{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}},
    channel_id=1,
    use_global_device_ids=true,
    dimensions={1}
  ROOT ag-done = bf16[16000,8000] all-gather-done(ag-start)
})",
      /*opcode=*/HloOpcode::kAllGatherStart,
      /*expected_latency=*/absl::Microseconds(5445),
  };

  EstimatorTestCase all_gather_all_ranks = {
      /*test_name=*/"all_gather_all_ranks",
      /*module_string=*/R"(
HloModule m, num_partitions=16

ENTRY main {
  p = bf16[16000,500] parameter(0)
  ag-start = (bf16[16000,500], bf16[16000,8000]) all-gather-start(p),
    replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}},
    channel_id=1,
    use_global_device_ids=true,
    dimensions={1}
  ROOT ag-done = bf16[16000,8000] all-gather-done(ag-start)
})",
      /*opcode=*/HloOpcode::kAllGatherStart,
      /*expected_latency=*/absl::Microseconds(2324),
  };

  EstimatorTestCase reduce_scatter_all_ranks = {
      /*test_name=*/"reduce_scatter_all_ranks",
      /*module_string=*/R"(
HloModule m, num_partitions=128

add {
  param_0 = bf16[] parameter(0)
  param_1 = bf16[] parameter(1)
  ROOT t = bf16[] add(param_0, param_1)
}

async_comp {
  param_3 = bf16[8192,128256] parameter(0)
  ROOT r = bf16[64,128256] reduce-scatter(param_3),
    dimensions={0},
    to_apply=add,
    replica_groups=[1,128]<=[128],
    channel_id=1,
    use_global_device_ids=true
}

ENTRY main {
  p = bf16[8192,128256] parameter(0)
  rs-start = ((bf16[8192,128256]), bf16[64,128256]) async-start(p), calls=async_comp
  ROOT rs-done = bf16[64,128256] async-done(rs-start)
})",
      /*opcode=*/HloOpcode::kAsyncStart,
      /*expected_latency=*/absl::Microseconds(18895),
  };

  return {
      all_gather_intra_host,
      all_gather_inter_host_pairwise,
      all_gather_all_ranks,
      reduce_scatter_all_ranks,
  };
}

INSTANTIATE_TEST_SUITE_P(SolLatencyEstimatorTests, SolLatencyEstimatorTest,
                         ValuesIn(GetSolLatencyEstimatorTestCases()),
                         [](const TestParamInfo<EstimatorTestCase>& info) {
                           return info.param.test_name;
                         });

}  // namespace
}  // namespace xla::gpu
