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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/sol_gpu_cost_model.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

// Define CostType to distinguish between collective and node costs
enum class CostType { kCollectiveTime, kNodeCost };

struct EstimatorTestCase {
  std::string test_name;
  std::string module_string;
  HloOpcode opcode_to_find;  // Renamed from opcode
  CostType cost_type;        // Added CostType
  absl::Duration expected_latency;
};

// Dummy fallback `LatencyEstimator` for `SolLatencyEstimator` instantiation.
class DummyLatencyEstimator : public LatencyEstimator {
 public:
  TimeCost GetLatencyBetween(const HloGraphNode& from,
                             const HloGraphNode& target) const override {
    return 0;
  }
  TimeCost NodeCost(const HloInstruction* instr) const override { return 0; }
  int CyclesPerMicrosecond() const override { return 0; }
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

  absl::Duration ComputeNodeCost(const HloInstruction& instr,
                                 const HloComputation* computation) {
    std::unique_ptr<SolLatencyEstimator> estimator =
        *SolLatencyEstimator::Create(
            scheduler_config_, std::make_unique<DummyLatencyEstimator>(),
            gpu_device_info_, shape_size_fn_, computation);
    LatencyEstimator::TimeCost cost_val = estimator->NodeCost(&instr);
    return absl::Microseconds(static_cast<int64_t>(cost_val));
  }

  HloCostAnalysis::ShapeSizeFunction shape_size_fn_;
  const se::DeviceDescription gpu_device_info_;
  const SolGPUCostModel::Config sol_flags_;
  SchedulerConfig scheduler_config_;
};

TEST_P(SolLatencyEstimatorTest, TestLatencyEstimation) {
  EstimatorTestCase test_case = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(test_case.module_string));

  HloInstruction* instr = hlo_query::FindInstruction(
      module->entry_computation(), test_case.opcode_to_find);
  CHECK_NE(instr, nullptr);
  absl::Duration actual_time_us;
  if (test_case.cost_type == CostType::kCollectiveTime) {
    actual_time_us =
        absl::Trunc(ComputeCollectiveTime(*instr), absl::Microseconds(1));
  } else if (test_case.cost_type == CostType::kNodeCost) {
    actual_time_us = ComputeNodeCost(*instr, module->entry_computation());
  } else {
    LOG(FATAL) << "Unreachable.";
  }

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
      /*opcode_to_find=*/HloOpcode::kAllGatherStart,
      /*cost_type=*/CostType::kCollectiveTime,
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
      /*opcode_to_find=*/HloOpcode::kAllGatherStart,
      /*cost_type=*/CostType::kCollectiveTime,
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
      /*opcode_to_find=*/HloOpcode::kAllGatherStart,
      /*cost_type=*/CostType::kCollectiveTime,
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
      /*opcode_to_find=*/HloOpcode::kAsyncStart,
      /*cost_type=*/CostType::kCollectiveTime,
      /*expected_latency=*/absl::Microseconds(18895),
  };

  EstimatorTestCase matmul_bf16_1024_4096_512 = {
      /*test_name=*/"matmul_bf16_1024_4096_512",
      /*module_string=*/R"(
HloModule bf16_matmul
ENTRY main {
  lhs = bf16[1024,512] parameter(0)
  rhs = bf16[512,4096] parameter(1)
  ROOT dot_op = bf16[1024,4096] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
})",
      /*opcode_to_find=*/HloOpcode::kDot,
      /*cost_type=*/CostType::kNodeCost,
      /*expected_latency=*/absl::Microseconds(12),
  };

  EstimatorTestCase matmul_f32_batch4_256_1024_256 = {
      /*test_name=*/"matmul_f32_batch4_256_1024_256",
      /*module_string=*/R"(
HloModule f32_batched_matmul
ENTRY main {
  lhs = f32[4,256,256] parameter(0)
  rhs = f32[4,256,1024] parameter(1)
  ROOT dot_op = f32[4,256,1024] dot(lhs, rhs), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={1}
})",
      /*opcode_to_find=*/HloOpcode::kDot,
      /*cost_type=*/CostType::kNodeCost,
      /*expected_latency=*/absl::Microseconds(10),
  };
  EstimatorTestCase triton_matmul_bf16_batch1_1024_1024_1024 = {
      /*test_name=*/"triton_matmul_bf16_batch1_1024_1024_1024",
      /*module_string=*/R"(
HloModule m

comp {
  p0 = bf16[1024,1024] parameter(0)
  p1 = bf16[1024,1024] parameter(1)
  ROOT dot = bf16[1024,1024] dot(p0,p1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[1024,1024] parameter(0)
  p1 = bf16[1024,1024] parameter(1)
  ROOT _ =  bf16[1024,1024] fusion(p0,p1),
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
})",
      /*opcode_to_find=*/HloOpcode::kFusion,
      /*cost_type=*/CostType::kNodeCost,
      /*expected_latency=*/absl::Microseconds(8),
  };

  EstimatorTestCase cublas_matmul_bf16_batch1_1024_1024_1024 = {
      /*test_name=*/"cublas_matmul_bf16_batch1_1024_1024_1024",
      /*module_string=*/R"(
HloModule m

ENTRY e {
  p0 = bf16[1024,1024] parameter(0)
  p1 = bf16[1024,1024] parameter(1)
  ROOT _ =  (bf16[1024,1024], s8[2097152]{0}) custom-call(p0,p1),
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
})",
      /*opcode_to_find=*/HloOpcode::kCustomCall,
      /*cost_type=*/CostType::kNodeCost,
      /*expected_latency=*/absl::Microseconds(8),
  };

  EstimatorTestCase simple_fusion_elementwise = {
      /*test_name=*/"simple_fusion_elementwise",
      /*module_string=*/R"(
HloModule m

comp {
  p0 = bf16[1024,1024] parameter(0)
  p1 = bf16[1024,1024] parameter(1)
  ROOT ret = bf16[1024,1024] add(p0,p1)
}

ENTRY e {
  p0 = bf16[1024,1024] parameter(0)
  p1 = bf16[1024,1024] parameter(1)
  ROOT _ =  bf16[1024,1024] fusion(p0,p1), kind=kInput, calls=comp
})",
      /*opcode_to_find=*/HloOpcode::kFusion,
      /*cost_type=*/CostType::kNodeCost,
      /*expected_latency=*/absl::Microseconds(8),
  };

  return {all_gather_intra_host,
          all_gather_inter_host_pairwise,
          all_gather_all_ranks,
          reduce_scatter_all_ranks,
          matmul_bf16_1024_4096_512,
          matmul_f32_batch4_256_1024_256,
          triton_matmul_bf16_batch1_1024_1024_1024,
          cublas_matmul_bf16_batch1_1024_1024_1024,
          simple_fusion_elementwise};
}

INSTANTIATE_TEST_SUITE_P(SolLatencyEstimatorTests, SolLatencyEstimatorTest,
                         ValuesIn(GetSolLatencyEstimatorTestCases()),
                         [](const TestParamInfo<EstimatorTestCase>& info) {
                           return info.param.test_name;
                         });

}  // namespace
}  // namespace xla::gpu
