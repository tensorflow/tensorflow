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
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/collective_interpolator.h"
#include "xla/service/gpu/model/sol_gpu_cost_model.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/restricted/hlo_test_base_legacy.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::TestParamInfo;
using ::testing::ValuesIn;
using ::testing::WithParamInterface;

// Define CostType to distinguish between collective and node costs
enum class CostType { kCollectiveTime, kNodeCost, kEdgeCost };

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
  int CyclesPerMicrosecond() const override { return 1; }
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
        }),
        collective_interpolator_(*CollectiveInterpolator::Create(
            sol_flags_.gpus_per_node, gpu_device_info_,
            /*analysis=*/nullptr)) {}

  absl::StatusOr<absl::Duration> ComputeCollectiveTime(
      const HloInstruction& instr) {
    return SolLatencyEstimator::ComputeCollectiveTime(
        instr, gpu_device_info_, shape_size_fn_, sol_flags_,
        collective_interpolator_.get());
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

  absl::Duration GetLatencyBetween(const HloGraphNode& from,
                                   const HloGraphNode& target,
                                   const HloComputation* computation) {
    std::unique_ptr<SolLatencyEstimator> estimator =
        *SolLatencyEstimator::Create(
            scheduler_config_, std::make_unique<DummyLatencyEstimator>(),
            gpu_device_info_, shape_size_fn_, computation);
    LatencyEstimator::TimeCost cost_val =
        estimator->GetLatencyBetween(from, target);
    return absl::Microseconds(static_cast<int64_t>(cost_val));
  }

  HloCostAnalysis::ShapeSizeFunction shape_size_fn_;
  const se::DeviceDescription gpu_device_info_;
  const SolGPUCostModel::Config sol_flags_;
  SchedulerConfig scheduler_config_;
  std::unique_ptr<CollectiveInterpolator> collective_interpolator_;
};

TEST_P(SolLatencyEstimatorTest, TestLatencyEstimation) {
  EstimatorTestCase test_case = GetParam();
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(test_case.module_string));

  HloInstruction* instr = hlo_query::FindInstruction(
      module->entry_computation(), test_case.opcode_to_find);
  ASSERT_NE(instr, nullptr);
  absl::Duration actual_time_us;
  if (test_case.cost_type == CostType::kCollectiveTime) {
    ASSERT_OK_AND_ASSIGN(absl::Duration time_us, ComputeCollectiveTime(*instr));
    actual_time_us = absl::Trunc(time_us, absl::Microseconds(1));
  } else if (test_case.cost_type == CostType::kNodeCost) {
    actual_time_us = ComputeNodeCost(*instr, module->entry_computation());
  } else if (test_case.cost_type == CostType::kEdgeCost) {
    actual_time_us = GetLatencyBetween(
        HloGraphNode(instr, /*original_position=*/-1),
        HloGraphNode(instr->users().front(), /*original_position=*/-1),
        module->entry_computation());
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
      /*expected_latency=*/absl::Microseconds(695),
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

  EstimatorTestCase reduce_scatter_intra_host = {
      /*test_name=*/"reduce_scatter_intra_host",
      /*module_string=*/R"(
HloModule m, num_partitions=8

add {
  param_0 = bf16[] parameter(0)
  param_1 = bf16[] parameter(1)
  ROOT t = bf16[] add(param_0, param_1)
}

async_comp {
  param_3 = bf16[8192,128256] parameter(0)
  ROOT r = bf16[1024,128256] reduce-scatter(param_3),
    dimensions={0},
    to_apply=add,
    replica_groups=[1,8]<=[8],
    channel_id=1,
    use_global_device_ids=true
}

ENTRY main {
  p = bf16[8192,128256] parameter(0)
  rs-start = ((bf16[8192,128256]), bf16[1024,128256]) async-start(p), calls=async_comp
  ROOT rs-done = bf16[1024,128256] async-done(rs-start)
})",
      /*opcode_to_find=*/HloOpcode::kAsyncStart,
      /*cost_type=*/CostType::kEdgeCost,
      /*expected_latency=*/absl::Microseconds(5716),
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
      "fusion_backend_config": {
        "kind":"__triton_gemm",
        "triton_gemm_config":{
          "block_m":"128",
          "block_n":"128",
          "block_k":"64",
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
  ROOT _ =  (bf16[1024,1024], s8[2097152]) custom-call(p0,p1),
    custom_call_target="__cublas$lt$matmul",
    backend_config={
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

  EstimatorTestCase cublaslt_matmul_mixed_fp8_batch1_1024_1024_1024 = {
      /*test_name=*/"cublas_matmul_mixed_fp8_batch1_1024_1024_1024",
      /*module_string=*/R"(
HloModule m

ENTRY e {
  p0 = f8e5m2[1024,1024] parameter(0)
  p1 = f8e4m3fn[1024,1024] parameter(1)
  ROOT _ =  (bf16[1024,1024], s8[2097152]) custom-call(p0,p1),
    custom_call_target="__cublas$lt$matmul$f8",
    backend_config={
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

  EstimatorTestCase cublaslt_matmul_f8e5m2_f8e4m3fn_batch1_1024_1024_1024 = {
      /*test_name=*/"cublaslt_matmul_f8e5m2_f8e4m3fn_batch1_1024_1024_1024",
      /*module_string=*/R"(
HloModule m

ENTRY e {
  p0 = f8e5m2[1024,1024] parameter(0)
  p1 = f8e4m3fn[1024,1024] parameter(1)
  ROOT _ =  (bf16[1024,1024], s8[2097152]) custom-call(p0,p1),
    custom_call_target="__cublas$lt$matmul$f8",
    backend_config={
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

  EstimatorTestCase cublaslt_matmul_f8e4m3fn_f8e5m2_batch1_1024_1024_1024 = {
      /*test_name=*/"cublaslt_matmul_f8e4m3fn_f8e5m2_batch1_1024_1024_1024",
      /*module_string=*/R"(
HloModule m

ENTRY e {
  p0 = f8e4m3fn[1024,1024] parameter(0)
  p1 = f8e5m2[1024,1024] parameter(1)
  ROOT _ =  (bf16[1024,1024], s8[2097152]) custom-call(p0,p1),
    custom_call_target="__cublas$lt$matmul$f8",
    backend_config={
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

  EstimatorTestCase cublaslt_matmul_f8e4m3fn_f8e4m3fn_batch1_1024_1024_1024 = {
      /*test_name=*/"cublaslt_matmul_f8e4m3fn_f8e4m3fn_batch1_1024_1024_1024",
      /*module_string=*/R"(
HloModule m

ENTRY e {
  p0 = f8e4m3fn[1024,1024] parameter(0)
  p1 = f8e4m3fn[1024,1024] parameter(1)
  ROOT _ =  (bf16[1024,1024], s8[2097152]) custom-call(p0,p1),
    custom_call_target="__cublas$lt$matmul$f8",
    backend_config={
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

  EstimatorTestCase pallas_custom_call = {
      /*test_name=*/"pallas_custom_call",
      /*module_string=*/R"(
HloModule m

ENTRY e {
  p0 = bf16[128,128] parameter(0)
  ROOT _ =  bf16[128,128] custom-call(p0),
    custom_call_target="mosaic_gpu_v2",
    frontend_attributes={latency_metadata="30000"}
})",
      /*opcode_to_find=*/HloOpcode::kCustomCall,
      /*cost_type=*/CostType::kNodeCost,
      /*expected_latency=*/absl::Microseconds(30),
  };

  EstimatorTestCase noop = {
      /*test_name=*/"noop",
      /*module_string=*/R"(
HloModule m

ENTRY e {
  ROOT _ = f16[] constant(3.14)
})",
      /*opcode_to_find=*/HloOpcode::kConstant,
      /*cost_type=*/CostType::kNodeCost,
      /*expected_latency=*/absl::ZeroDuration(),
  };
  // Test for CollectivePermuteCostModelType::kIntraPartitionTwoWayHasNonMutual
  EstimatorTestCase collective_permute_intra_host_ring_shift = {
      /*test_name=*/"collective_permute_intra_host_ring_shift",
      /*module_string=*/R"(
HloModule m, num_partitions=4

ENTRY main {
  %param.2 = f32[262144,1024] parameter(0), sharding={devices=[4,1]<=[4]}
  %collective-permute-start = (f32[262144,1024], f32[262144,1024]) collective-permute-start(%param.2), channel_id=1, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  ROOT %collective-permute-done = f32[262144,1024] collective-permute-done(%collective-permute-start)
})",
      /*opcode_to_find=*/HloOpcode::kCollectivePermuteStart,
      /*cost_type=*/CostType::kEdgeCost,
      /*expected_latency=*/absl::Microseconds(3706),
  };

  // Test for CollectivePermuteCostModelType::kIntraPartitionTwoWayAllMutual
  EstimatorTestCase collective_permute_intra_host_bidirectional = {
      /*test_name=*/"collective_permute_intra_host_bidirectional",
      /*module_string=*/R"(
HloModule m, num_partitions=4

ENTRY main {
  %param.2 = f32[262144,1024] parameter(0), sharding={devices=[4,1]<=[4]}
  %collective-permute-start = (f32[262144,1024], f32[262144,1024]) collective-permute-start(%param.2), channel_id=1, source_target_pairs={{0,1},{1,0},{2,3},{3,2}}
  ROOT %collective-permute-done = f32[262144,1024] collective-permute-done(%collective-permute-start)
})",
      /*opcode_to_find=*/HloOpcode::kCollectivePermuteStart,
      /*cost_type=*/CostType::kEdgeCost,
      /*expected_latency=*/absl::Microseconds(3696),
  };

  // Test for CollectivePermuteCostModelType::kIntraPartitionOneWay
  EstimatorTestCase collective_permute_intra_host_one_way = {
      /*test_name=*/"collective_permute_intra_host_one_way",
      /*module_string=*/R"(
HloModule m, num_partitions=4

ENTRY main {
  %param.2 = f32[262144,1024] parameter(0), sharding={devices=[4,1]<=[4]}
  %collective-permute-start = (f32[262144,1024], f32[262144,1024]) collective-permute-start(%param.2), channel_id=1, source_target_pairs={{0,1},{2,3}}
  ROOT %collective-permute-done = f32[262144,1024] collective-permute-done(%collective-permute-start)
})",
      /*opcode_to_find=*/HloOpcode::kCollectivePermuteStart,
      /*cost_type=*/CostType::kEdgeCost,
      /*expected_latency=*/absl::Microseconds(3961),
  };

  EstimatorTestCase collective_permute_inter_host_global = {
      /*test_name=*/"collective_permute_inter_host_global",
      /*module_string=*/R"(
HloModule m, num_partitions=16

ENTRY main {
  %param.2 = f32[262144,1024] parameter(0)
  %collective-permute-start = (f32[262144,1024], f32[262144,1024]) collective-permute-start(%param.2), channel_id=1,
      source_target_pairs={{0,15},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6},{8,7},{9,8},{10,9},{11,10},{12,11},{13,12},{14,13},{15,14}}
  ROOT %collective-permute-done = f32[262144,1024] collective-permute-done(%collective-permute-start)
})",
      /*opcode_to_find=*/HloOpcode::kCollectivePermuteStart,
      /*cost_type=*/CostType::kEdgeCost,
      /*expected_latency=*/absl::Microseconds(27816),
  };

  EstimatorTestCase collective_permute_inter_host_rail_aligned_bidirection = {
      /*test_name=*/"collective_permute_inter_host_rail_aligned_bidirection",
      /*module_string=*/R"(
HloModule m, num_partitions=16

ENTRY main {
  %param.2 = f32[262144,1024] parameter(0)
  %collective-permute-start = (f32[262144,1024], f32[262144,1024]) collective-permute-start(%param.2), channel_id=1,
      source_target_pairs={{0,8},{8,0},{1,9},{9,1},{2,10},{10,2},{3,11},{11,3},{4,12},{12,4},{5,13},{13,5},{6,14},{14,6},{7,15},{15,7}}
  ROOT %collective-permute-done = f32[262144,1024] collective-permute-done(%collective-permute-start)
})",
      /*opcode_to_find=*/HloOpcode::kCollectivePermuteStart,
      /*cost_type=*/CostType::kEdgeCost,
      /*expected_latency=*/absl::Microseconds(27816),
  };

  EstimatorTestCase collective_permute_inter_host_rail_aligned_unidirection = {
      /*test_name=*/"collective_permute_inter_host_rail_aligned_unidirection",
      /*module_string=*/R"(
HloModule m, num_partitions=16

ENTRY main {
  %param.2 = f32[262144,1024] parameter(0)
  %collective-permute-start = (f32[262144,1024], f32[262144,1024]) collective-permute-start(%param.2), channel_id=1,
      source_target_pairs={{0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}}
  ROOT %collective-permute-done = f32[262144,1024] collective-permute-done(%collective-permute-start)
})",
      /*opcode_to_find=*/HloOpcode::kCollectivePermuteStart,
      /*cost_type=*/CostType::kEdgeCost,
      /*expected_latency=*/absl::Microseconds(27816),
  };

  return {collective_permute_intra_host_ring_shift,
          collective_permute_intra_host_bidirectional,
          collective_permute_intra_host_one_way,
          collective_permute_inter_host_global,
          collective_permute_inter_host_rail_aligned_bidirection,
          collective_permute_inter_host_rail_aligned_unidirection,
          all_gather_intra_host,
          all_gather_inter_host_pairwise,
          all_gather_all_ranks,
          reduce_scatter_all_ranks,
          reduce_scatter_intra_host,
          matmul_bf16_1024_4096_512,
          matmul_f32_batch4_256_1024_256,
          triton_matmul_bf16_batch1_1024_1024_1024,
          cublas_matmul_bf16_batch1_1024_1024_1024,
          simple_fusion_elementwise,
          pallas_custom_call,
          noop};
}

INSTANTIATE_TEST_SUITE_P(SolLatencyEstimatorTests, SolLatencyEstimatorTest,
                         ValuesIn(GetSolLatencyEstimatorTestCases()),
                         [](const TestParamInfo<EstimatorTestCase>& info) {
                           return info.param.test_name;
                         });

TEST_F(HloHardwareIndependentTestBase, CollectiveCostModelDispatching) {
  const auto shape_size_fn = HloCostAnalysis::DefaultShapeSize;
  const auto gpu_info = TestGpuDeviceInfo::H100SXMDeviceInfo();
  const SolGPUCostModel::Config sol_flags = {
      absl::Microseconds(100), 100, absl::Microseconds(100),
      absl::Microseconds(100), 8,   4 * 1024 * 1024};
  auto interpolator =
      *CollectiveInterpolator::Create(sol_flags.gpus_per_node, gpu_info,
                                      /*analysis=*/nullptr);

  // NVLink domain collective should use CollectiveInterpolator.
  ASSERT_OK_AND_ASSIGN(auto nvl_module, ParseAndReturnVerifiedModule(R"(
HloModule m, num_partitions=16
ENTRY main {
  p = bf16[8,16000,1000] parameter(0)
  ROOT a2a = bf16[8,16000,1000] all-to-all(p),
    replica_groups={{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15}},
    channel_id=1, dimensions={0}
})"));
  HloInstruction* nvl_instr = hlo_query::FindInstruction(
      nvl_module->entry_computation(), HloOpcode::kAllToAll);
  EXPECT_FALSE(SolLatencyEstimator::ComputeCollectiveTime(
                   *nvl_instr, gpu_info, shape_size_fn, sol_flags,
                   /*collective_interpolator=*/nullptr)
                   .ok());
  EXPECT_TRUE(
      SolLatencyEstimator::ComputeCollectiveTime(
          *nvl_instr, gpu_info, shape_size_fn, sol_flags, interpolator.get())
          .ok());

  // Cross-partition collective should use S-curve model (world-level across 2
  // hosts).
  ASSERT_OK_AND_ASSIGN(auto ib_module, ParseAndReturnVerifiedModule(R"(
HloModule m, num_partitions=16
ENTRY main {
  p = bf16[16,16000,1000] parameter(0)
  ROOT a2a = bf16[16,16000,1000] all-to-all(p),
    replica_groups={{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}},
    channel_id=1, dimensions={0}
})"));
  HloInstruction* ib_instr = hlo_query::FindInstruction(
      ib_module->entry_computation(), HloOpcode::kAllToAll);
  EXPECT_TRUE(SolLatencyEstimator::ComputeCollectiveTime(
                  *ib_instr, gpu_info, shape_size_fn, sol_flags,
                  /*collective_interpolator=*/nullptr)
                  .ok());
}

class IsSolLatencyEstimatorEnabledTest : public HloTestBaseLegacy {
 protected:
  IsSolLatencyEstimatorEnabledTest()
      : gpu_device_info_(TestGpuDeviceInfo::RTXA6000DeviceInfo()) {}

  std::unique_ptr<HloModule> CreateTestModule(
      const HloModuleConfig& config,
      const std::string& module_name = "test_module") {
    auto module = std::make_unique<HloModule>(module_name, config);
    HloComputation::Builder builder("entry");
    auto param = builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {}), "param"));
    module->AddEntryComputation(builder.Build(param));
    return module;
  }

  // Helper to add an AllReduce instruction to a module's entry computation.
  void AddAllReduce(HloModule* module) {
    HloComputation* entry = module->entry_computation();
    Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
    auto dummy_operand = entry->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}})));
    Shape s(shape.element_type(), /*dimensions=*/{});
    HloComputation::Builder wrapped_computation("wrapped_computation");
    HloInstruction* a = wrapped_computation.AddInstruction(
        HloInstruction::CreateParameter(0, s, "p0.1"));
    HloInstruction* b = wrapped_computation.AddInstruction(
        HloInstruction::CreateParameter(1, s, "p0.2"));
    wrapped_computation.AddInstruction(
        HloInstruction::CreateBinary(s, HloOpcode::kAdd, a, b));

    HloComputation* subcomp =
        module->AddEmbeddedComputation(wrapped_computation.Build());
    entry->AddInstruction(HloInstruction::CreateAllReduce(
        shape, {dummy_operand}, subcomp,
        std::make_shared<CollectiveDeviceList>(),
        /*constrain_layout=*/false,
        /*channel_id=*/std::nullopt, /*use_global_device_ids=*/false));
  }

  // Helper to add a AllToAll instruction.
  void AddAlltoAll(HloModule* module) {
    HloComputation* entry = module->entry_computation();
    Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
    auto dummy_operand = entry->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}})));
    entry->AddInstruction(HloInstruction::CreateAllToAll(
        shape, {dummy_operand}, std::make_shared<CollectiveDeviceList>(),
        /*constrain_layout=*/false, /*channel_id=*/false,
        /*split_dimension=*/std::nullopt));
  }

  void AddCollectiveBcast(HloModule* module) {
    HloComputation* entry = module->entry_computation();
    Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
    auto dummy_operand = entry->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}})));
    entry->AddInstruction(HloInstruction::CreateCollectiveBroadcast(
        shape, {dummy_operand}, std::make_shared<CollectiveDeviceList>(),
        /*constrain_layout=*/false, /*channel_id=*/std::nullopt));
  }

  // Helper to add a CollectivePermute instruction.
  void AddCollectivePermute(HloModule* module) {
    HloComputation* entry = module->entry_computation();
    Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
    auto dummy_operand = entry->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}})));
    entry->AddInstruction(HloInstruction::CreateCollectivePermute(
        shape, dummy_operand, /*source_target_pairs=*/{}, std::nullopt));
  }

  absl::Status AddHostOffloaded(HloModule* module) {
    HloComputation* entry = module->entry_computation();
    Shape shape = ShapeUtil::MakeShape(F32, {2});
    auto dummy_operand = entry->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR1<float>({2})));
    HloInstruction* call =
        entry->AddInstruction(HloInstruction::CreateCall(shape, dummy_operand));
    ASSIGN_OR_RETURN(GpuBackendConfig new_backend_config,
                     call->backend_config<GpuBackendConfig>());
    new_backend_config.set_device_type(DEVICE_TYPE_HOST);
    return call->set_backend_config(new_backend_config);
  }

  se::DeviceDescription gpu_device_info_;
};

TEST_F(IsSolLatencyEstimatorEnabledTest, EnabledBySolEstimatorFlagOnHopper) {
  HloModuleConfig config;
  config.mutable_debug_options()
      .set_xla_gpu_enable_analytical_sol_latency_estimator(true);
  gpu_device_info_.set_cuda_compute_capability(
      stream_executor::CudaComputeCapability::Hopper());

  auto module = CreateTestModule(config);
  EXPECT_TRUE(
      SolLatencyEstimator::IsSupportedForModule(*module, gpu_device_info_));
}

TEST_F(IsSolLatencyEstimatorEnabledTest, DisabledIfFlagIsOffOnHopper) {
  HloModuleConfig config;

  gpu_device_info_.set_cuda_compute_capability(
      stream_executor::CudaComputeCapability::Hopper());

  config.mutable_debug_options()
      .set_xla_gpu_enable_analytical_sol_latency_estimator(false);

  auto module = CreateTestModule(config);

  EXPECT_FALSE(
      SolLatencyEstimator::IsSupportedForModule(*module, gpu_device_info_));
}

TEST_F(IsSolLatencyEstimatorEnabledTest,
       DisabledForHopperWithUnsupportedCollective) {
  HloModuleConfig config;
  config.mutable_debug_options()
      .set_xla_gpu_enable_analytical_sol_latency_estimator(true);

  gpu_device_info_.set_cuda_compute_capability(
      stream_executor::CudaComputeCapability::Hopper());

  auto module = CreateTestModule(config);
  AddCollectiveBcast(module.get());  // Unsupported collective

  EXPECT_FALSE(
      SolLatencyEstimator::IsSupportedForModule(*module, gpu_device_info_));
}

TEST_F(IsSolLatencyEstimatorEnabledTest,
       DisabledForHopperWithMixedCollectives) {
  HloModuleConfig config;
  config.mutable_debug_options()
      .set_xla_gpu_enable_analytical_sol_latency_estimator(true);

  gpu_device_info_.set_cuda_compute_capability(
      stream_executor::CudaComputeCapability::Hopper());

  auto module = CreateTestModule(config);
  AddAllReduce(module.get());          // Supported collective
  AddCollectivePermute(module.get());  // Supported collective
  AddAlltoAll(module.get());           // Supported collective
  AddCollectiveBcast(module.get());    // Unsupported collective

  EXPECT_FALSE(
      SolLatencyEstimator::IsSupportedForModule(*module, gpu_device_info_));
}

TEST_F(IsSolLatencyEstimatorEnabledTest, DisabledIfNotHopper) {
  HloModuleConfig config;
  config.mutable_debug_options()
      .set_xla_gpu_enable_analytical_sol_latency_estimator(true);

  gpu_device_info_.set_cuda_compute_capability(
      stream_executor::CudaComputeCapability::Ampere());  // Not Hopper

  auto module = CreateTestModule(config);
  AddAllReduce(module.get());  // Supported collective

  EXPECT_FALSE(
      SolLatencyEstimator::IsSupportedForModule(*module, gpu_device_info_));
}

TEST_F(IsSolLatencyEstimatorEnabledTest, DisabledForHopperWithHostOffloaded) {
  HloModuleConfig config;
  config.mutable_debug_options()
      .set_xla_gpu_enable_analytical_sol_latency_estimator(true);

  gpu_device_info_.set_cuda_compute_capability(
      stream_executor::CudaComputeCapability::Hopper());

  auto module = CreateTestModule(config);
  ASSERT_OK(AddHostOffloaded(module.get()));

  EXPECT_FALSE(
      SolLatencyEstimator::IsSupportedForModule(*module, gpu_device_info_));
}

// ---- Triton collective scheduler integration tests -----------------------
//
// These tests verify that SolLatencyEstimator correctly uses the NVLink-based
// cost formula when an AllReduceStart is annotated with a Triton kernel
// strategy, and that the sync case is correctly handled by NodeCost.

class TritonAllReduceSchedulerTest : public HloHardwareIndependentTestBase {
 protected:
  // 32768 F32 elements = 128 KB → one-shot strategy (< 256 KB threshold).
  // All 8 replicas in a single group → SINGLE_PARTITION communication type.
  static constexpr absl::string_view kOneShotAllReduceHlo = R"(
HloModule m, num_partitions=8

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT r = f32[] add(p0, p1)
}

ENTRY e {
  p = f32[32768] parameter(0)
  ar-start = f32[32768] all-reduce-start(p),
      replica_groups={{0,1,2,3,4,5,6,7}},
      to_apply=add
  ROOT ar-done = f32[32768] all-reduce-done(ar-start)
})";

  TritonAllReduceSchedulerTest() {
    // Use an RTXA6000 with Hopper capability.
    gpu_device_info_ =
        TestGpuDeviceInfo::RTXA6000DeviceInfo(se::CudaComputeCapability(9, 0));
    // Equip it with 18 active NVLink links (H100-like topology).
    se::DeviceInterconnectInfo interconnect;
    interconnect.active_links = 18;
    gpu_device_info_.set_device_interconnect_info(interconnect);

    // NVLink-aware sol_flags: 20 GB/s per lane, 1.5 µs barrier.
    sol_flags_ = {
        /*nccl_op_launch_time=*/absl::Microseconds(100),
        /*nic_speed_gbps=*/100,
        /*chunk_prep_time=*/absl::Microseconds(100),
        /*rtt=*/absl::Microseconds(100),
        /*gpus_per_node=*/8,
        /*chunk_size_bytes=*/4 * 1024 * 1024,
        /*partition_size=*/0,
        /*nvlink_bw_per_lane_gbps=*/20.0,
        /*nvlink_barrier_latency=*/absl::Microseconds(1.5),
    };
  }

  // Annotates `instr` with the given Triton kernel strategy.
  static void SetKernelStrategy(
      HloInstruction* instr,
      CollectiveBackendConfig::CollectiveKernelStrategy strategy) {
    GpuBackendConfig cfg = instr->backend_config<GpuBackendConfig>().value();
    cfg.mutable_collective_backend_config()->set_kernel_strategy(strategy);
    CHECK_OK(instr->set_backend_config(cfg));
  }

  // Marks `instr` as synchronous (runs on compute stream, no async overlap).
  static void SetSync(HloInstruction* instr) {
    GpuBackendConfig cfg = instr->backend_config<GpuBackendConfig>().value();
    cfg.mutable_collective_backend_config()->set_is_sync(true);
    CHECK_OK(instr->set_backend_config(cfg));
  }

  // Calls ComputeCollectiveTime with the fixture's device and flags.
  absl::StatusOr<absl::Duration> ComputeCollectiveTime(
      const HloInstruction& instr) {
    return SolLatencyEstimator::ComputeCollectiveTime(
        instr, gpu_device_info_, HloCostAnalysis::DefaultShapeSize, sol_flags_,
        /*collective_interpolator=*/nullptr);
  }

  // Creates a SolLatencyEstimator and calls NodeCost.
  absl::Duration NodeCost(const HloInstruction& instr,
                          const HloComputation* computation) {
    auto estimator = *SolLatencyEstimator::Create(
        scheduler_config_, std::make_unique<DummyLatencyEstimator>(),
        gpu_device_info_, HloCostAnalysis::DefaultShapeSize, computation);
    return absl::Microseconds(
        static_cast<int64_t>(estimator->NodeCost(&instr)));
  }

  se::DeviceDescription gpu_device_info_;
  SolGPUCostModel::Config sol_flags_;
  SchedulerConfig scheduler_config_;
};

// Test 1: async Triton ONE_SHOT AllReduce.
// With the KERNEL_STRATEGY_TRITON_ONE_SHOT annotation and 18 NVLink
// links at 20 GB/s, ComputeCollectiveTime must return the NVLink formula
// result:
//   transfer = 7 × 128 KB / (18 × 20 GB/s) ≈ 2.5 µs
//   total ≈ launch(1µs) + 2.5µs + barrier(1.5µs) ≈ 5 µs
// This must be different from the NCCL ring estimate (which uses nic_speed_gbps
// and has a much larger RTT term).
TEST_F(TritonAllReduceSchedulerTest, AsyncTritonOneShotUsesNvlinkFormula) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kOneShotAllReduceHlo));

  HloInstruction* ar_start = hlo_query::FindInstruction(
      module->entry_computation(), HloOpcode::kAllReduceStart);
  ASSERT_NE(ar_start, nullptr);

  // Annotate with Triton one-shot strategy (async: is_sync is false by default)
  SetKernelStrategy(ar_start,
                    CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT);

  // A real interpolator is passed so the NCCL fallback path doesn't crash.
  // If the Triton dispatch fires correctly, the interpolator is not called.
  auto interpolator = *CollectiveInterpolator::Create(
      sol_flags_.gpus_per_node, gpu_device_info_, /*analysis=*/nullptr);

  ASSERT_OK_AND_ASSIGN(
      absl::Duration triton_time,
      SolLatencyEstimator::ComputeCollectiveTime(
          *ar_start, gpu_device_info_, HloCostAnalysis::DefaultShapeSize,
          sol_flags_, interpolator.get()));
  triton_time = absl::Trunc(triton_time, absl::Microseconds(1));

  // The NVLink formula gives ~5 µs.
  EXPECT_EQ(triton_time, absl::Microseconds(5));

  // Verify this is different from the NCCL ring estimate: remove annotation.
  SetKernelStrategy(ar_start, CollectiveBackendConfig::KERNEL_STRATEGY_DEFAULT);
  ASSERT_OK_AND_ASSIGN(
      absl::Duration nccl_time,
      SolLatencyEstimator::ComputeCollectiveTime(
          *ar_start, gpu_device_info_, HloCostAnalysis::DefaultShapeSize,
          sol_flags_, interpolator.get()));
  // NCCL ring with nic=100 GB/s and rtt=100 µs gives a much larger value.
  EXPECT_NE(absl::Trunc(nccl_time, absl::Microseconds(1)), triton_time);
}

// Test 2: sync Triton ONE_SHOT AllReduce - NodeCost must NOT return kLowCost.
// When is_sync=true and kernel_strategy=TRITON_ONE_SHOT, NodeCost() must
// delegate to ComputeCollectiveTime so the latency is on the critical path.
// We verify this by calling ComputeCollectiveTime directly (which uses our
// NVLink flags) and checking it returns > kLowCost (1.0 µs).
TEST_F(TritonAllReduceSchedulerTest,
       SyncTritonOneShotComputeCollectiveTimeExceedsLowCost) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kOneShotAllReduceHlo));

  HloInstruction* ar_start = hlo_query::FindInstruction(
      module->entry_computation(), HloOpcode::kAllReduceStart);
  ASSERT_NE(ar_start, nullptr);

  // Mark as synchronous Triton one-shot.
  SetSync(ar_start);
  SetKernelStrategy(ar_start,
                    CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT);

  ASSERT_OK_AND_ASSIGN(absl::Duration cost, ComputeCollectiveTime(*ar_start));

  // The NVLink formula must return significantly more than kLowCost = 1.0 µs.
  // We expect ~5 µs.  Any value > 2 µs confirms we are not returning kLowCost.
  EXPECT_GT(cost, absl::Microseconds(2));
  EXPECT_EQ(absl::Trunc(cost, absl::Microseconds(1)), absl::Microseconds(5));
}

// Test 3: sync Triton ONE_SHOT AllReduce exercises the actual NodeCost
// branching logic end-to-end via SolLatencyEstimator::Create.
// SolLatencyEstimator::Create derives sol_flags from GetConfig(module,
// device_info), which calls GetIciBandwidthPerLaneGbps on the device.
// For RTXA6000 with SM9.0 (Hopper) and 18 active NVLink links:
//   nvlink_bw_per_lane_gbps = kSm90NvlinkBandwidth = 20.0 GB/s
//   nvlink_barrier_latency  = 1.5 µs  (kUnknownKey table entry)
// → NodeCost returns ~5 µs via ComputeCollectiveTime, confirming that the
//   IsGPUSyncCollective + TRITON_ONE_SHOT branch in NodeCost fires correctly
//   with production-derived (not test-injected) sol_flags.
TEST_F(TritonAllReduceSchedulerTest,
       SyncTritonOneShotNodeCostExceedsLowCostEndToEnd) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kOneShotAllReduceHlo));

  HloInstruction* ar_start = hlo_query::FindInstruction(
      module->entry_computation(), HloOpcode::kAllReduceStart);
  ASSERT_NE(ar_start, nullptr);

  SetSync(ar_start);
  SetKernelStrategy(ar_start,
                    CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT);

  // NodeCost() internally calls SolLatencyEstimator::Create which reads
  // sol_flags from GetConfig(module, device_info).  For SM9.0 + 18 links
  // the derived config gives nvlink_bw=20.0 GB/s and barrier=1.5 µs, so
  // the Triton one-shot formula produces ~5 µs — well above kLowCost=1 µs.
  absl::Duration node_cost = NodeCost(*ar_start, module->entry_computation());

  // The Triton one-shot formula: launch(1µs) + 7×128KB/(18×20GB/s) + 1.5µs
  // ≈ 5 µs.  Verify the NodeCost branch fires (> kLowCost) and matches.
  EXPECT_GT(node_cost, absl::Microseconds(1))
      << "NodeCost for sync Triton AllReduce should exceed kLowCost=1µs; "
         "verify that the TRITON_ONE_SHOT branch in NodeCost fired";
  EXPECT_EQ(node_cost, absl::Microseconds(5));
}

// Test 4: without the annotation (NCCL path), async AllReduceStart NodeCost
// returns SolLatencyEstimator::kLowCost = 1.0 (latency is hidden, not on
// critical path).
TEST_F(TritonAllReduceSchedulerTest, AsyncNcclAllReduceNodeCostIsLow) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kOneShotAllReduceHlo));

  HloInstruction* ar_start = hlo_query::FindInstruction(
      module->entry_computation(), HloOpcode::kAllReduceStart);
  ASSERT_NE(ar_start, nullptr);
  // No annotation: is_sync=false (default), kernel_strategy=DEFAULT (default).

  absl::Duration node_cost = NodeCost(*ar_start, module->entry_computation());

  // kLowCost = 1.0 µs: async collective start is treated as negligible node
  // cost because its latency is hidden by GetLatencyBetween.
  EXPECT_EQ(node_cost, absl::Microseconds(1));
}

}  // namespace
}  // namespace xla::gpu
