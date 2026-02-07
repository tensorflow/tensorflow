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

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "mlir/IR/MLIRContext.h"
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
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
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
        }),
        collective_interpolator_(*CollectiveInterpolator::Create(
            sol_flags_.gpus_per_node, gpu_device_info_,
            /*analysis=*/nullptr)) {}

  absl::StatusOr<absl::Duration> ComputeCollectiveTime(
      const HloInstruction& instr) {
    return SolLatencyEstimator::ComputeCollectiveTime(
        instr, gpu_device_info_, shape_size_fn_, sol_flags_, &mlir_context_,
        collective_interpolator_.get());
  }

  absl::Duration ComputeNodeCost(const HloInstruction& instr,
                                 const HloComputation* computation) {
    std::unique_ptr<SolLatencyEstimator> estimator =
        *SolLatencyEstimator::Create(
            scheduler_config_, std::make_unique<DummyLatencyEstimator>(),
            gpu_device_info_, shape_size_fn_, computation, &mlir_context_);
    LatencyEstimator::TimeCost cost_val = estimator->NodeCost(&instr);
    return absl::Microseconds(static_cast<int64_t>(cost_val));
  }

  absl::Duration GetLatencyBetween(const HloGraphNode& from,
                                   const HloGraphNode& target,
                                   const HloComputation* computation) {
    std::unique_ptr<SolLatencyEstimator> estimator =
        *SolLatencyEstimator::Create(
            scheduler_config_, std::make_unique<DummyLatencyEstimator>(),
            gpu_device_info_, shape_size_fn_, computation, &mlir_context_);
    LatencyEstimator::TimeCost cost_val =
        estimator->GetLatencyBetween(from, target);
    return absl::Microseconds(static_cast<int64_t>(cost_val));
  }

  HloCostAnalysis::ShapeSizeFunction shape_size_fn_;
  const se::DeviceDescription gpu_device_info_;
  const SolGPUCostModel::Config sol_flags_;
  SchedulerConfig scheduler_config_;
  std::unique_ptr<CollectiveInterpolator> collective_interpolator_;
  mlir::MLIRContext mlir_context_;
};

TEST_P(SolLatencyEstimatorTest, TestLatencyEstimation) {
  EstimatorTestCase test_case = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(test_case.module_string));

  HloInstruction* instr = hlo_query::FindInstruction(
      module->entry_computation(), test_case.opcode_to_find);
  ASSERT_NE(instr, nullptr);
  absl::Duration actual_time_us;
  if (test_case.cost_type == CostType::kCollectiveTime) {
    TF_ASSERT_OK_AND_ASSIGN(absl::Duration time_us,
                            ComputeCollectiveTime(*instr));
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
  ROOT _ =  (bf16[1024,1024], s8[2097152]) custom-call(p0,p1),
    custom_call_target="__cublas$gemm",
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
  mlir::MLIRContext mlir_ctx;
  auto interpolator =
      *CollectiveInterpolator::Create(sol_flags.gpus_per_node, gpu_info,
                                      /*analysis=*/nullptr);

  // NVLink domain collective should use CollectiveInterpolator.
  TF_ASSERT_OK_AND_ASSIGN(auto nvl_module, ParseAndReturnVerifiedModule(R"(
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
                   *nvl_instr, gpu_info, shape_size_fn, sol_flags, &mlir_ctx,
                   /*collective_interpolator=*/nullptr)
                   .ok());
  EXPECT_TRUE(SolLatencyEstimator::ComputeCollectiveTime(
                  *nvl_instr, gpu_info, shape_size_fn, sol_flags, &mlir_ctx,
                  interpolator.get())
                  .ok());

  // Cross-partition collective should use S-curve model (world-level across 2
  // hosts).
  TF_ASSERT_OK_AND_ASSIGN(auto ib_module, ParseAndReturnVerifiedModule(R"(
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
                  *ib_instr, gpu_info, shape_size_fn, sol_flags, &mlir_ctx,
                  /*collective_interpolator=*/nullptr)
                  .ok());
}

class IsSolLatencyEstimatorEnabledTest : public HloTestBase {
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
        /*device_list=*/CollectiveDeviceList(), /*constrain_layout=*/false,
        /*channel_id=*/std::nullopt, /*use_global_device_ids=*/false));
  }

  // Helper to add a AllToAll instruction.
  void AddAlltoAll(HloModule* module) {
    HloComputation* entry = module->entry_computation();
    Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
    auto dummy_operand = entry->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}})));
    entry->AddInstruction(HloInstruction::CreateAllToAll(
        shape, {dummy_operand},
        /*device_list=*/CollectiveDeviceList(),
        /*constrain_layout=*/false, /*channel_id=*/false,
        /*split_dimension=*/std::nullopt));
  }

  void AddCollectiveBcast(HloModule* module) {
    HloComputation* entry = module->entry_computation();
    Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
    auto dummy_operand = entry->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR2<float>({{1, 2}, {3, 4}})));
    entry->AddInstruction(HloInstruction::CreateCollectiveBroadcast(
        shape, {dummy_operand},
        /*device_list=*/CollectiveDeviceList(),
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
    TF_ASSIGN_OR_RETURN(GpuBackendConfig new_backend_config,
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
  TF_ASSERT_OK(AddHostOffloaded(module.get()));

  EXPECT_FALSE(
      SolLatencyEstimator::IsSupportedForModule(*module, gpu_device_info_));
}

}  // namespace
}  // namespace xla::gpu
