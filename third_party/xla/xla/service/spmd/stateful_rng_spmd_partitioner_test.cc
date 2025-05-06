/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/expanders/rng_expander.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/sharding_propagation.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace spmd {
namespace {

int64_t CountInstructions(const HloComputation &computation, HloOpcode opcode) {
  int64_t count = 0;
  for (const auto &instruction : computation.instructions()) {
    if (instruction->opcode() == opcode) {
      count++;
    }
  }
  return count;
}

class StatefulRngSpmdPartitionerTest : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> PartitionComputation(
      absl::string_view hlo_module, int64_t num_partitions,
      DebugOptions debug_options,
      std::function<void(HloPassPipeline &pipeline)> add_passes = nullptr,
      bool skip_checking_windowed_einsum_users = false,
      bool disable_ag_rewrite_for_multiple_consumers = false) {
    HloModuleConfig config = GetModuleConfigForTest(1, num_partitions);
    config.set_use_spmd_partitioning(true);
    config.set_debug_options(debug_options);
    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnVerifiedModule(hlo_module, config));
    HloPassPipeline pass("partitioning");
    pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/false);
    if (add_passes) {
      add_passes(pass);
    }
    pass.AddPass<ShardingPropagation>(/*is_spmd=*/true);
    pass.AddPass<StatefulRngSpmdPartitioner>(
        num_partitions,
        /*num_replicas=*/1,
        debug_options.xla_gpu_threshold_for_windowed_einsum_mib(),
        debug_options.xla_gpu_multi_streamed_windowed_einsum(),
        skip_checking_windowed_einsum_users,
        disable_ag_rewrite_for_multiple_consumers,
        debug_options.xla_gpu_operand_bytes_threshold_for_windowed_einsum());
    pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/false);
    TF_RETURN_IF_ERROR(pass.Run(module.get()).status());
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  void VerifyNoAllReduce(HloModule *module) {
    for (HloComputation *computation : module->computations()) {
      for (HloInstruction *hlo : computation->instructions()) {
        EXPECT_NE(hlo->opcode(), HloOpcode::kAllReduce);
      }
    }
  }

  DebugOptions GetDefaultDebugOptions() {
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_threshold_for_windowed_einsum_mib(1000000);
    debug_options.set_xla_gpu_multi_streamed_windowed_einsum(false);
    return debug_options;
  }
};

TEST_F(StatefulRngSpmdPartitionerTest, RngReplicatedConsumer) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
   %p0 = f32[50,100] parameter(0), sharding={replicated}
   %mu = f32[] constant(0)
   %sigma = f32[] constant(1)
   %rng = f32[50,100] rng(f32[] %mu, f32[] %sigma), distribution=rng_uniform
   ROOT %add = f32[50,100] add(%rng, %p0), sharding={replicated}
}
)";

  auto add_passes = [](HloPassPipeline &pipeline) {
    pipeline.AddPass<RngExpander>();
  };

  DebugOptions debug_options = GetDebugOptionsForTest();

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, PartitionComputation(hlo_string, /*num_partitions=*/2,
                                        GetDefaultDebugOptions(), add_passes));
  XLA_VLOG_LINES(1, module->ToString());
  VerifyNoAllReduce(module.get());
}

TEST_F(StatefulRngSpmdPartitionerTest, RngPartitionedConsumer) {
  absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
   %p0 = f32[50,100] parameter(0), sharding={replicated}
   %mu = f32[] constant(0)
   %sigma = f32[] constant(1)
   %rng = f32[50,100] rng(f32[] %mu, f32[] %sigma), distribution=rng_uniform
   ROOT %add = f32[50,100] add(%rng, %p0), sharding={devices=[2,1]0,1}
}
)";

  auto add_passes = [](HloPassPipeline &pipeline) {
    pipeline.AddPass<RngExpander>();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, PartitionComputation(hlo_string, /*num_partitions=*/2,
                                        GetDefaultDebugOptions(), add_passes));
  XLA_VLOG_LINES(1, module->ToString());
  VerifyNoAllReduce(module.get());
}

TEST_F(StatefulRngSpmdPartitionerTest,
       EinsumDisableRewriteForAgWithMultipleConsumers) {
  absl::string_view hlo_string = R"(
HloModule test, entry_computation_layout={(bf16[2,2048,24576]{2,1,0}, bf16[24576,98304]{1,0}, bf16[24576,98304]{1,0})->bf16[2,2048,98304]{2,1,0}}, num_partitions=4

ENTRY main {
  Arg_0.1 = bf16[2,2048,24576]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
  Arg_1.2 = bf16[24576,98304]{1,0} parameter(1), sharding={devices=[1,4]<=[4]}
  dot.5 = bf16[2,2048,98304]{2,1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}, sharding={devices=[1,1,4]<=[4]}
  Arg_2.3 = bf16[24576,98304]{1,0} parameter(2), sharding={devices=[1,4]<=[4]}
  dot.6 = bf16[2,2048,98304]{2,1,0} dot(Arg_0.1, Arg_2.3), lhs_contracting_dims={2}, rhs_contracting_dims={0}, sharding={devices=[1,1,4]<=[4]}
  ROOT add.8 = bf16[2,2048,98304]{2,1,0} add(dot.5, dot.6), sharding={devices=[1,1,4]<=[4]}
}

)";
  // With disable_ag_rewrite_for_multiple_consumers set to true, we expect only
  // 1 while loop to exist which is the rewritten windowed einsum loop for the
  // first ag->dot pattern. The second dot which shares the same operand with
  // the loop will remain as is.
  DebugOptions debug_options = GetDefaultDebugOptions();
  debug_options.set_xla_gpu_threshold_for_windowed_einsum_mib(0);
  debug_options.set_xla_gpu_multi_streamed_windowed_einsum(true);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_partitions=*/4, debug_options,
                           /*add_passes=*/nullptr,
                           /*skip_checking_windowed_einsum_users=*/true,
                           /*disable_ag_rewrite_for_multiple_consumers=*/true));
  XLA_VLOG_LINES(1, module->ToString());
  EXPECT_EQ(CountInstructions(*module->entry_computation(), HloOpcode::kWhile),
            1);
  EXPECT_EQ(CountInstructions(*module->entry_computation(), HloOpcode::kDot),
            1);
  EXPECT_EQ(
      CountInstructions(*module->entry_computation(), HloOpcode::kAllGather),
      1);
}

TEST_F(StatefulRngSpmdPartitionerTest, VerifyThresholdSetCorrectly) {
  auto debug_options = HloHardwareIndependentTestBase::GetDebugOptionsForTest();
  int64_t threshold = 400;
  debug_options.set_xla_gpu_threshold_for_windowed_einsum_mib(threshold);
  debug_options.set_xla_gpu_multi_streamed_windowed_einsum(true);

  StatefulRngSpmdPartitioner rng_spmd_partitioner(
      /*num_partitions=*/2, /*num_replicas*/ 1,
      debug_options.xla_gpu_threshold_for_windowed_einsum_mib(),
      debug_options.xla_gpu_multi_streamed_windowed_einsum());
  EXPECT_EQ(rng_spmd_partitioner.options().threshold_for_windowed_einsum_mib,
            threshold);
  EXPECT_EQ(rng_spmd_partitioner.options().unroll_windowed_einsum, true);
}

TEST_F(StatefulRngSpmdPartitionerTest,
       TotalFlopsThresholdOverrideOperandThreshold) {
  absl::string_view hlo_string = R"(
HloModule test, entry_computation_layout={(bf16[2,128,256]{2,1,0}, bf16[256,512]{1,0})->bf16[2,128,512]{2,1,0}}, num_partitions=4

ENTRY main {
  Arg_0.1 = bf16[2,128,256]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
  Arg_1.2 = bf16[256,512]{1,0} parameter(1), sharding={devices=[1,4]<=[4]}
  ROOT dot.5 = bf16[2,128,512]{2,1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}, sharding={devices=[1,1,4]<=[4]}
}

)";
  DebugOptions debug_options = GetDefaultDebugOptions();
  debug_options.set_xla_gpu_threshold_for_windowed_einsum_mib(0);
  debug_options.set_xla_gpu_multi_streamed_windowed_einsum(true);
  int64_t oper_bytes_threshold = 1 << 20;
  debug_options.set_xla_gpu_operand_bytes_threshold_for_windowed_einsum(
      oper_bytes_threshold);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_partitions=*/4, debug_options,
                           /*add_passes=*/nullptr,
                           /*skip_checking_windowed_einsum_users=*/true,
                           /*disable_ag_rewrite_for_multiple_consumers=*/true));
  XLA_VLOG_LINES(1, module->ToString());
  // The operand threshold is set to 0 but flops threshold is set to be
  // larger than the total flops of the gemm. So we don't expect any
  // windowed einsum loop but rather an all-gather.
  EXPECT_EQ(CountInstructions(*module->entry_computation(), HloOpcode::kWhile),
            0);
  EXPECT_EQ(
      CountInstructions(*module->entry_computation(), HloOpcode::kAllGather),
      1);
}

TEST_F(StatefulRngSpmdPartitionerTest,
       TotalFlopsThresholdShouldEnableWindowedEinsum) {
  absl::string_view hlo_string = R"(
HloModule test, entry_computation_layout={(bf16[2,128,256]{2,1,0}, bf16[256,512]{1,0})->bf16[2,128,512]{2,1,0}}, num_partitions=4

ENTRY main {
  Arg_0.1 = bf16[2,128,256]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
  Arg_1.2 = bf16[256,512]{1,0} parameter(1), sharding={devices=[1,4]<=[4]}
  ROOT dot.5 = bf16[2,128,512]{2,1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}, sharding={devices=[1,1,4]<=[4]}
}

)";
  DebugOptions debug_options = GetDefaultDebugOptions();
  debug_options.set_xla_gpu_multi_streamed_windowed_einsum(true);
  int64_t oper_bytes_threshold = 1 << 8;
  debug_options.set_xla_gpu_operand_bytes_threshold_for_windowed_einsum(
      oper_bytes_threshold);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_partitions=*/4, debug_options,
                           /*add_passes=*/nullptr,
                           /*skip_checking_windowed_einsum_users=*/true,
                           /*disable_ag_rewrite_for_multiple_consumers=*/true));
  XLA_VLOG_LINES(1, module->ToString());
  // The operand threshold is not set which defaults to 1000000 MB.
  // But the flops threshold is set, the windowed einsum should still kick in.
  EXPECT_EQ(CountInstructions(*module->entry_computation(), HloOpcode::kWhile),
            1);
  EXPECT_EQ(
      CountInstructions(*module->entry_computation(), HloOpcode::kAllGather),
      0);
}

}  // namespace
}  // namespace spmd
}  // namespace xla
