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

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/hlo_parser.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/rng_expander.h"
#include "xla/service/sharding_propagation.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace spmd {
namespace {

class StatefulRngSpmdPartitionerTest : public HloTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> PartitionComputation(
      absl::string_view hlo_module, int64_t num_partitions,
      std::function<void(HloPassPipeline &pipeline)> add_passes = nullptr) {
    TF_ASSIGN_OR_RETURN(
        auto module, ParseAndReturnVerifiedModule(
                         hlo_module, GetModuleConfigForTest(
                                         /*replica_count=*/1,
                                         /*num_partitions=*/num_partitions)));
    HloPassPipeline pass("partitioning");
    pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/false);
    if (add_passes) {
      add_passes(pass);
    }
    pass.AddPass<ShardingPropagation>(/*is_spmd=*/true);
    pass.AddPass<StatefulRngSpmdPartitioner>(num_partitions,
                                             /*num_replicas=*/1);
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

  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      PartitionComputation(hlo_string, /*num_partitions=*/2, add_passes));
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
      auto module,
      PartitionComputation(hlo_string, /*num_partitions=*/2, add_passes));
  XLA_VLOG_LINES(1, module->ToString());
  VerifyNoAllReduce(module.get());
}

TEST_F(StatefulRngSpmdPartitionerTest, VerifyThresholdSetCorrectly) {
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
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
}  // namespace
}  // namespace spmd
}  // namespace xla
