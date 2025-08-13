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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace spmd {
namespace {

class DotHandlerTest : public HloHardwareIndependentTestBase {
 public:
  absl::StatusOr<std::unique_ptr<HloModule>> PartitionComputation(
      absl::string_view hlo_module, int64_t num_partitions,
      int64_t max_windowed_einsum_iteration = 32,         // Default value
      int64_t threshold_for_windowed_einsum_mib = 256) {  // Default value
    HloModuleConfig config = GetModuleConfigForTest(1, num_partitions);
    config.set_use_spmd_partitioning(true);

    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_threshold_for_windowed_einsum_mib(
        threshold_for_windowed_einsum_mib);
    debug_options.set_xla_gpu_multi_streamed_windowed_einsum(true);
    config.set_debug_options(debug_options);

    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnVerifiedModule(hlo_module, config));

    HloPassPipeline pass("partitioning");
    pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/false);
    pass.AddPass<ShardingPropagation>(/*is_spmd=*/true);
    pass.AddPass<StatefulRngSpmdPartitioner>(
        num_partitions,
        /*num_replicas=*/1, threshold_for_windowed_einsum_mib,
        /*windowed_einsum_use_multiple_streams=*/
        debug_options.xla_gpu_multi_streamed_windowed_einsum(),
        /*skip_checking_windowed_einsum_users=*/true,  // Skip user checking
        /*disable_ag_rewrite_for_multiple_consumers=*/false,
        /*enable_partial_windowed_einsums=*/false,
        /*total_bytes_windowed_einsum_threshold=*/std::nullopt,
        /*max_windowed_einsum_iteration=*/max_windowed_einsum_iteration);
    pass.AddPass<HloVerifier>(/*layout_sensitive=*/false,
                              /*allow_mixed_precision=*/false);

    TF_RETURN_IF_ERROR(pass.Run(module.get()).status());
    return absl::StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }

  int64_t CountInstructions(const HloComputation& computation,
                            HloOpcode opcode) {
    int64_t count = 0;
    for (const auto& instruction : computation.instructions()) {
      if (instruction->opcode() == opcode) {
        count++;
      }
    }
    return count;
  }
};

TEST_F(DotHandlerTest, VerifyDefaultMaxWindowedEinsumIterationInPartitioner) {
  // Verify that StatefulRngSpmdPartitioner correctly sets the default
  // max_windowed_einsum_iteration when not explicitly provided

  // Create partitioner without specifying max_windowed_einsum_iteration
  StatefulRngSpmdPartitioner partitioner_default(
      /*num_partitions=*/4,
      /*num_replicas=*/1,
      /*threshold_for_windowed_einsum_mib=*/0,
      /*windowed_einsum_use_multiple_streams=*/false);

  // The default should be 32
  EXPECT_EQ(partitioner_default.options().max_windowed_einsum_iteration, 32)
      << "Default max_windowed_einsum_iteration should be 32";

  // Create partitioner with explicit max_windowed_einsum_iteration
  const int64_t custom_max = 42;
  StatefulRngSpmdPartitioner partitioner_custom(
      /*num_partitions=*/4,
      /*num_replicas=*/1,
      /*threshold_for_windowed_einsum_mib=*/0,
      /*windowed_einsum_use_multiple_streams=*/false,
      /*skip_checking_windowed_einsum_users=*/false,
      /*disable_ag_rewrite_for_multiple_consumers=*/false,
      /*enable_partial_windowed_einsums=*/false,
      /*total_bytes_windowed_einsum_threshold=*/std::nullopt,
      /*max_windowed_einsum_iteration=*/custom_max);

  EXPECT_EQ(partitioner_custom.options().max_windowed_einsum_iteration,
            custom_max)
      << "Custom max_windowed_einsum_iteration should be respected";
}

TEST_F(DotHandlerTest, MaxWindowedEinsumIterationWithContractingDims) {
  // Test with contracting dimension sharding pattern
  // This pattern should trigger reduce-scatter windowed einsum
  absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  Arg_0 = bf16[2048,24576]{1,0} parameter(0), sharding={devices=[1,4]<=[4]}
  Arg_1 = bf16[24576,98304]{1,0} parameter(1), sharding={devices=[4,1]<=[4]}
  ROOT dot = bf16[2048,98304]{1,0} dot(Arg_0, Arg_1), 
    lhs_contracting_dims={1}, rhs_contracting_dims={0}, 
    sharding={devices=[1,4]<=[4]}
}
)";

  // With contracting dims sharded and matching, windowed einsum for
  // reduce-scatter pattern should respect max_windowed_einsum_iteration
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto module,
        PartitionComputation(hlo_string, /*num_partitions=*/4,
                             /*max_windowed_einsum_iteration=*/2,
                             /*threshold_for_windowed_einsum_mib=*/0));

    // Should not create windowed einsum loop
    EXPECT_EQ(
        CountInstructions(*module->entry_computation(), HloOpcode::kWhile), 0)
        << "Expected no While loops for contracting dims when max_iterations "
           "too low";
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto module,
        PartitionComputation(hlo_string, /*num_partitions=*/4,
                             /*max_windowed_einsum_iteration=*/4,
                             /*threshold_for_windowed_einsum_mib=*/0));

    // Should create windowed einsum loop
    EXPECT_EQ(
        CountInstructions(*module->entry_computation(), HloOpcode::kWhile), 1)
        << "Expected While loop for contracting dims when max_iterations "
           "allows";
  }
}

TEST_F(DotHandlerTest, MaxWindowedEinsumIterationBatchDims) {
  // Test with batch dimension sharding
  absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  Arg_0 = bf16[8,2048,256]{2,1,0} parameter(0), sharding={devices=[4,1,1]<=[4]}
  Arg_1 = bf16[8,256,512]{2,1,0} parameter(1), sharding={devices=[4,1,1]<=[4]}
  ROOT dot = bf16[8,2048,512]{2,1,0} dot(Arg_0, Arg_1), 
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}, 
    sharding={devices=[4,1,1]<=[4]}
}
)";

  // Batch dims with windowed einsum should also respect max_iterations
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto module,
        PartitionComputation(hlo_string, /*num_partitions=*/4,
                             /*max_windowed_einsum_iteration=*/3,
                             /*threshold_for_windowed_einsum_mib=*/0));

    // With batch dims matching and partitioned, and max_iterations <
    // num_partitions, windowed einsum should be disabled
    EXPECT_EQ(
        CountInstructions(*module->entry_computation(), HloOpcode::kWhile), 0)
        << "Expected no While loops for batch dims when max_iterations too low";
  }
}

TEST_F(DotHandlerTest, DefaultMaxWindowedEinsumIterationWithReduceScatter) {
  // Test that the default max_windowed_einsum_iteration (32) works correctly
  // for reduce-scatter pattern

  // Pattern with 16 partitions (should work with default)
  {
    absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  Arg_0 = bf16[128,256]{1,0} parameter(0), sharding={devices=[1,16]<=[16]}
  Arg_1 = bf16[256,512]{1,0} parameter(1), sharding={devices=[16,1]<=[16]}
  ROOT dot = bf16[128,512]{1,0} dot(Arg_0, Arg_1), 
    lhs_contracting_dims={1}, rhs_contracting_dims={0}, 
    sharding={devices=[1,16]<=[16]}
}
)";

    TF_ASSERT_OK_AND_ASSIGN(
        auto module,
        PartitionComputation(hlo_string, /*num_partitions=*/16,
                             /*max_windowed_einsum_iteration=*/32,
                             /*threshold_for_windowed_einsum_mib=*/0));

    EXPECT_EQ(
        CountInstructions(*module->entry_computation(), HloOpcode::kWhile), 1)
        << "Default should enable RS windowed einsum for 16 partitions";
  }

  // Pattern with exactly 32 partitions (at the limit)
  {
    absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  Arg_0 = bf16[128,256]{1,0} parameter(0), sharding={devices=[1,32]<=[32]}
  Arg_1 = bf16[256,512]{1,0} parameter(1), sharding={devices=[32,1]<=[32]}
  ROOT dot = bf16[128,512]{1,0} dot(Arg_0, Arg_1), 
    lhs_contracting_dims={1}, rhs_contracting_dims={0}, 
    sharding={devices=[1,32]<=[32]}
}
)";

    TF_ASSERT_OK_AND_ASSIGN(
        auto module,
        PartitionComputation(hlo_string, /*num_partitions=*/32,
                             /*max_windowed_einsum_iteration=*/32,
                             /*threshold_for_windowed_einsum_mib=*/0));

    EXPECT_EQ(
        CountInstructions(*module->entry_computation(), HloOpcode::kWhile), 1)
        << "Default should enable RS windowed einsum for exactly 32 partitions "
           "(at limit)";
  }

  // Pattern with 64 partitions (should exceed default limit)
  {
    absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  Arg_0 = bf16[128,256]{1,0} parameter(0), sharding={devices=[1,64]<=[64]}
  Arg_1 = bf16[256,512]{1,0} parameter(1), sharding={devices=[64,1]<=[64]}
  ROOT dot = bf16[128,512]{1,0} dot(Arg_0, Arg_1), 
    lhs_contracting_dims={1}, rhs_contracting_dims={0}, 
    sharding={devices=[1,64]<=[64]}
}
)";

    TF_ASSERT_OK_AND_ASSIGN(
        auto module,
        PartitionComputation(hlo_string, /*num_partitions=*/64,
                             /*max_windowed_einsum_iteration=*/32,
                             /*threshold_for_windowed_einsum_mib=*/0));

    EXPECT_EQ(
        CountInstructions(*module->entry_computation(), HloOpcode::kWhile), 0)
        << "Default should disable windowed einsum for 64 partitions (exceeds "
           "default limit)";
  }
}

TEST_F(DotHandlerTest, MaxWindowedEinsumIterationEdgeCases) {
  // Test edge cases for max_windowed_einsum_iteration
  absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  Arg_0 = bf16[128,256]{1,0} parameter(0), sharding={devices=[1,8]<=[8]}
  Arg_1 = bf16[256,512]{1,0} parameter(1), sharding={devices=[8,1]<=[8]}
  ROOT dot = bf16[128,512]{1,0} dot(Arg_0, Arg_1), 
    lhs_contracting_dims={1}, rhs_contracting_dims={0}, 
    sharding={devices=[1,8]<=[8]}
}
)";

  // Test with max_windowed_einsum_iteration = 0 (should disable)
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto module,
        PartitionComputation(hlo_string, /*num_partitions=*/8,
                             /*max_windowed_einsum_iteration=*/0,
                             /*threshold_for_windowed_einsum_mib=*/0));

    EXPECT_EQ(
        CountInstructions(*module->entry_computation(), HloOpcode::kWhile), 0)
        << "max_windowed_einsum_iteration=0 should disable windowed einsum";
  }

  // Test with max_windowed_einsum_iteration = 1 (should disable for 8
  // partitions)
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto module,
        PartitionComputation(hlo_string, /*num_partitions=*/8,
                             /*max_windowed_einsum_iteration=*/1,
                             /*threshold_for_windowed_einsum_mib=*/0));

    EXPECT_EQ(
        CountInstructions(*module->entry_computation(), HloOpcode::kWhile), 0)
        << "max_windowed_einsum_iteration=1 should disable windowed einsum for "
           "8 partitions";
  }

  // Test with max_windowed_einsum_iteration = INT64_MAX (should enable)
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto module,
        PartitionComputation(hlo_string, /*num_partitions=*/8,
                             /*max_windowed_einsum_iteration=*/INT64_MAX,
                             /*threshold_for_windowed_einsum_mib=*/0));

    EXPECT_EQ(
        CountInstructions(*module->entry_computation(), HloOpcode::kWhile), 1)
        << "max_windowed_einsum_iteration=INT64_MAX should enable windowed "
           "einsum";
  }
}

}  // namespace
}  // namespace spmd
}  // namespace xla
