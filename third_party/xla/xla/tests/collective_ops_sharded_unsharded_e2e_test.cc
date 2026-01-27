/* Copyright 2023 The OpenXLA Authors.

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
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/regexp.h"

namespace xla {
namespace {

// E2E tests comparing the results of sharded and unsharded execution.
class CollectiveOpsTestE2EShardedUnsharded : public CollectiveOpsE2ETestBase {
 public:
  CollectiveOpsTestE2EShardedUnsharded()
      : CollectiveOpsE2ETestBase(/*memory_size=*/64 * kMB,
                                 /*collectives_memory_size=*/0) {}

  void CollectiveOpsCompareShardedUnsharded(
      const std::string& hlo_text, const int64_t num_partitions = 2,
      bool enable_enzyme_comms_opt = false) {
    const int64_t num_replicas = 1;
    if (hlo_runner_->device_count() < num_replicas * num_partitions) {
      GTEST_SKIP() << "Test requires at least " << num_replicas * num_partitions
                   << " devices (" << hlo_runner_->device_count()
                   << " available)";
    }

    TF_ASSERT_OK_AND_ASSIGN(ExecutionResult ref_execution_result,
                            ExecuteUnsharded(hlo_text));
    const std::vector<Literal>& ref_results = ref_execution_result.results;
    ASSERT_EQ(ref_results.size(), 1);

    TF_ASSERT_OK_AND_ASSIGN(
        ExecutionResult execution_result,
        ExecuteSharded(hlo_text, num_partitions, enable_enzyme_comms_opt));
    const std::vector<Literal>& results = execution_result.results;
    ASSERT_EQ(results.size(), num_partitions);

    ErrorSpec error_spec{1e-4, 1e-4};
    CompareShardedUnsharded(hlo_text, num_partitions, ref_results, results,
                            error_spec);
  }

 private:
  // Execute the unsharded case.
  absl::StatusOr<ExecutionResult> ExecuteUnsharded(
      const std::string& hlo_text) {
    // Create the unsharded reference case by removing the sharding metadata
    // from the HLO string.
    std::string hlo_text_ref = hlo_text;
    RE2::GlobalReplace(&hlo_text_ref, R"(, sharding=\{devices=\[[0-9,]*\].*\})",
                       "");
    RE2::GlobalReplace(&hlo_text_ref, R"(, sharding=\{replicated\})", "");

    HloModuleConfig ref_config = GetModuleConfigForTest();
    ref_config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);

    TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> ref_module,
                        ParseAndReturnVerifiedModule(hlo_text_ref, ref_config));

    ref_module->mutable_config().set_replica_count(1);
    ref_module->mutable_config().set_num_partitions(1);

    const int64_t num_params =
        ref_module->entry_computation()->num_parameters();

    auto fake_args = xla::MakeFakeArguments(ref_module.get()).value();
    std::vector<Literal*> ref_fake_ptrs(num_params);
    for (int i = 0; i < num_params; ++i) {
      ref_fake_ptrs[i] = &fake_args[i];
    }

    return ExecuteReplicated(std::move(ref_module), ref_fake_ptrs);
  }

  // Execute the sharded case.
  absl::StatusOr<ExecutionResult> ExecuteSharded(
      const std::string& hlo_text, int64_t num_partitions,
      bool enable_enzyme_comms_opt = false) {
    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/1, /*num_partitions=*/num_partitions);
    config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);
    if (enable_enzyme_comms_opt) {
      config.mutable_debug_options().set_xla_enable_enzyme_comms_opt(true);
    }
    TF_ASSIGN_OR_RETURN(std::unique_ptr<VerifiedHloModule> module,
                        ParseAndReturnVerifiedModule(hlo_text, config));
    const int64_t num_params = module->entry_computation()->num_parameters();

    std::vector<std::vector<int64_t>> param_dims(num_params);
    std::vector<std::vector<int64_t>> param_dims_per_shard(num_params);
    std::vector<std::vector<int64_t>> param_sharded_dims(num_params);
    for (int i = 0; i < num_params; ++i) {
      auto dimensions = module->entry_computation()
                            ->parameter_instruction(i)
                            ->shape()
                            .dimensions();
      param_dims[i] = std::vector(dimensions.begin(), dimensions.end());
      param_dims_per_shard[i] = param_dims[i];
      HloSharding parameter_sharding =
          module->entry_computation()->parameter_instruction(i)->sharding();
      EvaluateShardedDims(param_dims_per_shard[i], param_sharded_dims[i],
                          parameter_sharding);
    }

    // Slice the tiled inputs to match the prescribed sharding.
    auto fake_args = xla::MakeFakeArguments(module.get()).value();
    std::vector<std::vector<Literal>> fake_args_sliced(num_params);
    std::vector<std::vector<Literal*>> fake_ptrs(num_partitions);
    for (int k = 0; k < num_params; ++k) {
      if (!param_sharded_dims[k].empty()) {
        std::vector<int64_t> lower(param_dims_per_shard[k].size(), 0);
        std::vector<int64_t> upper(param_dims_per_shard[k].begin(),
                                   param_dims_per_shard[k].end());
        for (int i = 0; i < num_partitions; ++i) {
          fake_args_sliced[k].push_back(fake_args[k].Slice(lower, upper));
          for (int m = param_sharded_dims[k].size() - 1; m >= 0; --m) {
            if (upper[param_sharded_dims[k][m]] <
                param_dims[k][param_sharded_dims[k][m]]) {
              upper[param_sharded_dims[k][m]] +=
                  param_dims_per_shard[k][param_sharded_dims[k][m]];
              break;
            }
            upper[param_sharded_dims[k][m]] =
                param_dims_per_shard[k][param_sharded_dims[k][m]];
          }
          absl::c_transform(upper, param_dims_per_shard[k], lower.begin(),
                            std::minus<int64_t>());
        }
      } else {
        fake_args_sliced[k].push_back(fake_args[k].Clone());
      }
    }
    for (int k = 0; k < num_params; ++k) {
      for (int i = 0; i < num_partitions; ++i) {
        if (!param_sharded_dims[k].empty()) {
          fake_ptrs[i].push_back(&fake_args_sliced[k][i]);
        } else {
          fake_ptrs[i].push_back(&fake_args_sliced[k][0]);
        }
      }
    }

    return ExecuteReplicated(std::move(module), fake_ptrs);
  }

  // Slice the unsharded reference results and compare to the sharded case.
  void CompareShardedUnsharded(const std::string& hlo_text,
                               int64_t num_partitions,
                               const std::vector<Literal>& ref_results,
                               const std::vector<Literal>& results,
                               ErrorSpec& error_spec) {
    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/1, /*num_partitions=*/num_partitions);
    config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);

    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    auto dimensions =
        module->entry_computation()->root_instruction()->shape().dimensions();
    std::vector<int64_t> root_dims(dimensions.begin(), dimensions.end());
    std::vector<int64_t> root_dims_per_shard = root_dims;
    std::vector<int64_t> root_sharded_dims;
    {
      HloSharding root_sharding =
          module->entry_computation()->root_instruction()->sharding();
      EvaluateShardedDims(root_dims_per_shard, root_sharded_dims,
                          root_sharding);
    }
    if (!root_sharded_dims.empty()) {
      std::vector<int64_t> lower(root_dims_per_shard.size(), 0);
      std::vector<int64_t> upper(root_dims_per_shard.begin(),
                                 root_dims_per_shard.end());
      for (const Literal& result : results) {
        Literal ref_results_slice = ref_results[0].Slice(lower, upper);
        EXPECT_TRUE(
            LiteralTestUtil::Near(ref_results_slice, result, error_spec));
        for (int m = root_sharded_dims.size() - 1; m >= 0; --m) {
          if (upper[root_sharded_dims[m]] < root_dims[root_sharded_dims[m]]) {
            upper[root_sharded_dims[m]] +=
                root_dims_per_shard[root_sharded_dims[m]];
            break;
          }
          upper[root_sharded_dims[m]] =
              root_dims_per_shard[root_sharded_dims[m]];
        }
        absl::c_transform(upper, root_dims_per_shard, lower.begin(),
                          std::minus<int64_t>());
      }
    } else {
      EXPECT_TRUE(
          LiteralTestUtil::Near(ref_results[0], results[0], error_spec));
    }
  }

  void EvaluateShardedDims(std::vector<int64_t>& dims_per_shard,
                           std::vector<int64_t>& sharded_dims,
                           const HloSharding& sharding) {
    if (!sharding.IsReplicated()) {
      for (int k = 0; k < sharding.num_dimensions(); ++k) {
        if (sharding.dimension(k) > 1) {
          dims_per_shard[k] /= sharding.dimension(k);
          sharded_dims.push_back(k);
        }
      }
    }
  }
};

TEST_F(CollectiveOpsTestE2EShardedUnsharded, DotBatchAndBatch) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f32[4,16,8]{2,1,0}, f32[4,4,8]{2,1,0})->f32[4,16,4]{2,1,0}}, num_partitions=2

ENTRY entry {
  lhs = f32[4,16,8]{2,1,0} parameter(0), sharding={devices=[2,1,1]<=[2]}
  rhs = f32[4,4,8]{2,1,0} parameter(1), sharding={devices=[2,1,1]<=[2]}
  ROOT dot = f32[4,16,4]{2,1,0} dot(lhs, rhs), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={2}, sharding={devices=[1,2,1]<=[2]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text);
}

// This is an execution test for the example in Option 2 in go/dus-spmd. This
// test should pass regardless of which DUS SPMD implementation option is used.
TEST_F(CollectiveOpsTestE2EShardedUnsharded,
       DusSingleDimensionInPartitionMode) {
  const std::string hlo_text = R"(
    HloModule module, entry_computation_layout={(s32[16]{0}, s32[8]{0})->s32[16]{0}}, num_partitions=4

    ENTRY entry {
      %input = s32[16] parameter(0), sharding={devices=[4]<=[4]}
      %update = s32[8] parameter(1), sharding={devices=[4]<=[4]}
      %c3 = s32[] constant(3)
      ROOT %dynamic-update-slice = s32[16] dynamic-update-slice(%input, %update, %c3), sharding={devices=[4]<=[4]}
    })";
  CollectiveOpsCompareShardedUnsharded(hlo_text, /*num_partitions=*/4,
                                       /*enable_enzyme_comms_opt=*/true);
  // This test should pass regardless if enzyme comms opt is enabled or not.
  CollectiveOpsCompareShardedUnsharded(hlo_text, /*num_partitions=*/4,
                                       /*enable_enzyme_comms_opt=*/false);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded,
       KeepPartitionedNonSlicedDimensionWithConstantIndices) {
  const std::string hlo_text = R"(
    HloModule module, entry_computation_layout={(bf16[2,24,24,32]{3,2,1,0}, bf16[2,4,4,32]{3,2,1,0})->bf16[2,56,56,32]{3,2,1,0}}, num_partitions=8

    ENTRY entry {
      p1 = bf16[2,24,24,32]{3,2,1,0} parameter(0), sharding={replicated}
      p2 = bf16[2,4,4,32]{3,2,1,0} parameter(1), sharding={replicated}
      c1 = bf16[2,24,24,32]{3,2,1,0} copy(p1), sharding={devices=[2,2,2,1]<=[8]}
      c2 = bf16[2,4,4,32]{3,2,1,0} copy(p2), sharding={devices=[2,2,2,1]<=[8]}
      constant.1163 = bf16[] constant(0), sharding={replicated}
      constant.1165 = s32[] constant(0), sharding={replicated}
      pad.179 = bf16[2,56,56,32]{3,2,1,0} pad(c1, constant.1163), padding=0_0x16_16x16_16x0_0, sharding={devices=[2,2,2,1]<=[8]}
      add.439 = bf16[2,4,4,32]{3,2,1,0} add(c2, c2), sharding={devices=[2,2,2,1]<=[8]}
      constant.1070 = s32[] constant(48), sharding={replicated}
      dynamic-update-slice.128 = bf16[2,56,56,32]{3,2,1,0} dynamic-update-slice(pad.179, add.439, constant.1165, constant.1070, constant.1070, /*index=5*/constant.1165), sharding={devices=[2,2,2,1]<=[8]}
      ROOT c = bf16[2,56,56,32]{3,2,1,0} copy(dynamic-update-slice.128), sharding={devices=[2,2,2,1]<=[8]}
    })";
  CollectiveOpsCompareShardedUnsharded(hlo_text, /*num_partitions=*/8,
                                       /*enable_enzyme_comms_opt=*/true);
  CollectiveOpsCompareShardedUnsharded(hlo_text, /*num_partitions=*/8,
                                       /*enable_enzyme_comms_opt=*/false);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded, DotBatchAndNonContracting) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f32[4,16,8]{2,1,0}, f32[4,4,8]{2,1,0})->f32[4,16,4]{2,1,0}}, num_partitions=2

ENTRY entry {
  lhs = f32[4,16,8]{2,1,0} parameter(0), sharding={devices=[2,1,1]<=[2]}
  rhs = f32[4,4,8]{2,1,0} parameter(1), sharding={devices=[1,2,1]<=[2]}
  ROOT dot = f32[4,16,4]{2,1,0} dot(lhs, rhs), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={2}, sharding={devices=[2,1,1]<=[2]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded, DotContractingAndContracting) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f32[16,8]{1,0}, f32[4,8]{1,0})->f32[16,4]{1,0}}, num_partitions=2

ENTRY entry {
  lhs = f32[16,8]{1,0} parameter(0), sharding={devices=[1,2]<=[2]}
  rhs = f32[4,8]{1,0} parameter(1), sharding={devices=[1,2]<=[2]}
  ROOT dot = f32[16,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={1}, sharding={devices=[2,1]<=[2]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded, DotNonContractingAndContracting) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f32[16,8]{1,0}, f32[4,8]{1,0})->f32[16,4]{1,0}}, num_partitions=2

ENTRY entry {
  lhs = f32[16,8]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  rhs = f32[4,8]{1,0} parameter(1), sharding={devices=[1,2]<=[2]}
  ROOT dot = f32[16,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={1}, sharding={devices=[2,1]<=[2]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded, DotContractingAndReplicated) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f32[16,8]{1,0}, f32[4,8]{1,0})->f32[16,4]{1,0}}, num_partitions=2

ENTRY entry {
  lhs = f32[16,8]{1,0} parameter(0), sharding={devices=[1,2]<=[2]}
  rhs = f32[4,8]{1,0} parameter(1), sharding={replicated}
  ROOT dot = f32[16,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={1}, sharding={devices=[2,1]<=[2]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded, DotReplicatedAndReplicated) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f32[4,4]{1,0}, f32[1,4]{1,0})->f32[4,1]{1,0}}, num_partitions=2

ENTRY entry {
  lhs = f32[4,4]{1,0} parameter(0), sharding={replicated}
  rhs = f32[1,4]{1,0} parameter(1), sharding={replicated}
  ROOT dot = f32[4,1]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={1}, sharding={devices=[2,1]<=[2]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded,
       DotContractingNonContractingAndContractingNonContracting) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f32[16,8]{1,0}, f32[4,8]{1,0})->f32[16,4]{1,0}}, num_partitions=4

ENTRY entry {
  lhs = f32[16,8]{1,0} parameter(0), sharding={devices=[2,2]<=[4]}
  rhs = f32[4,8]{1,0} parameter(1), sharding={devices=[2,2]<=[4]}
  ROOT dot = f32[16,4]{1,0} dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={1}, sharding={devices=[2,2]<=[4]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text, /*num_partitions=*/4);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded, BlockScaledDotBatchAndBatch) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f8e4m3fn[4,16,64]{2,1,0}, f8e8m0fnu[4,16,2]{2,1,0}, f8e4m3fn[4,4,64]{2,1,0}, f8e8m0fnu[4,4,2]{2,1,0})->f32[4,16,4]{2,1,0}}, num_partitions=2

ENTRY entry {
  lhs = f8e4m3fn[4,16,64]{2,1,0} parameter(0), sharding={devices=[2,1,1]<=[2]}
  lhs_scale = f8e8m0fnu[4,16,2]{2,1,0} parameter(1), sharding={devices=[2,1,1]<=[2]}
  rhs = f8e4m3fn[4,4,64]{2,1,0} parameter(2), sharding={devices=[2,1,1]<=[2]}
  rhs_scale = f8e8m0fnu[4,4,2]{2,1,0} parameter(3), sharding={devices=[2,1,1]<=[2]}
  ROOT block_scaled_dot = f32[4,16,4]{2,1,0} custom-call(lhs, rhs, lhs_scale, rhs_scale), custom_call_target="__op$block_scaled_dot", sharding={devices=[1,2,1]<=[2]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded,
       BlockScaledDotBatchAndNonContracting) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f8e4m3fn[4,16,64]{2,1,0}, f8e8m0fnu[4,16,2]{2,1,0}, f8e4m3fn[4,4,64]{2,1,0}, f8e8m0fnu[4,4,2]{2,1,0})->f32[4,16,4]{2,1,0}}, num_partitions=2

ENTRY entry {
  lhs = f8e4m3fn[4,16,64]{2,1,0} parameter(0), sharding={devices=[2,1,1]<=[2]}
  lhs_scale = f8e8m0fnu[4,16,2]{2,1,0} parameter(1), sharding={devices=[2,1,1]<=[2]}
  rhs = f8e4m3fn[4,4,64]{2,1,0} parameter(2), sharding={devices=[1,2,1]<=[2]}
  rhs_scale = f8e8m0fnu[4,4,2]{2,1,0} parameter(3), sharding={devices=[1,2,1]<=[2]}
  ROOT block_scaled_dot = f32[4,16,4]{2,1,0} custom-call(lhs, rhs, lhs_scale, rhs_scale), custom_call_target="__op$block_scaled_dot", sharding={devices=[2,1,1]<=[2]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded,
       BlockScaledDotContractingAndContracting) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f8e4m3fn[16,64]{1,0}, f8e8m0fnu[16,2]{1,0}, f8e4m3fn[4,64]{1,0}, f8e8m0fnu[4,2]{1,0})->f32[16,4]{1,0}}, num_partitions=2

ENTRY entry {
  lhs = f8e4m3fn[16,64]{1,0} parameter(0), sharding={devices=[1,2]<=[2]}
  lhs_scale = f8e8m0fnu[16,2]{1,0} parameter(1), sharding={devices=[1,2]<=[2]}
  rhs = f8e4m3fn[4,64]{1,0} parameter(2), sharding={devices=[1,2]<=[2]}
  rhs_scale = f8e8m0fnu[4,2]{1,0} parameter(3), sharding={devices=[1,2]<=[2]}
  ROOT block_scaled_dot = f32[16,4]{1,0} custom-call(lhs, rhs, lhs_scale, rhs_scale), custom_call_target="__op$block_scaled_dot", sharding={devices=[2,1]<=[2]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded,
       BlockScaledDotNonContractingAndContracting) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f8e4m3fn[16,128]{1,0}, f8e8m0fnu[16,4]{1,0}, f8e4m3fn[4,128]{1,0}, f8e8m0fnu[4,4]{1,0})->f32[16,4]{1,0}}, num_partitions=2

ENTRY entry {
  lhs = f8e4m3fn[16,128]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  lhs_scale = f8e8m0fnu[16,4]{1,0} parameter(1), sharding={devices=[2,1]<=[2]}
  rhs = f8e4m3fn[4,128]{1,0} parameter(2), sharding={devices=[1,2]<=[2]}
  rhs_scale = f8e8m0fnu[4,4]{1,0} parameter(3), sharding={devices=[1,2]<=[2]}
  ROOT block_scaled_dot = f32[16,4]{1,0} custom-call(lhs, rhs, lhs_scale, rhs_scale), custom_call_target="__op$block_scaled_dot", sharding={devices=[2,1]<=[2]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded,
       BlockScaledDotContractingAndReplicated) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f8e4m3fn[16,128]{1,0}, f8e8m0fnu[16,4]{1,0}, f8e4m3fn[4,128]{1,0}, f8e8m0fnu[4,4]{1,0})->f32[16,4]{1,0}}, num_partitions=2

ENTRY entry {
  lhs = f8e4m3fn[16,128]{1,0} parameter(0), sharding={devices=[1,2]<=[2]}
  lhs_scale = f8e8m0fnu[16,4]{1,0} parameter(1), sharding={devices=[1,2]<=[2]}
  rhs = f8e4m3fn[4,128]{1,0} parameter(2), sharding={replicated}
  rhs_scale = f8e8m0fnu[4,4]{1,0} parameter(3), sharding={replicated}
  ROOT block_scaled_dot = f32[16,4]{1,0} custom-call(lhs, rhs, lhs_scale, rhs_scale), custom_call_target="__op$block_scaled_dot", sharding={devices=[2,1]<=[2]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded,
       BlockScaledDotReplicatedAndReplicated) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f8e4m3fn[4,128]{1,0}, f8e8m0fnu[4,4], f8e4m3fn[1,128]{1,0}, f8e8m0fnu[1,4]{1,0})->f32[4,1]{1,0}}, num_partitions=2

ENTRY entry {
  lhs = f8e4m3fn[4,128]{1,0} parameter(0), sharding={replicated}
  lhs_scale = f8e8m0fnu[4,4]{1,0} parameter(1), sharding={replicated}
  rhs = f8e4m3fn[1,128]{1,0} parameter(2), sharding={replicated}
  rhs_scale = f8e8m0fnu[1,4]{1,0} parameter(3), sharding={replicated}
  ROOT block_scaled_dot = f32[4,1]{1,0} custom-call(lhs, rhs, lhs_scale, rhs_scale), custom_call_target="__op$block_scaled_dot", sharding={devices=[2,1]<=[2]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text);
}

TEST_F(CollectiveOpsTestE2EShardedUnsharded,
       BlockScaledDotContractingNonContractingAndContractingNonContracting) {
  const std::string hlo_text = R"(
HloModule module, entry_computation_layout={(f8e4m3fn[8,128]{1,0}, f8e8m0fnu[8,4]{1,0}, f8e4m3fn[4,128]{1,0}, f8e8m0fnu[4,4]{1,0})->f32[8,4]{1,0}}, num_partitions=4

ENTRY entry {
  lhs = f8e4m3fn[8,128]{1,0} parameter(0), sharding={devices=[2,2]<=[4]}
  lhs_scale = f8e8m0fnu[8,4]{1,0} parameter(1), sharding={devices=[2,2]<=[4]}
  rhs = f8e4m3fn[4,128]{1,0} parameter(2), sharding={devices=[2,2]<=[4]}
  rhs_scale = f8e8m0fnu[4,4]{1,0} parameter(3), sharding={devices=[2,2]<=[4]}
  ROOT dot = f32[8,4]{1,0} custom-call(lhs, rhs, lhs_scale, rhs_scale), custom_call_target="__op$block_scaled_dot", sharding={devices=[2,2]<=[4]}
})";
  CollectiveOpsCompareShardedUnsharded(hlo_text, /*num_partitions=*/4);
}

}  // namespace
}  // namespace xla
