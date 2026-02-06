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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::match;
using ::testing::NotNull;

bool IsAsync(const HloInstruction* inst) {
  return !inst->backend_config<gpu::GpuBackendConfig>()
              .value()
              .collective_backend_config()
              .is_sync();
}

class CollectiveOpsTestE2E : public CollectiveOpsE2ETestBase {
 public:
  explicit CollectiveOpsTestE2E(size_t memory_size = 128 * kMB,
                                size_t collectives_memory_size = 0)
      : CollectiveOpsE2ETestBase(memory_size, collectives_memory_size) {}

  bool HasFp8Support() {
    if (Capability().IsCuda()) {
      return Capability().cuda_compute_capability()->IsAtLeast(8, 9);
    }
    return Capability().rocm_compute_capability()->has_fp8_support() &&
           GetDebugOptionsForTest().xla_gpu_enable_cublaslt();
  }

  void CollectiveOpsVerifyF8Matmul(absl::string_view hlo_text,
                                   const DebugOptions& options) {
    if (!HasFp8Support()) {
      return;
    }
    const int64_t kNumReplicas = 1;
    const int64_t kNumPartitions = 4;
    if (hlo_runner_->device_count() < kNumReplicas * kNumPartitions) {
      GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                   << " devices (" << hlo_runner_->device_count()
                   << " available)";
    }

    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
    config.set_debug_options(options);
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_text, config));

    TF_ASSERT_OK_AND_ASSIGN(auto executable, hlo_runner_->CreateExecutable(
                                                 std::move(module),
                                                 /*run_hlo_passes=*/true));
    TF_ASSERT_OK_AND_ASSIGN(
        const HloModule* const hlo_module,
        hlo_runner_->HloModuleFromWrapped(executable.get()));
    std::vector<HloInstruction*> gemm_ops =
        FindInstructions(hlo_module, HloOpcode::kCustomCall);
    for (HloInstruction* gemm_op : gemm_ops) {
      EXPECT_EQ(gemm_op->custom_call_target(), "__cublas$lt$matmul$f8");
    }
  }
};

class AsyncCollectiveOps : public CollectiveOpsWithFlagsBase,
                           public ::testing::WithParamInterface<bool> {
 public:
  AsyncCollectiveOps()
      : CollectiveOpsWithFlagsBase(/*enable_async=*/GetParam(),
                                   /*enable_p2p_memcpy=*/false,
                                   /*memory_size=*/8 * kGB,
                                   /*collectives_memory_size=*/0) {}
};

class MemcpyCollectiveOps : public CollectiveOpsWithFlagsBase,
                            public ::testing::WithParamInterface<bool> {
 public:
  MemcpyCollectiveOps()
      : CollectiveOpsWithFlagsBase(/*enable_async=*/true,
                                   /*enable_p2p_memcpy=*/GetParam(),
                                   /*memory_size=*/32 * kMB,
                                   /*collectives_memory_size=*/0) {}
};

class AsyncMemcpyCollectiveOps
    : public CollectiveOpsWithFlagsBase,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {
 public:
  AsyncMemcpyCollectiveOps()
      : CollectiveOpsWithFlagsBase(
            /*enable_async=*/std::get<0>(GetParam()),
            /*enable_p2p_memcpy=*/std::get<1>(GetParam()),
            /*memory_size=*/32 * kMB,
            /*collectives_memory_size=*/0) {}
};

std::string GetAsyncTestName(bool is_async) {
  return is_async ? "async" : "sync";
}

std::string GetMemcpyTestName(bool is_memcpy) {
  return is_memcpy ? "memcpy" : "nccl";
}

std::string GetAsyncTestSuiteName(const ::testing::TestParamInfo<bool>& info) {
  return GetAsyncTestName(info.param);
}

std::string GetMemcpyTestSuiteName(const ::testing::TestParamInfo<bool>& info) {
  return GetMemcpyTestName(info.param);
}

std::string GetAsyncMemcpyTestSuiteName(
    const ::testing::TestParamInfo<std::tuple<bool, bool>>& info) {
  return absl::StrCat(GetAsyncTestName(std::get<0>(info.param)), "_",
                      GetMemcpyTestName(std::get<1>(info.param)));
}

TEST_P(AsyncCollectiveOps, AsyncAllReduce) {
  const absl::string_view kModuleStr = R"(
      HloModule test

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        ROOT all-reduce = u32[] all-reduce(id), to_apply=apply_op
      }
    )";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  const bool enable_async_all_reduce = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* all_reduce_start =
      FindInstruction(hlo_module, HloOpcode::kAllReduceStart);
  HloInstruction* all_reduce_done =
      FindInstruction(hlo_module, HloOpcode::kAllReduceDone);
  EXPECT_THAT(all_reduce_start, NotNull());
  EXPECT_THAT(all_reduce_done, NotNull());
  EXPECT_EQ(IsAsync(all_reduce_start), enable_async_all_reduce);

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  // sum [0, num_devices)
  const uint32_t expected = kNumReplicas * (kNumReplicas - 1) / 2;
  for (int i = 0; i < kNumReplicas; ++i) {
    LiteralTestUtil::ExpectR0Equal<uint32_t>(expected, results[i]);
  }
}

TEST_P(AsyncCollectiveOps, AsyncAllGather) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[1, 2] broadcast(id), dimensions={}
    a0 = u32[1, 2] constant({{10, 15}})
    a1 = u32[1, 2] add(id2, a0)
    allgather = u32[2, 2] all-gather(a1), dimensions={0}
    ROOT out = u32[4] reshape(allgather)
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  const bool enable_async_all_gather = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* all_gather_start =
      FindInstruction(hlo_module, HloOpcode::kAllGatherStart);
  HloInstruction* all_gather_done =
      FindInstruction(hlo_module, HloOpcode::kAllGatherDone);
  EXPECT_THAT(all_gather_start, NotNull());
  EXPECT_THAT(all_gather_done, NotNull());
  EXPECT_EQ(IsAsync(all_gather_start), enable_async_all_gather);

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (const Literal& result : results) {
    LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, result);
  }
}

TEST_P(AsyncCollectiveOps, AsyncAllGatherMixedTypes) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[1, 2] broadcast(id), dimensions={}
    a0 = u32[1, 2] constant({{10, 15}})
    a1 = u32[1, 2] add(id2, a0)
    a2 = f32[1, 2] convert(a1)
    allgather = (u32[2, 2], f32[2,2]) all-gather(a1, a2), dimensions={0}
    gte0 = u32[2,2] get-tuple-element(allgather), index=0
    gte1 = f32[2,2] get-tuple-element(allgather), index=1
    out0 = u32[4] reshape(gte0)
    out1 = f32[4] reshape(gte1)
    ROOT out = (u32[4], f32[4]) tuple(out0, out1)
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  const bool enable_async_all_gather = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* all_gather_start =
      FindInstruction(hlo_module, HloOpcode::kAllGatherStart);
  HloInstruction* all_gather_done =
      FindInstruction(hlo_module, HloOpcode::kAllGatherDone);
  EXPECT_THAT(all_gather_start, NotNull());
  EXPECT_THAT(all_gather_done, NotNull());
  EXPECT_EQ(IsAsync(all_gather_start), enable_async_all_gather);

  std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (Literal& result : results) {
    std::vector<Literal> tuple_results = result.DecomposeTuple();
    LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16},
                                             tuple_results[0]);
    LiteralTestUtil::ExpectR1Equal<float>({10.0, 15.0, 11.0, 16.0},
                                          tuple_results[1]);
  }
}

TEST_P(AsyncCollectiveOps, AsyncCollectiveBroadcast) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    replica = u32[] replica-id()
    ten = u32[] constant(10)
    sum = u32[] add(replica, ten)
    p = u32[2] broadcast(sum), dimensions={}
    bcast = u32[2] collective-broadcast(p), replica_groups={{1, 0}}
    ROOT res = copy(bcast)
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  const bool enable_async_collective_broadcast = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* cb_start =
      FindInstruction(hlo_module, HloOpcode::kAsyncStart);
  HloInstruction* cb_done = FindInstruction(hlo_module, HloOpcode::kAsyncDone);
  EXPECT_THAT(cb_start, NotNull());
  EXPECT_THAT(cb_done, NotNull());
  EXPECT_EQ(IsAsync(cb_start), enable_async_collective_broadcast);

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 11}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 11}, results[1]);
}

TEST_P(AsyncCollectiveOps, AsyncCollectivePermute) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    replica = u32[] replica-id()
    ten = u32[] constant(10)
    sum = u32[] add(replica, ten)
    p = u32[2] broadcast(sum), dimensions={}
    permute = u32[2] collective-permute(p), source_target_pairs={{1,0}, {0,1}}
    ROOT copy = u32[2] copy(permute)
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  const bool enable_async_collective_permute = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* cp_start =
      FindInstruction(hlo_module, HloOpcode::kCollectivePermuteStart);
  HloInstruction* cp_done =
      FindInstruction(hlo_module, HloOpcode::kCollectivePermuteDone);
  EXPECT_THAT(cp_start, NotNull());
  EXPECT_THAT(cp_done, NotNull());
  EXPECT_EQ(IsAsync(cp_start), enable_async_collective_permute);

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 11}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 10}, results[1]);
}

TEST_P(AsyncCollectiveOps, CombinedCollectivePermute) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    replica = u32[] replica-id()
    ten = u32[] constant(10)
    sum = u32[] add(replica, ten)
    replica.1 = u32[2] broadcast(replica), dimensions={}
    sum.1 = u32[2] broadcast(sum), dimensions={}
    permute = (u32[2], u32[2]) collective-permute(replica.1, sum.1), source_target_pairs={{1,0}, {0,1}}
    gte0 = get-tuple-element(permute), index=0
    gte1 = get-tuple-element(permute), index=1
    ROOT concat = u32[4] concatenate(gte0, gte1), dimensions={0}
  }
  )";
  const int64_t kNumReplicas = 2;
  const bool enable_async_collective_permute = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* cp_start =
      FindInstruction(hlo_module, HloOpcode::kCollectivePermuteStart);
  HloInstruction* cp_done =
      FindInstruction(hlo_module, HloOpcode::kCollectivePermuteDone);
  EXPECT_THAT(cp_start, NotNull());
  EXPECT_THAT(cp_done, NotNull());
  EXPECT_EQ(IsAsync(cp_start), enable_async_collective_permute);

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 11, 11}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 0, 10, 10}, results[1]);
}

TEST_P(AsyncCollectiveOps, CollectivePermuteCombiner) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    replica = u32[] replica-id()
    ten = u32[] constant(10)
    sum = u32[] add(replica, ten)
    replica.1 = u32[2] broadcast(replica), dimensions={}
    sum.1 = u32[2] broadcast(sum), dimensions={}
    replica.2 = u32[2] add(replica.1, replica.1)
    permute.0 = u32[2] collective-permute(replica.1), source_target_pairs={{0,1}, {1, 2}, {2, 3}, {3, 0}}
    permute.1 = u32[2] collective-permute(replica.2), source_target_pairs={{0,1}, {1, 2}, {2, 3}, {3, 0}}
    permute.2 = u32[2] collective-permute(sum.1), source_target_pairs={{0,1}, {1, 2}, {2, 3}, {3, 0}}
    ROOT concat = u32[6] concatenate(permute.0, permute.1, permute.2), dimensions={0}
  }
  )";
  const int64_t kNumReplicas = 4;
  const bool enable_async_collective_permute = GetParam();
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* cp_start =
      FindInstruction(hlo_module, HloOpcode::kCollectivePermuteStart);
  HloInstruction* cp_done =
      FindInstruction(hlo_module, HloOpcode::kCollectivePermuteDone);

  EXPECT_THAT(cp_start, NotNull());
  // Count the number of collective permute start instructions in the module
  int cp_start_count = 0;
  for (const auto& computation : hlo_module->computations()) {
    for (const auto& instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCollectivePermuteStart) {
        cp_start_count++;
      }
    }
  }
  EXPECT_EQ(cp_start_count, 1)
      << "Expected exactly one CollectivePermuteStart instruction";

  // Expect 3 collective permute instructions combined into one.
  EXPECT_EQ(cp_start->operand_count(), 3);
  EXPECT_THAT(cp_done, NotNull());
  EXPECT_EQ(IsAsync(cp_start), enable_async_collective_permute);

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({3, 3, 6, 6, 13, 13}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 0, 0, 0, 10, 10}, results[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2, 11, 11}, results[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 2, 4, 4, 12, 12}, results[3]);
}

TEST_P(AsyncCollectiveOps, AsyncReduceScatter) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  add {
    lhs = u32[] parameter(0)
    rhs = u32[] parameter(1)
    ROOT add = u32[] add(lhs, rhs)
  }

  ENTRY main {
    c0 = u32[8] constant({1, 2, 3, 4, 5, 6, 7, 8})
    c1 = u32[8] constant({10, 11, 12, 13, 14, 15, 16, 17})
    zero = u32[] constant(0)
    id = u32[] replica-id()
    p = pred[] compare(id, zero), direction=EQ
    pb = pred[8] broadcast(p), dimensions={}
    // data = c0 for replica 0 and c1 for replica 1
    data = u32[8] select(pb, c0, c1)
    ROOT ars = u32[4] reduce-scatter(data), replica_groups={},
                      dimensions={0}, to_apply=add
  }
  )";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  const bool enable_async_reduce_scatter = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* rs_start =
      FindInstruction(hlo_module, HloOpcode::kAsyncStart);
  HloInstruction* rs_done = FindInstruction(hlo_module, HloOpcode::kAsyncDone);
  ASSERT_THAT(rs_start, NotNull());
  ASSERT_THAT(rs_done, NotNull());
  HloAsyncInstruction* rs_start_async = Cast<HloAsyncInstruction>(rs_start);
  EXPECT_EQ(rs_start_async->async_wrapped_opcode(), HloOpcode::kReduceScatter);
  EXPECT_EQ(IsAsync(rs_start), enable_async_reduce_scatter);

  const std::vector<Literal>& results = execution_result.results;
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 13, 15, 17}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({19, 21, 23, 25}, results[1]);
}

TEST_P(AsyncCollectiveOps, AsyncAllToAllWithSplitDim) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2] broadcast(id), dimensions={}
    a0 = u32[2] constant({10, 15})
    a1 = u32[2] add(id2, a0)
    ROOT a2a = u32[2] all-to-all(u32[2] a1), dimensions={0}
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  const bool enable_async_all_to_all = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* a2a_start =
      FindInstruction(hlo_module, HloOpcode::kAsyncStart);
  HloInstruction* a2a_done = FindInstruction(hlo_module, HloOpcode::kAsyncDone);
  ASSERT_THAT(a2a_start, NotNull());
  ASSERT_THAT(a2a_done, NotNull());
  HloAsyncInstruction* a2a_start_async = Cast<HloAsyncInstruction>(a2a_start);
  EXPECT_EQ(a2a_start_async->async_wrapped_opcode(), HloOpcode::kAllToAll);
  EXPECT_EQ(IsAsync(a2a_start), enable_async_all_to_all);

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 11}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({15, 16}, results[1]);
}

TEST_F(CollectiveOpsTestE2E, AsyncAllToAllMemCpyWithSplitDim) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2, 2] broadcast(id), dimensions={}
    a0 = u32[2, 2] constant({{10, 15}, {20, 25}})
    a1 = u32[2, 2] add(id2, a0)
    all2all = u32[2, 2] all-to-all(a1), dimensions={0}
    ROOT out = u32[4] reshape(all2all)
  }
  )";
  const int64_t kNumReplicas = 2;

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options().set_xla_gpu_use_memcpy_local_p2p(true);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* executable_module = execution_result.optimized_module;
  // Verify that the all-to-all is not decomposed into a tuple all-to-all.
  const HloInstruction* all_to_all =
      FindInstruction(executable_module, HloOpcode::kAllToAll);
  EXPECT_THAT(all_to_all, op::Shape("u32[2, 2]"));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({20, 25, 21, 26}, results[1]);
}

TEST_P(AsyncCollectiveOps, AsyncAllToAllWithoutSplitDim) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2] broadcast(id), dimensions={}
    a0 = u32[2] constant({10, 15})
    a1 = u32[2] add(id2, a0)
    a2 = u32[2] constant({4, 4})
    a3 = u32[2] multiply(a1, a2)
    // r0 : a1 = {10, 15}, a2 = {40, 60)
    // r1 : a1 = {11, 16}, a1 = {44, 64}
    // r0: a2a element 0 = {10, 15}, a2a element 1 = {11, 16}
    // r0: a2a element 0 = {40, 60}, a2a element 1 = {44, 64}
    a2a = (u32[2], u32[2]) all-to-all(u32[2] a1, u32[2] a3), replica_groups={{0,1}}
    gte0 = get-tuple-element(a2a), index=0
    gte1 = get-tuple-element(a2a), index=1
    ROOT x = u32[4] concatenate(gte0, gte1), dimensions={0}
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  const bool enable_async_all_to_all = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* a2a_start =
      FindInstruction(hlo_module, HloOpcode::kAsyncStart);
  HloInstruction* a2a_done = FindInstruction(hlo_module, HloOpcode::kAsyncDone);
  ASSERT_THAT(a2a_start, NotNull());
  ASSERT_THAT(a2a_done, NotNull());
  HloAsyncInstruction* a2a_start_async = Cast<HloAsyncInstruction>(a2a_start);
  EXPECT_EQ(a2a_start_async->async_wrapped_opcode(), HloOpcode::kAllToAll);
  EXPECT_EQ(IsAsync(a2a_start_async), enable_async_all_to_all);

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({40, 60, 44, 64}, results[1]);
}

TEST_P(AsyncCollectiveOps, AsyncAllToAllMemCpyWithoutSplitDim) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2] broadcast(id), dimensions={}
    a0 = u32[2] constant({10, 15})
    a1 = u32[2] add(id2, a0)
    a2 = u32[2] constant({4, 4})
    a3 = u32[2] multiply(a1, a2)
    // r0 : a1 = {10, 15}, a2 = {40, 60)
    // r1 : a1 = {11, 16}, a1 = {44, 64}
    // r0: a2a element 0 = {10, 15}, a2a element 1 = {11, 16}
    // r0: a2a element 0 = {40, 60}, a2a element 1 = {44, 64}
    a2a = (u32[2], u32[2]) all-to-all(u32[2] a1, u32[2] a3), replica_groups={{0,1}}
    gte0 = get-tuple-element(a2a), index=0
    gte1 = get-tuple-element(a2a), index=1
    ROOT x = u32[4] concatenate(gte0, gte1), dimensions={0}
  }
  )";
  const int64_t kNumReplicas = 2;
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options().set_xla_gpu_use_memcpy_local_p2p(true);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({40, 60, 44, 64}, results[1]);
}

TEST_P(AsyncCollectiveOps, AsyncAllToAllNumberOfElementsLargerThanInt32Max) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    id_u8 = u8[] convert(id)
    a0 = u8[2,32768,32768] broadcast(id_u8), dimensions={}
    ROOT a2a = u8[2,32768,32768] all-to-all(u8[2,32768,32768] a0),
      replica_groups={{0,1}}, dimensions={0}
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  const bool enable_async_all_to_all = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* a2a_start =
      FindInstruction(hlo_module, HloOpcode::kAsyncStart);
  HloInstruction* a2a_done = FindInstruction(hlo_module, HloOpcode::kAsyncDone);
  ASSERT_THAT(a2a_start, NotNull());
  ASSERT_THAT(a2a_done, NotNull());
  HloAsyncInstruction* a2a_start_async = Cast<HloAsyncInstruction>(a2a_start);
  EXPECT_EQ(a2a_start_async->async_wrapped_opcode(), HloOpcode::kAllToAll);
  EXPECT_EQ(IsAsync(a2a_start_async), enable_async_all_to_all);

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  // Sanity check only a few elements in each result, because checking all 2GB
  // would take too long.
  EXPECT_EQ(results[0].Get<uint8_t>({0, 0, 0}), 0);
  EXPECT_EQ(results[0].Get<uint8_t>({1, 0, 0}), 1);

  EXPECT_EQ(results[1].Get<uint8_t>({0, 0, 0}), 0);
  EXPECT_EQ(results[1].Get<uint8_t>({1, 0, 0}), 1);
}

TEST_P(AsyncCollectiveOps, AsyncRaggedAllToAll_2GPUs_BF16) {
  const absl::string_view kModuleStr = R"(
HloModule test
ENTRY entry {
  input = bf16[2] constant({4., 8.})
  output = bf16[2] constant({0., 0.})
  input_offsets = s64[2] constant({0, 1})
  send_sizes = s64[2] constant({1, 1})
  c0 = s64[2] constant({0, 0})
  replica_id = u32[] replica-id()
  replica_id_s64 = s64[] convert(replica_id)
  broadcast_replica_id = s64[2] broadcast(replica_id_s64), dimensions={}
  output_offsets = s64[2] add(broadcast_replica_id, c0)
  recv_sizes = s64[2] constant({1, 1})
  ROOT ragged-all-to-all = bf16[2] ragged-all-to-all(input, output,
    input_offsets, send_sizes, output_offsets, recv_sizes),
    replica_groups={{0,1}}
}
)";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  const bool enable_async_ragged_all_to_all = GetParam();
  HloInstruction* ra2a_start =
      FindInstruction(hlo_module, HloOpcode::kAsyncStart);
  HloInstruction* ra2a_done =
      FindInstruction(hlo_module, HloOpcode::kAsyncDone);
  ASSERT_THAT(ra2a_start, NotNull());
  ASSERT_THAT(ra2a_done, NotNull());
  EXPECT_EQ(IsAsync(ra2a_start), enable_async_ragged_all_to_all);

  HloAsyncInstruction* ra2a_start_async = Cast<HloAsyncInstruction>(ra2a_start);
  EXPECT_EQ(ra2a_start_async->async_wrapped_opcode(),
            HloOpcode::kRaggedAllToAll);

  // Check that the element type of ragged-all-to-all was not changed from bf16.
  EXPECT_EQ(
      ra2a_start_async->async_wrapped_instruction()->shape().element_type(),
      BF16);

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  const bfloat16 four = static_cast<bfloat16>(4.);
  const bfloat16 eight = static_cast<bfloat16>(8.);
  LiteralTestUtil::ExpectR1Equal<bfloat16>({four, four}, results[0]);
  LiteralTestUtil::ExpectR1Equal<bfloat16>({eight, eight}, results[1]);
}

TEST_P(AsyncMemcpyCollectiveOps, AsyncAllToAllMultipleReplicaGroups) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2] broadcast(id), dimensions={}
    a0 = u32[2] constant({10, 20})
    a1 = u32[2] add(id2, a0)
    ROOT a2a = u32[2] all-to-all(u32[2] a1), dimensions={0}, replica_groups={{0,3},{1,2}}
  }
  )";
  const int64_t kNumReplicas = 4;
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 13}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 12}, results[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({21, 22}, results[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({20, 23}, results[3]);
}

TEST_P(AsyncMemcpyCollectiveOps, AsyncAllToAllDegenerateWithSplitDim) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2] broadcast(id), dimensions={}
    a0 = u32[2] constant({10, 20})
    a1 = u32[2] add(id2, a0)
    ROOT a2a = u32[2] all-to-all(u32[2] a1), dimensions={0}, replica_groups={{0},{1}}
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 20}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 21}, results[1]);
}

TEST_P(AsyncMemcpyCollectiveOps, AsyncAllToAllDegenerateWithoutSplitDim) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2] broadcast(id), dimensions={}
    a0 = u32[2] constant({10, 20})
    a1 = u32[2] add(id2, a0)
    a2a = (u32[2]) all-to-all(u32[2] a1), replica_groups={{0},{1}}
    ROOT gte0 = get-tuple-element(a2a), index=0
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 20}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 21}, results[1]);
}

TEST_P(MemcpyCollectiveOps, AllToAll8Gpus) {
  // Module computes the a2a of (10*replica-id + iota).
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    ten = u32[] constant(10)
    id_times_ten = u32[] multiply(id, ten)
    broadcast = u32[16] broadcast(id_times_ten), dimensions={}
    iota = u32[16] iota(), iota_dimension=0
    added = u32[16] add(broadcast, iota)
    ROOT all2all = u32[16] all-to-all(added), dimensions={0}
  }
  )";
  const int64_t kNumReplicas = 8;
  const int64_t kNumPartitions = 1;
  if (hlo_runner_->device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << hlo_runner_->device_count()
                 << " available)";
  }

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));
  const std::vector<Literal>& results = execution_result.results;

  Array<uint32_t> expected({16});
  expected.SetValues(
      {0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71});
  for (int device_id = 0; device_id < kNumReplicas; ++device_id) {
    LiteralTestUtil::ExpectR1Equal<uint32_t>(
        absl::MakeSpan(expected.data(), expected.num_elements()),
        results[device_id]);
    expected.Each(
        [&](absl::Span<const int64_t> indices, uint32_t* val) { *val += 2; });
  }
}

TEST_P(AsyncCollectiveOps, MatmulReplicated) {
  // collective_permute = f32[16,32]{1,0} collective-permute(x_unscaled),
  // source_target_pairs={{0,1}, {1,2}, {2,3}, {3,0}}
  absl::string_view kModuleReplicatedStr = R"(
    HloModule test

    ENTRY test {
      x_f32 = f32[16,32] parameter(0)
      y_f32 = f32[16,32] parameter(1)
      replica_id = u32[] replica-id()
      addend = f32[] convert(replica_id)
      addend_bcast = f32[16,32] broadcast(addend), dimensions={}
      x_add = f32[16,32] add(addend_bcast, x_f32)
      ROOT dot_a = f32[16,16] dot(x_add, y_f32), lhs_contracting_dims={1}, rhs_contracting_dims={1}
   }
  )";

  absl::string_view kModuleSingleStr = R"(
    HloModule test

    ENTRY test {
      x_f32 = f32[16,32] parameter(0)
      y_f32 = f32[16,32] parameter(1)
      replica_id = u32[] parameter(2)
      addend = f32[] convert(replica_id)
      addend_bcast = f32[16,32] broadcast(addend), dimensions={}
      x_add = f32[16,32] add(addend_bcast, x_f32)
      ROOT dot_a = f32[16,16] dot(x_add, y_f32), lhs_contracting_dims={1}, rhs_contracting_dims={1}
   }
  )";
  const int64_t kNumReplicas = 4;
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  bool enable_cublaslt = GetParam();
  VLOG(0) << "Running with CUBLAS enabled: " << enable_cublaslt;
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options().set_xla_gpu_enable_cublaslt(enable_cublaslt);

  ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));

  auto fake_arguments = xla::MakeFakeArguments(module.get()).value();
  std::vector<Literal*> fake_ptrs(fake_arguments.size());
  for (int i = 0; i < fake_arguments.size(); i++) {
    fake_ptrs[i] = &fake_arguments[i];
  }
  ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                       ExecuteReplicated(std::move(module), fake_ptrs));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);

  ASSERT_OK_AND_ASSIGN(auto ref_module,
                       ParseAndReturnVerifiedModule(kModuleSingleStr, config));
  ASSERT_OK_AND_ASSIGN(auto ref_exec, hlo_runner_->CreateExecutable(
                                          std::move(ref_module), true));

  ErrorSpec error_spec{5e-3, 5e-3};
  fake_ptrs.push_back(nullptr);
  for (int i = 0; i < kNumReplicas; i++) {
    auto replica_id =
        LiteralUtil::CreateFullWithDescendingLayout<uint32_t>({}, i);
    fake_ptrs.back() = &replica_id;
    ASSERT_OK_AND_ASSIGN(auto res, hlo_runner_->ExecuteWithExecutable(
                                       ref_exec.get(), fake_ptrs));
    EXPECT_TRUE(LiteralTestUtil::Near(res, results[i], error_spec));
  }
}

INSTANTIATE_TEST_SUITE_P(AsyncCollectiveOps, AsyncCollectiveOps,
                         ::testing::Bool(), GetAsyncTestSuiteName);

INSTANTIATE_TEST_SUITE_P(MemcpyCollectiveOps, MemcpyCollectiveOps,
                         ::testing::Bool(), GetMemcpyTestSuiteName);

INSTANTIATE_TEST_SUITE_P(AsyncMemcpyCollectiveOps, AsyncMemcpyCollectiveOps,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()),
                         GetAsyncMemcpyTestSuiteName);

// Tests for HLO level transforms.
TEST_F(CollectiveOpsTestE2E, WhileLoopReduceScatterCodeMotion) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  %add {
    %x = u32[] parameter(0)
    %y = u32[] parameter(1)
    ROOT %add = u32[] add(%x, %y)
  }

  %cond {
    %param = (u32[], u32[2], u32[1]) parameter(0)
    %count = get-tuple-element(%param), index=0
    %limit = u32[] constant(3)
    ROOT %result = pred[] compare(%count, %limit), direction=LT
  }

  %body {
    %param = (u32[], u32[2], u32[1]) parameter(0)

    %count = u32[] get-tuple-element(%param), index=0
    %increment = u32[] constant(1)
    %new_count = u32[] add(%count, %increment)

    // iter0: replica0 = {10, 15}, replica1 = {11, 16}
    // iter1: replica0 = {11, 17}, replica1 = {12, 18}
    // iter2: replica0 = {12, 19}, replica1 = {13, 20}

    %rs_input = u32[2] get-tuple-element(%param), index=1

    // iter0: replica0 = 21, replica1 = 31
    // iter1: replica0 = 23, replica1 = 35
    // iter2: replicq0 = 25, replica1 = 39
    %rs = u32[1] reduce-scatter(%rs_input), replica_groups={{0,1}}, to_apply=%add, dimensions={0}

    // iter0: replica0 = 5, replica1 = 5
    // iter1: replica0 = 26, replica1 = 36
    // iter2: replica0 = 49, replica1 = 70
    %old_accum = u32[1] get-tuple-element(%param), index=2

    // iter0: replica0 = 26, replica1 = 36
    // iter1: replica0 = 49, replica1 = 71
    // iter2: replica0 = 74, replica1 = 110
    %new_accum = u32[1] add(%rs, %old_accum)

    %input_inc = u32[2] constant({1, 2})

    // iter0: replica0 = {11, 17}, replica1 = {12, 18}
    // iter1: replica0 = {12, 19}, replica1 = {13, 20}
    // iter2: replica0 = {13, 21}, replica1 = {14, 22}
    %new_rs_input = u32[2] add(%rs_input, %input_inc)

    ROOT ret = (u32[], u32[2], u32[1]) tuple(%new_count, %new_rs_input, %new_accum)
  }

  ENTRY test_computation {
    // loop that executes 3 times.
    %count = u32[] constant(0)
    %id = u32[] replica-id()
    %id2 = u32[2] broadcast(id), dimensions={}
    %a0 = u32[2] constant({10, 15})
    // replica0: {10, 15}, replica1 : {11, 16}
    %init_rs_input = u32[2] add(id2, a0)
    %init_rs_accum = u32[1] constant({5})
    %while_init = (u32[], u32[2], u32[1]) tuple(%count, %init_rs_input, %init_rs_accum)
    %while_result = (u32[], u32[2], u32[1]) while(%while_init), body=%body, condition=%cond
    ROOT gte = u32[1] get-tuple-element(%while_result), index=2
  }
  )";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options()
      .set_xla_gpu_enable_while_loop_reduce_scatter_code_motion(true);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* executable_module = execution_result.optimized_module;

  // Verify that the reduce-scatter get hoisted out of the while loop.
  const HloInstruction* while_loop =
      FindInstruction(executable_module, HloOpcode::kWhile);
  ASSERT_THAT(while_loop, NotNull());
  const HloInstruction* reduce_scatter =
      FindInstruction(executable_module, HloOpcode::kAsyncStart);
  ASSERT_THAT(reduce_scatter, NotNull());

  const HloAsyncInstruction* rs_async =
      Cast<HloAsyncInstruction>(reduce_scatter);
  EXPECT_EQ(rs_async->async_wrapped_opcode(), HloOpcode::kReduceScatter);

  // Verify that the reduce-scatter has been hoisted out of the while loop and
  // into the entry computation.
  const HloComputation* entry = executable_module->entry_computation();
  EXPECT_EQ(reduce_scatter->parent(), entry);

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({74}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({110}, results[1]);
}

// Verify that all-to-all with split dims is not decomposed to tuples.
TEST_F(CollectiveOpsTestE2E, NoAllToAllDecomposition) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2, 2] broadcast(id), dimensions={}
    a0 = u32[2, 2] constant({{10, 15}, {20, 25}})
    a1 = u32[2, 2] add(id2, a0)
    all2all = u32[2, 2] all-to-all(a1), replica_groups={{0,1}}, dimensions={0}
    ROOT out = u32[4] reshape(all2all)
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));
  const HloModule* executable_module = execution_result.optimized_module;

  // Verify that the all-to-all is not decomposed into a tuple all-to-all.
  const HloInstruction* all_to_all =
      FindInstruction(executable_module, HloOpcode::kAllToAll);
  EXPECT_THAT(all_to_all, op::Shape("u32[2, 2]"));

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({20, 25, 21, 26}, results[1]);
}

// Verify that collectives won't be transformed into async ones.
TEST_F(CollectiveOpsTestE2E, NoAsyncCollectives) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  apply_op {
    x = u32[] parameter(0)
    y = u32[] parameter(1)
    ROOT apply_op = u32[] add(x, y)
  }

  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2, 2] broadcast(id), dimensions={}
    a0 = u32[2, 2] constant({{10, 15}, {20, 25}})
    a1 = u32[2, 2] add(id2, a0)
    all2all = u32[2, 2] all-to-all(a1), replica_groups={{0,1}}, dimensions={0}
    ROOT ag = u32[2, 2] all-reduce(all2all), replica_groups={{0,1}}, to_apply=apply_op
  }
  )";
  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options().add_xla_disable_hlo_passes(
      "gpu-convert-async-collectives-to-sync");
  config.mutable_debug_options().add_xla_gpu_disable_async_collectives(
      DebugOptions::ALLCOLLECTIVES);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable, hlo_runner_->CreateExecutable(std::move(module),
                                                     /*run_hlo_passes=*/true));
  TF_ASSERT_OK_AND_ASSIGN(const HloModule* const executable_module,
                          hlo_runner_->HloModuleFromWrapped(executable.get()));

  // Verify that the all-to-all is a sync collective.
  const HloInstruction* all_to_all =
      FindInstruction(executable_module, HloOpcode::kAsyncStart);
  EXPECT_FALSE(IsAsync(all_to_all));

  // Verify that the all-reduce is a sync collective.
  const HloInstruction* all_reduce =
      FindInstruction(executable_module, HloOpcode::kAllReduceStart);

  EXPECT_FALSE(IsAsync(all_reduce));
}

TEST_F(CollectiveOpsTestE2E, HostMemoryOffloadingWithDonation) {
  const absl::string_view kModuleStr = R"(
  HloModule test, entry_computation_layout={(f32[128,128]{1,0})->f32[128,128]{1,0:S(5)}}

  ENTRY test_computation {
    p0 = f32[128,128] parameter(0)
    ROOT copy.4 = f32[128,128]{1,0:S(5)} copy(p0)
  }
  )";

  const int64_t kNumReplicas = 1;

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options().set_xla_gpu_enable_host_memory_offloading(
      true);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));

  TF_ASSERT_OK(module->input_output_alias_config().SetUpAlias(
      /*output_index=*/{},
      /*param_number=*/0,
      /*param_index=*/{},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kMustAlias));

  auto executable_or = hlo_runner_->CreateExecutable(std::move(module),
                                                     /*run_hlo_passes=*/false);

  EXPECT_FALSE(executable_or.ok())
      << "Expected buffer assignment error but compilation succeeded";

  std::string error_message = executable_or.status().ToString();
  EXPECT_TRUE(absl::StrContains(
      error_message, "Shape and memory space of the result at index {} "))
      << "(f32[128,128]) must be the same as the shape and memory space"
      << "of aliased parameter 0 at index {} (f32[128,128])" << error_message;
}

// E2E tests comparing the results of windowed einsum and non-windowed cases.
class CollectiveOpsTestE2EWindowedNonWindowed : public CollectiveOpsTestE2E {
 public:
  CollectiveOpsTestE2EWindowedNonWindowed()
      : CollectiveOpsTestE2E(/*memory_size=*/4 * kGB,
                             /*collectives_memory_size=*/0) {}

  void CollectiveOpsCompareWindowedNonWindowed(
      absl::string_view hlo_text, bool disable_dot_merger = false,
      bool enable_a2a_rewrite = false) {
    const int64_t kNumReplicas = 1;
    const int64_t kNumPartitions = 4;
    if (hlo_runner_->device_count() < kNumReplicas * kNumPartitions) {
      GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                   << " devices (" << hlo_runner_->device_count()
                   << " available)";
    }

    HloModuleConfig config = GetModuleConfigForTest(
        /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);

    DebugOptions debug_options = config.mutable_debug_options();
    debug_options.set_xla_gpu_graph_min_graph_size(200);
    debug_options.set_xla_gpu_enable_triton_gemm(false);
    if (disable_dot_merger) {
      debug_options.add_xla_disable_hlo_passes("dot-merger");
    }

    // Run with reference config.
    TF_ASSERT_OK_AND_ASSIGN(auto ref_module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    ASSERT_OK_AND_ASSIGN(auto ref_executable, hlo_runner_->CreateExecutable(
                                                  std::move(ref_module),
                                                  /*run_hlo_passes=*/true));
    ASSERT_OK_AND_ASSIGN(
        const HloModule* ref_optimized_module,
        hlo_runner_->HloModuleFromWrapped(ref_executable.get()));

    auto fake_ref_arguments =
        xla::MakeFakeArguments(ref_optimized_module).value();
    std::vector<Literal*> ref_fake_ptrs(fake_ref_arguments.size());
    for (int i = 0; i < fake_ref_arguments.size(); i++) {
      ref_fake_ptrs[i] = &fake_ref_arguments[i];
    }
    std::vector<std::vector<Literal*>> ref_fake_ptrs_replicated(
        kNumReplicas * kNumPartitions, ref_fake_ptrs);

    ASSERT_OK_AND_ASSIGN(
        std::vector<Literal> ref_results,
        ExecuteReplicated(ref_executable.get(), ref_fake_ptrs_replicated));

    debug_options.set_xla_gpu_threshold_for_windowed_einsum_mib(0);
    debug_options.set_xla_gpu_multi_streamed_windowed_einsum(true);
    debug_options.set_xla_gpu_experimental_enable_alltoall_windowed_einsum(
        enable_a2a_rewrite);
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_text, config));

    TF_ASSERT_OK_AND_ASSIGN(
        ExecutionResult execution_result,
        ExecuteReplicated(std::move(module), ref_fake_ptrs));
    const std::vector<Literal>& results = execution_result.results;
    ASSERT_EQ(results.size(), kNumPartitions);

    ASSERT_EQ(ref_results.size(), kNumPartitions);
    ErrorSpec error_spec{1e-2, 1e-2};
    // Results should be the same between windowed einsum and non-windowed cases
    for (int i = 0; i < kNumPartitions; i++) {
      EXPECT_TRUE(
          LiteralTestUtil::Near(ref_results[i], results[i], error_spec));
    }
  }
};

TEST_F(CollectiveOpsTestE2EWindowedNonWindowed,
       WindowedEinsumE2EAllgatherMultiConsumer) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,16,48]{2,1,0}, bf16[48,192]{1,0}, bf16[48,192]{1,0}, bf16[192,48]{1,0})->bf16[2,16,48]{2,1,0}}, allow_spmd_sharding_propagation_to_parameters={false,false,false,false}, num_partitions=4

ENTRY main.12 {
  Arg_0.1 = bf16[2,16,48]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
  Arg_1.2 = bf16[48,192]{1,0} parameter(1), sharding={devices=[1,4]<=[4]}
  dot.5 = bf16[2,16,192]{2,1,0} dot(Arg_0.1, Arg_1.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  custom-call.7 = bf16[2,16,192]{2,1,0} custom-call(dot.5), custom_call_target="Sharding", sharding={devices=[1,1,4]<=[4]}
  Arg_2.3 = bf16[48,192]{1,0} parameter(2), sharding={devices=[1,4]<=[4]}
  dot.6 = bf16[2,16,192]{2,1,0} dot(Arg_0.1, Arg_2.3), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  add.8 = bf16[2,16,192]{2,1,0} add(custom-call.7, dot.6)
  Arg_3.4 = bf16[192,48]{1,0} parameter(3), sharding={devices=[4,1]<=[4]}
  dot.9 = bf16[2,16,48]{2,1,0} dot(add.8, Arg_3.4), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  tuple.10 = (bf16[2,16,48]{2,1,0}) tuple(dot.9)
  ROOT get-tuple-element.11 = bf16[2,16,48]{2,1,0} get-tuple-element(tuple.10), index=0, sharding={devices=[1,4,1]<=[4]}
} // main.12
)";

  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr);
}

TEST_F(CollectiveOpsTestE2EWindowedNonWindowed, WindowedEinsumE2EAllGatherF8) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(f8e4m3fn[2,16,48]{2,1,0}, f8e4m3fn[48,192]{1,0}, bf16[], bf16[])->bf16[2,16,192]{2,1,0}}, allow_spmd_sharding_propagation_to_parameters={false,false,false,false}, num_partitions=4

ENTRY main {
  lhs = f8e4m3fn[2,16,48]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
  rhs = f8e4m3fn[48,192]{1,0} parameter(1), sharding={devices=[1,4]<=[4]}
  scale_lhs = bf16[] parameter(2)
  scale_rhs = bf16[] parameter(3)
  scale_lhs_bcast = bf16[2,16,48]{2,1,0} broadcast(scale_lhs), dimensions={}
  scale_rhs_bcast = bf16[48,192]{1,0} broadcast(scale_rhs), dimensions={}
  lhs_bf16 = bf16[2,16,48]{2,1,0} convert(lhs)
  rhs_bf16 = bf16[48,192]{1,0} convert(rhs)
  lhs_scaled = bf16[2,16,48]{2,1,0} multiply(scale_lhs_bcast, lhs_bf16)
  rhs_scaled = bf16[48,192]{1,0} multiply(scale_rhs_bcast, rhs_bf16)
  dot = bf16[2,16,192]{2,1,0} dot(lhs_scaled, rhs_scaled), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT custom-call = bf16[2,16,192]{2,1,0} custom-call(dot), custom_call_target="Sharding", sharding={devices=[1,1,4]<=[4]}
} // main
)";

  // Disable the dot merger pass which can prevent the creation of FP8 GEMM
  // Custom Calls.
  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr,
                                          /*disable_dot_merger=*/true);

  // Verify the creation of FP8 GEMM Custom Calls on Hopper and newer
  // architectures.
  DebugOptions opts = GetDebugOptionsForTest();
  opts.set_xla_gpu_threshold_for_windowed_einsum_mib(0);
  opts.set_xla_gpu_multi_streamed_windowed_einsum(true);
  opts.set_xla_gpu_graph_min_graph_size(200);
  opts.set_xla_gpu_enable_triton_gemm(false);
  opts.add_xla_disable_hlo_passes("dot-merger");
  CollectiveOpsVerifyF8Matmul(kModuleReplicatedStr, opts);
}

TEST_F(CollectiveOpsTestE2EWindowedNonWindowed,
       WindowedEinsumE2EAllGatherReshapeF8) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule windowed_einsum_e2e_all_gather_multi_consumer_f8, entry_computation_layout={(f8e4m3fn[2,16,48]{2,1,0}, f8e4m3fn[2,24,192]{2,1,0}, bf16[], bf16[])->bf16[2,16,192]{2,1,0}}, allow_spmd_sharding_propagation_to_parameters={false,false,false,false}, num_partitions=4

ENTRY main {
  lhs = f8e4m3fn[2,16,48]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
  rhs = f8e4m3fn[2,24,192]{2,1,0} parameter(1), sharding={devices=[1,1,4]<=[4]}
  scale_lhs = bf16[] parameter(2)
  scale_rhs = bf16[] parameter(3)
  scale_lhs_bcast = bf16[2,16,48]{2,1,0} broadcast(scale_rhs), dimensions={}
  scale_rhs_bcast = bf16[2,24,192]{2,1,0} broadcast(scale_lhs), dimensions={}
  lhs_bf16 = bf16[2,16,48]{2,1,0} convert(lhs)
  rhs_bf16 = bf16[2,24,192]{2,1,0} convert(rhs)
  lhs_scaled = bf16[2,16,48]{2,1,0} multiply(scale_lhs_bcast, lhs_bf16)
  rhs_scaled = bf16[2,24,192]{2,1,0} multiply(scale_rhs_bcast, rhs_bf16)
  rhs_reshaped = bf16[48,192]{1,0} reshape(rhs_scaled)
  dot = bf16[2,16,192]{2,1,0} dot(lhs_scaled, rhs_reshaped), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT custom-call = bf16[2,16,192]{2,1,0} custom-call(dot), custom_call_target="Sharding", sharding={devices=[1,1,4]<=[4]}
} // main
)";

  // Disable the dot merger pass which can prevent the creation of FP8 GEMM
  // Custom Calls.
  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr,
                                          /*disable_dot_merger=*/true);

  // Verify the creation of FP8 GEMM Custom Calls on Hopper and newer
  // architectures.
  DebugOptions opts = GetDebugOptionsForTest();
  opts.set_xla_gpu_threshold_for_windowed_einsum_mib(0);
  opts.set_xla_gpu_multi_streamed_windowed_einsum(true);
  opts.set_xla_gpu_graph_min_graph_size(200);
  opts.set_xla_gpu_enable_triton_gemm(false);
  opts.add_xla_disable_hlo_passes("dot-merger");
  CollectiveOpsVerifyF8Matmul(kModuleReplicatedStr, opts);
}

TEST_F(CollectiveOpsTestE2EWindowedNonWindowed,
       WindowedEinsumE2EAllGatherMultiConsumerF8) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule windowed_einsum_e2e_all_gather_multi_consumer_f8, entry_computation_layout={(f8e4m3fn[2,16,48]{2,1,0}, f8e4m3fn[48,192]{1,0}, f8e4m3fn[48,192]{1,0}, bf16[], bf16[], bf16[])->bf16[2,16,192]{2,1,0}}, allow_spmd_sharding_propagation_to_parameters={false,false,false,false}, num_partitions=4

ENTRY main {
  lhs = f8e4m3fn[2,16,48]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
  rhs0 = f8e4m3fn[48,192]{1,0} parameter(1), sharding={devices=[1,4]<=[4]}
  scale_lhs = bf16[] parameter(3)
  scale_rhs0 = bf16[] parameter(4)
  scale_lhs_bcast = bf16[2,16,48]{2,1,0} broadcast(scale_lhs), dimensions={}
  scale_rhs0_bcast = bf16[48,192]{1,0} broadcast(scale_rhs0), dimensions={}
  lhs_bf16 = bf16[2,16,48]{2,1,0} convert(lhs)
  rhs0_bf16 = bf16[48,192]{1,0} convert(rhs0)
  lhs_scaled = bf16[2,16,48]{2,1,0} multiply(scale_lhs_bcast, lhs_bf16)
  rhs0_scaled = bf16[48,192]{1,0} multiply(scale_rhs0_bcast, rhs0_bf16)
  dot0 = bf16[2,16,192]{2,1,0} dot(lhs_scaled, rhs0_scaled), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  rhs1 = f8e4m3fn[48,192]{1,0} parameter(2), sharding={devices=[1,4]<=[4]}
  scale_rhs1 = bf16[] parameter(5)
  scale_rhs1_bcast = bf16[48,192]{1,0} broadcast(scale_rhs1), dimensions={}
  rhs1_bf16 = bf16[48,192]{1,0} convert(rhs1)
  rhs1_scaled = bf16[48,192]{1,0} multiply(scale_rhs1_bcast, rhs1_bf16)
  dot1 = bf16[2,16,192]{2,1,0} dot(lhs_scaled, rhs1_scaled), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT add = bf16[2,16,192]{2,1,0} add(dot0, dot1)
} // main
)";

  // Disable the dot merger pass which can prevent the creation of FP8 GEMM
  // Custom Calls.
  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr,
                                          /*disable_dot_merger=*/true);

  // Verify the creation of FP8 GEMM Custom Calls on Hopper and newer
  // architectures.
  DebugOptions opts = GetDebugOptionsForTest();
  opts.set_xla_gpu_threshold_for_windowed_einsum_mib(0);
  opts.set_xla_gpu_multi_streamed_windowed_einsum(true);
  opts.set_xla_gpu_graph_min_graph_size(200);
  opts.set_xla_gpu_enable_triton_gemm(false);
  opts.add_xla_disable_hlo_passes("dot-merger");
  CollectiveOpsVerifyF8Matmul(kModuleReplicatedStr, opts);
}

TEST_F(CollectiveOpsTestE2EWindowedNonWindowed,
       WindowedEinsumE2EReduceScatterF8) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(f8e4m3fn[2,16,192]{2,1,0}, f8e4m3fn[192,48]{1,0}, bf16[], bf16[])->bf16[2,16,48]{2,1,0}}, allow_spmd_sharding_propagation_to_parameters={false,false,false,false}, num_partitions=4

ENTRY main {
  lhs = f8e4m3fn[2,16,192]{2,1,0} parameter(0), sharding={devices=[1,1,4]<=[4]}
  rhs = f8e4m3fn[192,48]{1,0} parameter(1), sharding={devices=[4,1]<=[4]}
  scale_lhs = bf16[] parameter(2)
  scale_rhs = bf16[] parameter(3)
  scale_lhs_bcast = bf16[2,16,192]{2,1,0} broadcast(scale_lhs), dimensions={}
  scale_rhs_bcast = bf16[192,48]{1,0} broadcast(scale_rhs), dimensions={}
  lhs_bf16 = bf16[2,16,192]{2,1,0} convert(lhs)
  rhs_bf16 = bf16[192,48]{1,0} convert(rhs)
  lhs_scaled = bf16[2,16,192]{2,1,0} multiply(scale_lhs_bcast, lhs_bf16)
  rhs_scaled = bf16[192,48]{1,0} multiply(scale_rhs_bcast, rhs_bf16)
  dot = bf16[2,16,48]{2,1,0} dot(lhs_scaled, rhs_scaled), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT custom-call = bf16[2,16,48]{2,1,0} custom-call(dot), custom_call_target="Sharding", sharding={devices=[1,4,1]<=[4]}
} // main
)";

  // Disable the dot merger pass which can prevent the creation of FP8 GEMM
  // Custom Calls.
  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr,
                                          /*disable_dot_merger=*/true);

  // Verify the creation of FP8 GEMM Custom Calls on Hopper and newer
  // architectures.
  DebugOptions opts = GetDebugOptionsForTest();
  opts.set_xla_gpu_threshold_for_windowed_einsum_mib(0);
  opts.set_xla_gpu_multi_streamed_windowed_einsum(true);
  opts.set_xla_gpu_graph_min_graph_size(200);
  opts.set_xla_gpu_enable_triton_gemm(false);
  opts.add_xla_disable_hlo_passes("dot-merger");
  CollectiveOpsVerifyF8Matmul(kModuleReplicatedStr, opts);
}

TEST_F(CollectiveOpsTestE2EWindowedNonWindowed,
       WindowedEinsumE2EAllToAllDecompose) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[1,128,64]{2,1,0}, bf16[1,4,64,128]{3,2,1,0})->bf16[1,4,64,64]{3,2,1,0}}, num_partitions=4

ENTRY main.9_spmd {
  param0 = bf16[1,128,64]{2,1,0} parameter(0)
  param1 = bf16[1,4,64,128]{3,2,1,0} parameter(1)
  all-to-all = bf16[1,4,64,128]{3,2,1,0} all-to-all(param1), channel_id=4, replica_groups={{0,1,2,3}}, dimensions={1}
  ROOT dot.12 = bf16[1,4,64,64]{3,2,1,0} dot(all-to-all, param0), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}
}
)";

  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr);
}

TEST_F(CollectiveOpsTestE2EWindowedNonWindowed,
       WindowedEinsumE2EAllToAllTransposeDecompose) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[1,64,128]{2,1,0}, bf16[1,1,64,4,1,32]{5,4,3,2,1,0})->bf16[1,4,32,128]{3,2,1,0}}, num_partitions=4
ENTRY main.9_spmd {
  param.9 = bf16[1,64,128]{2,1,0} parameter(0)
  param.10 = bf16[1,1,64,4,1,32]{5,4,3,2,1,0} parameter(1)
  all-to-all = bf16[1,1,64,4,1,32]{5,4,3,2,1,0} all-to-all(param.10), channel_id=4, replica_groups={{0,1,2,3}}, dimensions={3}
  transpose.15 = bf16[1,4,1,64,1,32]{5,4,1,3,2,0} transpose(all-to-all), dimensions={0,3,1,2,4,5}
  reshape.2170 = bf16[1,4,64,1,32]{4,3,2,1,0} reshape(transpose.15)
  reshape.2173 = bf16[4,64,1,32]{3,2,1,0} reshape(reshape.2170)
  transpose.16 = bf16[1,4,32,64]{2,0,3,1} transpose(reshape.2173), dimensions={2,0,3,1}
  copy.53 = bf16[1,4,32,64]{3,2,1,0} copy(transpose.16)
  ROOT dot.12 = bf16[1,4,32,128]{3,2,1,0} dot(copy.53, param.9), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}
}
)";

  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr,
                                          /*disable_dot_merger=*/false,
                                          /*enable_a2a_rewrite=*/true);
}

TEST_F(CollectiveOpsTestE2EWindowedNonWindowed,
       WindowedEinsumE2EGemmAllToAllDecompose) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[1,64,128]{2,1,0}, bf16[1,4,32,128]{3,2,1,0})->bf16[1,4,32,64]{3,2,1,0}}, num_partitions=4

ENTRY main.9_spmd {
  param.9 = bf16[1,64,128]{2,1,0} parameter(0)
  param.10 = bf16[1,4,32,128]{3,2,1,0} parameter(1)
  dot.12 = bf16[1,4,32,64]{3,2,1,0} dot(param.10, param.9), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={2}
  ROOT all-to-all = bf16[1,4,32,64]{3,2,1,0} all-to-all(dot.12), channel_id=4, replica_groups={{0,1,2,3}}, dimensions={1}
}
)";

  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr,
                                          /*disable_dot_merger=*/false,
                                          /*enable_a2a_rewrite=*/true);
}

TEST_F(CollectiveOpsTestE2EWindowedNonWindowed,
       WindowedEinsumE2EGemmAllToAllTransposeDecompose) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[1,4,32,128]{3,2,1,0}, bf16[1,128,64]{2,1,0})->bf16[1,4,1,1,32,64]{5,4,3,2,1,0}}, num_partitions=4

ENTRY main.9_spmd {
  param.9 = bf16[1,4,32,128]{3,2,1,0} parameter(0)
  param.10 = bf16[1,128,64]{2,1,0} parameter(1)
  dot.13 = bf16[1,4,32,64]{3,2,1,0} dot(param.9, param.10), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}
  copy.55 = bf16[1,4,32,64]{3,2,1,0} copy(dot.13)
  transpose.17 = bf16[4,1,32,64]{3,2,0,1} transpose(copy.55), dimensions={1,0,2,3}
  copy.56 = bf16[4,1,32,64]{3,2,1,0} copy(transpose.17)
  reshape.2216 = bf16[1,4,1,32,64]{4,3,2,1,0} reshape(copy.56)
  reshape.2219 = bf16[1,4,1,1,32,64]{5,4,3,2,1,0} reshape(reshape.2216)
  ROOT all-to-all.1 = bf16[1,4,1,1,32,64]{5,4,3,2,1,0} all-to-all(reshape.2219), channel_id=7, replica_groups={{0,1,2,3}}, dimensions={1}
}
)";

  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr,
                                          /*disable_dot_merger=*/false,
                                          /*enable_a2a_rewrite=*/true);
}

TEST_F(CollectiveOpsTestE2EWindowedNonWindowed, WindowedEinsumE2EPartial) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(f32[8,2048,3264]{2,1,0}, f32[3264,2176]{1,0})->f32[8,2048,2176]{2,1,0}}, num_partitions=4

ENTRY entry {
  p0 = f32[8,2048,3264]{2,1,0} parameter(0), sharding={devices=[2,2,1]<=[4]}
  p1 = f32[3264,2176]{1,0} parameter(1), sharding={devices=[1,2,2]<=[2,2]T(1,0) last_tile_dim_replicate}
  dot = f32[8,2048,2176]{2,1,0} dot(f32[8,2048,3264]{2,1,0} p0, f32[3264,2176]{1,0} p1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT custom-call = f32[8,2048,2176]{2,1,0} custom-call(dot), custom_call_target="Sharding", sharding={devices=[2,1,2]<=[4]}
})";

  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr,
                                          /*disable_dot_merger=*/false,
                                          /*enable_a2a_rewrite=*/true);
}

TEST_F(CollectiveOpsTestE2E, CollectivePipelinerF8) {
  // Verify that FP8 patterns are preserved when collectives are pipelined so
  // the GEMM rewriter can create FP8 matmuls.
  if (!HasFp8Support()) {
    GTEST_SKIP() << "Test requires Hopper or newer architecture.";
  }

  absl::string_view kModuleReplicatedStr = R"(
HloModule module, entry_computation_layout={(bf16[128,128], bf16[32,128], bf16[], bf16[])->bf16[512,128]}, allow_spmd_sharding_propagation_to_parameters={false,false,false,false}, num_partitions=4
while_cond {
  input = (s32[], bf16[128,128], bf16[32,128], bf16[], bf16[], bf16[512,128]) parameter(0)
  loop_counter = s32[] get-tuple-element(input), index=0
  c4 = s32[] constant(4)
  ROOT compare = pred[] compare(loop_counter, c4), direction=LT
}
while_body {
  input = (s32[], bf16[128,128], bf16[32,128], bf16[], bf16[], bf16[512,128]) parameter(0)
  loop_counter = s32[] get-tuple-element(input), index=0
  lhs = bf16[128,128] get-tuple-element(input), index=1
  rhs = bf16[32,128] get-tuple-element(input), index=2
  partial_dot_output = bf16[512,128] get-tuple-element(input), index=5
  lhs_f8 = f8e4m3fn[128,128] convert(lhs)
  rhs_f8 = f8e4m3fn[32,128] convert(rhs)
  lhs_bf16 = bf16[128,128] convert(lhs_f8)
  rhs_bf16 = bf16[32,128] convert(rhs_f8)
  scale_lhs = bf16[] get-tuple-element(input), index=3
  scale_rhs = bf16[] get-tuple-element(input), index=4
  scale_lhs_bcast = bf16[128,128] broadcast(scale_lhs), dimensions={}
  scale_rhs_bcast = bf16[32,128] broadcast(scale_rhs), dimensions={}
  lhs_scaled = bf16[128,128] multiply(lhs_bf16, scale_lhs_bcast)
  rhs_scaled = bf16[32,128] multiply(rhs_bf16, scale_rhs_bcast)
  rhs_scaled_all_gathered = bf16[128,128] all-gather(rhs_scaled), channel_id=1, use_global_device_ids=true, dimensions={0}, replica_groups={{0,1,2,3}}
  dot = bf16[128,128] dot(lhs_scaled, rhs_scaled_all_gathered), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  c0 = s32[] constant(0)
  size = s32[] constant(128)
  iteration_offset = s32[] multiply(loop_counter, size)
  updated_dot_output = bf16[512,128] dynamic-update-slice(partial_dot_output, dot, iteration_offset, c0)
  c1 = s32[] constant(1)
  loop_counter_plus_one = s32[] add(loop_counter, c1)
  ROOT tuple = (s32[], bf16[128,128], bf16[32,128], bf16[], bf16[], bf16[512,128]) tuple(loop_counter_plus_one, lhs, rhs, scale_lhs, scale_rhs, updated_dot_output)
}
ENTRY entry {
  c0 = s32[] constant(0)
  lhs = bf16[128,128] parameter(0)
  rhs = bf16[32,128] parameter(1)
  scale_lhs = bf16[] parameter(2)
  scale_rhs = bf16[] parameter(3)
  result_buffer = bf16[512,128] constant(0.)
  while_input = (s32[], bf16[128,128], bf16[32,128], bf16[], bf16[], bf16[512,128]) tuple(c0, lhs, rhs, scale_lhs, scale_rhs, result_buffer)
  while = (s32[], bf16[128,128], bf16[32,128], bf16[], bf16[], bf16[512,128]) while(while_input), condition=while_cond, body=while_body
  ROOT dot_output = bf16[512,128] get-tuple-element(while), index=5
}
)";

  auto opts = GetDebugOptionsForTest();
  opts.set_xla_gpu_enable_triton_gemm(false);
  CollectiveOpsVerifyF8Matmul(kModuleReplicatedStr, opts);
}

// E2E tests comparing the results with and without pipelining of collectives.
class CollectiveOpsTestE2EPipelinedNonPipelined : public CollectiveOpsTestE2E {
 public:
  void CollectiveOpsComparePipelinedNonPipelined(absl::string_view hlo_string) {
    const int64_t kNumReplicas = 1;
    const int64_t kNumPartitions = 2;
    ASSERT_GE(hlo_runner_->device_count(), kNumReplicas * kNumPartitions)
        << "Test requires at least " << kNumReplicas * kNumPartitions
        << " devices (" << hlo_runner_->device_count() << " available)";

    HloModuleConfig config =
        GetModuleConfigForTest(kNumReplicas, kNumPartitions);
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_string, config));
    auto fake_arguments = xla::MakeFakeArguments(module.get()).value();
    std::vector<Literal*> fake_ptrs(fake_arguments.size());
    for (int i = 0; i < fake_arguments.size(); ++i) {
      fake_ptrs[i] = &fake_arguments[i];
    }

    TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                            ExecuteReplicated(std::move(module), fake_ptrs));
    const std::vector<Literal>& results = execution_result.results;
    ASSERT_EQ(results.size(), kNumPartitions);

    HloModuleConfig ref_config =
        GetModuleConfigForTest(kNumReplicas, kNumPartitions);
    DebugOptions& ref_opts = ref_config.mutable_debug_options();
    ref_opts.set_xla_gpu_enable_pipelined_all_reduce(false);
    ref_opts.set_xla_gpu_enable_pipelined_all_gather(false);
    ref_opts.set_xla_gpu_enable_pipelined_reduce_scatter(false);

    TF_ASSERT_OK_AND_ASSIGN(
        auto ref_module, ParseAndReturnVerifiedModule(hlo_string, ref_config));
    auto fake_ref_arguments = xla::MakeFakeArguments(ref_module.get()).value();
    std::vector<Literal*> ref_fake_ptrs(fake_ref_arguments.size());
    for (int i = 0; i < fake_ref_arguments.size(); ++i) {
      ref_fake_ptrs[i] = &fake_ref_arguments[i];
    }

    TF_ASSERT_OK_AND_ASSIGN(
        ExecutionResult ref_execution_result,
        ExecuteReplicated(std::move(ref_module), ref_fake_ptrs));
    const std::vector<Literal>& ref_results = ref_execution_result.results;
    ASSERT_EQ(ref_results.size(), kNumPartitions);
    ErrorSpec error_spec{1e-5, 1e-5};
    // Expect same results with and without pipelining of collectives.
    for (int i = 0; i < kNumPartitions; ++i) {
      EXPECT_TRUE(
          LiteralTestUtil::Near(ref_results[i], results[i], error_spec));
    }
  }
};

TEST_F(CollectiveOpsTestE2EPipelinedNonPipelined, CollectivePipelinerForward) {
  constexpr absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(bf16[5,8,16])->bf16[5,8,16]}, allow_spmd_sharding_propagation_to_parameters={false,false}, num_partitions=2

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[5,8,16], bf16[5,8,16]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  c5 = s32[] constant(5)
  ROOT cmp = pred[] compare(loop_index, c5), direction=LT
}

while_body {
  param = (s32[], bf16[5,8,16], bf16[5,8,16]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  partial_output = bf16[5,8,16] get-tuple-element(param), index=1
  slice_input = bf16[5,8,16] get-tuple-element(param), index=2
  c0 = s32[] constant(0)
  c1 = s32[] constant(1)
  next_loop_index = s32[] add(loop_index, c1)
  dynamic_slice = bf16[1,8,16] dynamic-slice(slice_input, loop_index, c0, c0), dynamic_slice_sizes={1,8,16}
  all_reduce = bf16[1,8,16] all-reduce(dynamic_slice), replica_groups={}, to_apply=add, channel_id=1
  updated_partial_output = bf16[5,8,16] dynamic-update-slice(partial_output, all_reduce, loop_index, c0, c0)
  ROOT tuple = (s32[], bf16[5,8,16], bf16[5,8,16]) tuple(next_loop_index, updated_partial_output, slice_input)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[5,8,16] parameter(0)
  tuple = (s32[], bf16[5,8,16], bf16[5,8,16]) tuple(c0, p0, p0)
  while = (s32[], bf16[5,8,16], bf16[5,8,16]) while(tuple), condition=while_cond, body=while_body
  ROOT gte = bf16[5,8,16] get-tuple-element(while), index=1
}
)";

  CollectiveOpsComparePipelinedNonPipelined(hlo_string);
}

TEST_F(CollectiveOpsTestE2EPipelinedNonPipelined,
       CollectivePipelinerForwardElementwise) {
  constexpr absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(bf16[5,8,16], bf16[])->bf16[5,8,16]}, allow_spmd_sharding_propagation_to_parameters={false,false}, num_partitions=2

add {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

while_cond {
  param = (s32[], bf16[5,8,16], bf16[5,8,16], bf16[]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  c5 = s32[] constant(5)
  ROOT cmp = pred[] compare(loop_index, c5), direction=LT
}

while_body {
  param = (s32[], bf16[5,8,16], bf16[5,8,16], bf16[]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  partial_output = bf16[5,8,16] get-tuple-element(param), index=1
  slice_input = bf16[5,8,16] get-tuple-element(param), index=2
  scale = bf16[] get-tuple-element(param), index=3
  scale_bcast = bf16[1,8,16] broadcast(scale), dimensions={}
  c0 = s32[] constant(0)
  c1 = s32[] constant(1)
  next_loop_index = s32[] add(loop_index, c1)
  dynamic_slice = bf16[1,8,16] dynamic-slice(slice_input, loop_index, c0, c0), dynamic_slice_sizes={1,8,16}
  all_reduce = bf16[1,8,16] all-reduce(dynamic_slice), replica_groups={}, to_apply=add, channel_id=1
  all_reduce_scaled = bf16[1,8,16] multiply(all_reduce, scale_bcast)
  updated_partial_output = bf16[5,8,16] dynamic-update-slice(partial_output, all_reduce_scaled, loop_index, c0, c0)
  ROOT tuple = (s32[], bf16[5,8,16], bf16[5,8,16], bf16[]) tuple(next_loop_index, updated_partial_output, slice_input, scale)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[5,8,16] parameter(0)
  p1 = bf16[] parameter(1)
  tuple = (s32[], bf16[5,8,16], bf16[5,8,16], bf16[]) tuple(c0, p0, p0, p1)
  while = (s32[], bf16[5,8,16], bf16[5,8,16], bf16[]) while(tuple), condition=while_cond, body=while_body
  ROOT gte = bf16[5,8,16] get-tuple-element(while), index=1
}
)";

  CollectiveOpsComparePipelinedNonPipelined(hlo_string);
}

TEST_F(CollectiveOpsTestE2EPipelinedNonPipelined, CollectivePipelinerBackward) {
  constexpr absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(bf16[5,4,16], bf16[5,1,2,16])->bf16[5,4,16]}, allow_spmd_sharding_propagation_to_parameters={false,false}, num_partitions=2

while_cond {
  param = (s32[], bf16[5,4,16], bf16[5,1,2,16]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  c5 = s32[] constant(5)
  ROOT cmp = pred[] compare(loop_index, c5), direction=LT
}

while_body {
  param = (s32[], bf16[5,4,16], bf16[5,1,2,16]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  partial_output = bf16[5,4,16] get-tuple-element(param), index=1
  slice_input = bf16[5,1,2,16] get-tuple-element(param), index=2
  c0 = s32[] constant(0)
  c1 = s32[] constant(1)
  next_loop_index = s32[] add(loop_index, c1)
  dynamic_slice = bf16[1,1,2,16] dynamic-slice(slice_input, loop_index, c0, c0, c0), dynamic_slice_sizes={1,1,2,16}
  dynamic_slice_reshape = bf16[1,2,16] reshape(dynamic_slice)
  all_gather = bf16[1,4,16] all-gather(dynamic_slice_reshape), dimensions={1}, replica_groups={}
  updated_partial_output = bf16[5,4,16] dynamic-update-slice(partial_output, all_gather, loop_index, c0, c0)
  ROOT tuple = (s32[], bf16[5,4,16], bf16[5,1,2,16]) tuple(next_loop_index, updated_partial_output, slice_input)
}

ENTRY entry {
  c0 = s32[] constant(0)
  p0 = bf16[5,4,16] parameter(0)
  p1 = bf16[5,1,2,16] parameter(1)
  tuple = (s32[], bf16[5,4,16], bf16[5,1,2,16]) tuple(c0, p0, p1)
  while = (s32[], bf16[5,4,16], bf16[5,1,2,16]) while(tuple), condition=while_cond, body=while_body
  ROOT gte = bf16[5,4,16] get-tuple-element(while), index=1
}
)";

  CollectiveOpsComparePipelinedNonPipelined(hlo_string);
}

TEST_F(CollectiveOpsTestE2EPipelinedNonPipelined,
       CollectivePipelinerBackwardStartFromOne) {
  constexpr absl::string_view hlo_string = R"(
HloModule module, entry_computation_layout={(bf16[5,4,16], bf16[5,1,2,16])->bf16[5,4,16]}, allow_spmd_sharding_propagation_to_parameters={false,false}, num_partitions=2

while_cond {
  param = (s32[], bf16[5,4,16], bf16[5,1,2,16]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  c6 = s32[] constant(6)
  ROOT cmp = pred[] compare(loop_index, c6), direction=LT
}

while_body {
  param = (s32[], bf16[5,4,16], bf16[5,1,2,16]) parameter(0)
  loop_index = s32[] get-tuple-element(param), index=0
  partial_output = bf16[5,4,16] get-tuple-element(param), index=1
  slice_input = bf16[5,1,2,16] get-tuple-element(param), index=2
  c0 = s32[] constant(0)
  c1 = s32[] constant(1)
  next_loop_index = s32[] add(loop_index, c1)
  loop_index_minus_one = s32[] subtract(loop_index, c1)
  dynamic_slice = bf16[1,1,2,16] dynamic-slice(slice_input, loop_index_minus_one, c0, c0, c0), dynamic_slice_sizes={1,1,2,16}
  dynamic_slice_reshape = bf16[1,2,16] reshape(dynamic_slice)
  all_gather = bf16[1,4,16] all-gather(dynamic_slice_reshape), dimensions={1}, replica_groups={}
  updated_partial_output = bf16[5,4,16] dynamic-update-slice(partial_output, all_gather, loop_index_minus_one, c0, c0)
  ROOT tuple = (s32[], bf16[5,4,16], bf16[5,1,2,16]) tuple(next_loop_index, updated_partial_output, slice_input)
}

ENTRY entry {
  c1 = s32[] constant(1)
  p0 = bf16[5,4,16] parameter(0)
  p1 = bf16[5,1,2,16] parameter(1)
  tuple = (s32[], bf16[5,4,16], bf16[5,1,2,16]) tuple(c1, p0, p1)
  while = (s32[], bf16[5,4,16], bf16[5,1,2,16]) while(tuple), condition=while_cond, body=while_body
  ROOT gte = bf16[5,4,16] get-tuple-element(while), index=1
}
)";

  CollectiveOpsComparePipelinedNonPipelined(hlo_string);
}

TEST_F(CollectiveOpsTestE2E, AllToAllQuantizeCollectiveQuantizer) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={()->bf16[2]}, num_partitions=2
ENTRY entry {
  input = f32[2] constant({2., 4.})
  scale = f32[] constant(2.)
  scale_bcast = f32[2] broadcast(scale), dimensions={}
  input_scaled = f32[2] multiply(input, scale_bcast)
  all-to-all = f32[2] all-to-all(input_scaled), channel_id=1, replica_groups={{0,1}}, dimensions={0}
  ROOT convert = bf16[2] convert(all-to-all)
}
)";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas * kNumPartitions)
      << "Test requires at least " << kNumReplicas * kNumPartitions
      << " devices (" << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr,
                                                kNumReplicas, kNumPartitions));
  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* all_to_all =
      FindInstruction(hlo_module, HloOpcode::kAllToAll);
  EXPECT_THAT(all_to_all, NotNull());
  EXPECT_EQ(all_to_all->shape().element_type(), BF16);

  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumPartitions);
  const bfloat16 four = static_cast<bfloat16>(4.);
  const bfloat16 eight = static_cast<bfloat16>(8.);
  LiteralTestUtil::ExpectR1Equal<bfloat16>({four, four}, results[0]);
  LiteralTestUtil::ExpectR1Equal<bfloat16>({eight, eight}, results[1]);
}

TEST_F(CollectiveOpsTestE2E, DequantizeAllToAllCollectiveQuantizer) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={()->f32[2]}, num_partitions=2
ENTRY entry {
  input = bf16[2] constant({2., 4.})
  input_f32 = f32[2] convert(input)
  scale = f32[] constant(2.)
  scale_bcast = f32[2] broadcast(scale), dimensions={}
  input_scaled = f32[2] multiply(input_f32, scale_bcast)
  ROOT all-to-all = f32[2] all-to-all(input_scaled), channel_id=1, replica_groups={{0,1}}, dimensions={0}
}
)";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas * kNumPartitions)
      << "Test requires at least " << kNumReplicas * kNumPartitions
      << " devices (" << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr,
                                                kNumReplicas, kNumPartitions));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module)));

  // Verify that the element type of the all-to-all has been changed to BF16.
  const HloModule* hlo_module = execution_result.optimized_module;
  HloInstruction* all_to_all =
      FindInstruction(hlo_module, HloOpcode::kAllToAll);
  EXPECT_THAT(all_to_all, NotNull());
  EXPECT_EQ(all_to_all->shape().element_type(), BF16);

  // Execute the test on 2 partitions.
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumPartitions);
  LiteralTestUtil::ExpectR1Equal<float>({4., 4.}, results[0]);
  LiteralTestUtil::ExpectR1Equal<float>({8., 8.}, results[1]);
}

TEST_F(CollectiveOpsTestE2E, AllGatherQuantizeCollectiveQuantizer) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule module, entry_computation_layout={(f32[2], f32[1])->bf16[4]}, num_partitions=4
max {
    a = f32[] parameter(0)
    b = f32[] parameter(1)
    ROOT max = f32[] maximum(a, b)
  }

ENTRY entry {
  param = f32[2] parameter(0)
  all-gather = f32[4] all-gather(param), dimensions={0}, replica_groups={{0,1},{2,3}}, channel_id=1, use_global_device_ids=true
  scale = f32[1] parameter(1), sharding={devices=[4]<=[4]}
  scalar_scale = f32[] reshape(scale)
  all_reduced_scale = f32[] all-reduce(scalar_scale), to_apply=max, replica_groups={{0,1},{2,3}}, channel_id=2, use_global_device_ids=true
  scale_bcast = f32[4] broadcast(all_reduced_scale), dimensions={}
  divide = f32[4] divide(all-gather, scale_bcast)
  clamp_lower = f32[] constant(-448.0)
  clamp_lower_bcast = f32[4] broadcast(clamp_lower), dimensions={}
  clamp_upper = f32[] constant(448.0)
  clamp_upper_bcast = f32[4] broadcast(clamp_upper), dimensions={}
  clamp = f32[4] clamp(clamp_lower_bcast, divide, clamp_upper_bcast)
  ROOT convert = bf16[4] convert(clamp)
}
)";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 4;
  if (hlo_runner_->device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << hlo_runner_->device_count()
                 << " available)";
  }

  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable, hlo_runner_->CreateExecutable(std::move(module),
                                                     /*run_hlo_passes=*/true));
  TF_ASSERT_OK_AND_ASSIGN(const HloModule* const hlo_module,
                          hlo_runner_->HloModuleFromWrapped(executable.get()));
  HloInstruction* all_gather =
      FindInstruction(hlo_module, HloOpcode::kAllGatherStart);

  EXPECT_THAT(all_gather, NotNull());
  EXPECT_EQ(all_gather->shape().tuple_shapes(0).element_type(), BF16);
  EXPECT_EQ(all_gather->shape().tuple_shapes(1).element_type(), BF16);
}

TEST_F(CollectiveOpsTestE2E, NoErrorOnDuplicateChannelId) {
  absl::string_view kModuleReplicatedStr = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(f32[4,32,128]{2,1,0})->(f32[4,32,128]{2,1,0}, f32[4,32,128]{2,1,0})}, num_partitions=4
ENTRY entry {
  param = f32[4,32,128]{2,1,0} parameter(0)
  all-to-all = f32[4,32,128]{2,1,0} all-to-all(param), channel_id=1, replica_groups={{0,1,2,3}}, dimensions={1}
  all-to-all.1 = f32[4,32,128]{2,1,0} all-to-all(param), channel_id=1, replica_groups={{0,1,2,3}}, dimensions={0}
  ROOT tuple = (f32[4,32,128]{2,1,0}, f32[4,32,128]{2,1,0}) tuple(all-to-all, all-to-all.1)
}
)";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 4;
  if (hlo_runner_->device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << hlo_runner_->device_count()
                 << " available)";
  }

  if (hlo_runner_->device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << hlo_runner_->device_count()
                 << " available)";
  }

  HloModuleConfig config = GetModuleConfigForTest(
      /*replica_count=*/kNumReplicas, /*num_partitions=*/kNumPartitions);
  config.mutable_debug_options().set_xla_ignore_channel_id(true);

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable, hlo_runner_->CreateExecutable(std::move(module),
                                                     /*run_hlo_passes=*/true));

  TF_ASSERT_OK_AND_ASSIGN(const HloModule* const hlo_module,
                          hlo_runner_->HloModuleFromWrapped(executable.get()));
  EXPECT_NE(hlo_module, nullptr);
}

TEST_F(CollectiveOpsTestE2E, MemcpyP2pWhileLoopCorrectness) {
  absl::string_view hlo_string = R"(
HloModule MemcpyP2pWhileLoopCorrectness, entry_computation_layout={(bf16[128,96]{1,0})->(bf16[32,384]{1,0}, bf16[32,384]{1,0})}, allow_spmd_sharding_propagation_to_output={true,true}, num_partitions=4

None.4 {
  Arg_1.6 = bf16[32,96]{1,0} parameter(1)
  Arg_0.5 = bf16[32,96]{1,0} parameter(0)
  collective-permute.9 = bf16[32,96]{1,0} collective-permute(Arg_0.5), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
  constant.7 = bf16[] constant(2)
  broadcast.8 = bf16[32,96]{1,0} broadcast(constant.7), dimensions={}
  multiply.10 = bf16[32,96]{1,0} multiply(Arg_0.5, broadcast.8)
  ROOT tuple.11 = (bf16[32,96]{1,0}, bf16[32,96]{1,0}) tuple(collective-permute.9, multiply.10)
} // None.4

region_0.12 {
  arg_tuple.13 = (s32[], bf16[32,96]{1,0}, bf16[32,96]{1,0}) parameter(0)
  get-tuple-element.14 = s32[] get-tuple-element(arg_tuple.13), index=0
  constant.17 = s32[] constant(1)
  add.21 = s32[] add(get-tuple-element.14, constant.17)
  get-tuple-element.15 = bf16[32,96]{1,0} get-tuple-element(arg_tuple.13), index=1
  get-tuple-element.16 = bf16[32,96]{1,0} get-tuple-element(arg_tuple.13), index=2
  call.18 = (bf16[32,96]{1,0}, bf16[32,96]{1,0}) call(get-tuple-element.15, get-tuple-element.16), to_apply=None.4
  get-tuple-element.19 = bf16[32,96]{1,0} get-tuple-element(call.18), index=0
  get-tuple-element.20 = bf16[32,96]{1,0} get-tuple-element(call.18), index=1
  ROOT tuple.22 = (s32[], bf16[32,96]{1,0}, bf16[32,96]{1,0}) tuple(add.21, get-tuple-element.19, get-tuple-element.20)
} // region_0.12

region_1.23 {
  arg_tuple.24 = (s32[], bf16[32,96]{1,0}, bf16[32,96]{1,0}) parameter(0)
  get-tuple-element.26 = bf16[32,96]{1,0} get-tuple-element(arg_tuple.24), index=1
  get-tuple-element.27 = bf16[32,96]{1,0} get-tuple-element(arg_tuple.24), index=2
  get-tuple-element.25 = s32[] get-tuple-element(arg_tuple.24), index=0
  constant.28 = s32[] constant(3)
  ROOT compare.29 = pred[] compare(get-tuple-element.25, constant.28), direction=LT
} // region_1.23

shmap_body.30 {
  constant.32 = s32[] constant(0)
  Arg_0.31 = bf16[32,96]{1,0} parameter(0)
  constant.33 = bf16[] constant(0)
  broadcast.34 = bf16[32,96]{1,0} broadcast(constant.33), dimensions={}
  tuple.35 = (s32[], bf16[32,96]{1,0}, bf16[32,96]{1,0}) tuple(constant.32, Arg_0.31, broadcast.34)
  while.36 = (s32[], bf16[32,96]{1,0}, bf16[32,96]{1,0}) while(tuple.35), condition=region_1.23, body=region_0.12
  get-tuple-element.37 = s32[] get-tuple-element(while.36), index=0
  get-tuple-element.38 = bf16[32,96]{1,0} get-tuple-element(while.36), index=1
  get-tuple-element.39 = bf16[32,96]{1,0} get-tuple-element(while.36), index=2
  ROOT tuple.40 = (bf16[32,96]{1,0}, bf16[32,96]{1,0}) tuple(get-tuple-element.38, get-tuple-element.39)
} // shmap_body.30

ENTRY main.49 {
  Arg_0.1 = bf16[128,96]{1,0} parameter(0), sharding={devices=[4,1]<=[4]}
  custom-call.2 = bf16[128,96]{1,0} custom-call(Arg_0.1), custom_call_target="Sharding", sharding={devices=[4,1]<=[4]}
  custom-call.3 = bf16[32,96]{1,0} custom-call(custom-call.2), custom_call_target="SPMDFullToShardShape", sharding={manual}
  call.41 = (bf16[32,96]{1,0}, bf16[32,96]{1,0}) call(custom-call.3), to_apply=shmap_body.30
  get-tuple-element.42 = bf16[32,96]{1,0} get-tuple-element(call.41), index=0
  custom-call.44 = bf16[32,96]{1,0} custom-call(get-tuple-element.42), custom_call_target="Sharding", sharding={manual}
  custom-call.45 = bf16[32,384]{1,0} custom-call(custom-call.44), custom_call_target="SPMDShardToFullShape", sharding={devices=[1,4]<=[4]}
  get-tuple-element.43 = bf16[32,96]{1,0} get-tuple-element(call.41), index=1
  custom-call.46 = bf16[32,96]{1,0} custom-call(get-tuple-element.43), custom_call_target="Sharding", sharding={manual}
  custom-call.47 = bf16[32,384]{1,0} custom-call(custom-call.46), custom_call_target="SPMDShardToFullShape", sharding={devices=[1,4]<=[4]}
  ROOT tuple.48 = (bf16[32,384]{1,0}, bf16[32,384]{1,0}) tuple(custom-call.45, custom-call.47)
} // main.49
)";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 4;
  if (hlo_runner_->device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << hlo_runner_->device_count()
                 << " available)";
  }

  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas, kNumPartitions);
  config.mutable_debug_options().set_xla_gpu_use_memcpy_local_p2p(true);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_string, config));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> executable,
                       hlo_runner_->CreateExecutable(std::move(module),
                                                     /*run_hlo_passes=*/true));

  ASSERT_OK_AND_ASSIGN(const HloModule* optimized_module,
                       hlo_runner_->HloModuleFromWrapped(executable.get()));

  ASSERT_OK_AND_ASSIGN(auto fake_arguments,
                       xla::MakeFakeArguments(optimized_module));
  std::vector<Literal*> fake_ptrs(fake_arguments.size());
  for (int i = 0; i < fake_arguments.size(); ++i) {
    fake_ptrs[i] = &fake_arguments[i];
  }

  std::vector<std::vector<Literal*>> fake_ptrs_replicated(
      kNumReplicas * kNumPartitions, fake_ptrs);

  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(executable.get(), fake_ptrs_replicated));
  ASSERT_EQ(results.size(), kNumPartitions);

  HloModuleConfig ref_config =
      GetModuleConfigForTest(kNumReplicas, kNumPartitions);
  ref_config.mutable_debug_options().set_xla_gpu_use_memcpy_local_p2p(false);

  TF_ASSERT_OK_AND_ASSIGN(auto ref_module,
                          ParseAndReturnVerifiedModule(hlo_string, ref_config));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult ref_execution_result,
                          ExecuteReplicated(std::move(ref_module), fake_ptrs));
  const std::vector<Literal>& ref_results = ref_execution_result.results;
  ASSERT_EQ(ref_results.size(), kNumPartitions);
  ErrorSpec error_spec{1e-5, 1e-5};
  // Expect same results with and without pipelining of collectives.
  for (int i = 0; i < kNumPartitions; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Near(ref_results[i], results[i], error_spec));
  }
}

TEST_F(CollectiveOpsTestE2E, MemcpyP2pLargeMessage) {
  absl::string_view hlo_string = R"(
HloModule MemcpyP2pLargeMessage, entry_computation_layout={(bf16[256,16000]{1,0})->bf16[256,16000]{1,0}}, num_partitions=4

ENTRY main {
  Arg_0.5 = bf16[256,16000]{1,0} parameter(0)
  collective-permute.0 = bf16[256,16000]{1,0} collective-permute(Arg_0.5), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
  collective-permute.1 = bf16[256,16000]{1,0} collective-permute(collective-permute.0), channel_id=2, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  collective-permute.2 = bf16[256,16000]{1,0} collective-permute(collective-permute.1), channel_id=3, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  collective-permute.3 = bf16[256,16000]{1,0} collective-permute(collective-permute.2), channel_id=4, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  collective-permute.4 = bf16[256,16000]{1,0} collective-permute(collective-permute.3), channel_id=5, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  collective-permute.5 = bf16[256,16000]{1,0} collective-permute(collective-permute.4), channel_id=6, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  collective-permute.6 = bf16[256,16000]{1,0} collective-permute(collective-permute.5), channel_id=7, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  collective-permute.7 = bf16[256,16000]{1,0} collective-permute(collective-permute.6), channel_id=8, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}

  constant.0 = bf16[] constant(2)
  broadcast.0 = bf16[256,16000]{1,0} broadcast(constant.0), dimensions={}
  collective-permute.8 = bf16[256,16000]{1,0} collective-permute(broadcast.0), channel_id=6, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
  collective-permute.9 = bf16[256,16000]{1,0} collective-permute(collective-permute.8), channel_id=9, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
  collective-permute.10 = bf16[256,16000]{1,0} collective-permute(collective-permute.9), channel_id=10, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
  collective-permute.11 = bf16[256,16000]{1,0} collective-permute(collective-permute.10), channel_id=11, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}

  ROOT multiply.10 = bf16[256,16000]{1,0} multiply(collective-permute.7, collective-permute.11)
} // main
)";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 4;
  if (hlo_runner_->device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << hlo_runner_->device_count()
                 << " available)";
  }

  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas, kNumPartitions);
  config.mutable_debug_options().set_xla_gpu_use_memcpy_local_p2p(true);
  config.mutable_debug_options().add_xla_disable_hlo_passes(
      "gpu-convert-async-collectives-to-sync");
  config.set_use_spmd_partitioning(false);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  auto fake_arguments = xla::MakeFakeArguments(module.get()).value();
  std::vector<Literal*> fake_ptrs(fake_arguments.size());
  for (int i = 0; i < fake_arguments.size(); ++i) {
    fake_ptrs[i] = &fake_arguments[i];
  }

  HloModuleConfig ref_config =
      GetModuleConfigForTest(kNumReplicas, kNumPartitions);
  ref_config.mutable_debug_options().set_xla_gpu_use_memcpy_local_p2p(false);

  TF_ASSERT_OK_AND_ASSIGN(auto ref_module,
                          ParseAndReturnVerifiedModule(hlo_string, ref_config));
  auto fake_ref_arguments = xla::MakeFakeArguments(ref_module.get()).value();
  std::vector<Literal*> ref_fake_ptrs(fake_ref_arguments.size());
  for (int i = 0; i < fake_ref_arguments.size(); ++i) {
    ref_fake_ptrs[i] = &fake_ref_arguments[i];
  }

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(module), fake_ptrs));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumPartitions);

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult ref_execution_result,
      ExecuteReplicated(std::move(ref_module), ref_fake_ptrs));
  const std::vector<Literal>& ref_results = ref_execution_result.results;
  ASSERT_EQ(ref_results.size(), kNumPartitions);
  ErrorSpec error_spec{1e-5, 1e-5};
  // Expect same results with and without pipelining of collectives.
  for (int i = 0; i < kNumPartitions; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Near(ref_results[i], results[i], error_spec));
  }
}

TEST_F(CollectiveOpsTestE2E, AllgatherMemspaceWithNcclUserBuffer) {
  absl::string_view hlo_string = R"(
HloModule AllgatherMemspaceWithNcclUserBuffer, entry_computation_layout={(bf16[1024,1024]{1,0},bf16[1024,1024]{1,0})->bf16[4096,1024]{1,0}}, num_partitions=4

ENTRY main {
  Arg_1 = bf16[1024,1024]{1,0} parameter(0)
  Arg_2 = bf16[1024,1024]{1,0} parameter(1)

  add = bf16[1024,1024]{1,0} add(Arg_1, Arg_2)
  all-gather-start = (bf16[1024,1024]{1,0},bf16[4096,1024]{1,0}) all-gather-start(add), dimensions={0}
  all-gather-done = bf16[4096,1024]{1,0} all-gather-done(all-gather-start)

  ROOT add2 = bf16[4096,1024]{1,0} add(all-gather-done, all-gather-done)
} // main
)";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 4;
  if (hlo_runner_->device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << hlo_runner_->device_count()
                 << " available)";
  }

  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas, kNumPartitions);
  config.mutable_debug_options().set_xla_gpu_enable_nccl_user_buffers(true);
  config.mutable_debug_options().add_xla_disable_hlo_passes(
      "gpu-convert-async-collectives-to-sync");
  config.set_use_spmd_partitioning(false);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable, hlo_runner_->CreateExecutable(std::move(module),
                                                     /*run_hlo_passes=*/false));
  TF_ASSERT_OK_AND_ASSIGN(const HloModule* const executable_module,
                          hlo_runner_->HloModuleFromWrapped(executable.get()));
  HloInstruction* ag_start =
      FindInstructions(executable_module, HloOpcode::kAllGatherStart)[0];
  // Both ag and its producer should have collective memory space 1
  EXPECT_EQ(ag_start->shape().tuple_shapes()[1].layout().memory_space(), 1);
  EXPECT_EQ(ag_start->operand(0)->shape().layout().memory_space(), 1);
}

TEST_F(CollectiveOpsTestE2E,
       CollectiveConsumingConstantAndModuleShouldHaveCopies) {
  absl::string_view hlo_string = R"(
HloModule CollectiveCopies, entry_computation_layout={(bf16[1024,1024]{1,0})->(bf16[1024,1024]{1,0}, bf16[])}, num_partitions=4
apply_op {
x = bf16[] parameter(0)
y = bf16[] parameter(1)
ROOT apply_op = bf16[] add(x, y)
}
ENTRY main {
Arg_1 = bf16[1024,1024]{1,0} parameter(0)
constant0 = bf16[] constant(10)
all-reduce-start.const = bf16[] all-reduce-start(constant0), to_apply=apply_op, replica_groups={{0,1,2,3}}
all-reduce-done.const = bf16[] all-reduce-done(all-reduce-start.const)
all-reduce-start = bf16[1024,1024]{1,0} all-reduce-start(Arg_1), to_apply=apply_op, replica_groups={{0,1,2,3}}
all-reduce-done = bf16[1024,1024]{1,0} all-reduce-done(all-reduce-start)
ROOT tuple = (bf16[1024,1024]{1,0}, bf16[]) tuple(all-reduce-done, all-reduce-done.const)
} // main
)";

  const int64_t kNumReplicas = 1;
  const int64_t kNumPartitions = 4;
  if (hlo_runner_->device_count() < kNumReplicas * kNumPartitions) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas * kNumPartitions
                 << " devices (" << hlo_runner_->device_count()
                 << " available)";
  }

  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas, kNumPartitions);
  config.mutable_debug_options().set_xla_gpu_enable_nccl_user_buffers(true);
  config.mutable_debug_options().add_xla_disable_hlo_passes(
      "gpu-convert-async-collectives-to-sync");
  config.set_use_spmd_partitioning(false);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable, hlo_runner_->CreateExecutable(std::move(module),
                                                     /*run_hlo_passes=*/false));
  TF_ASSERT_OK_AND_ASSIGN(const HloModule* const executable_module,
                          hlo_runner_->HloModuleFromWrapped(executable.get()));
  std::vector<HloInstruction*> all_ar =
      FindInstructions(executable_module, HloOpcode::kAllReduceStart);
  // Both allreduces should have their operands copied to collective memory
  // space.
  for (auto ar : all_ar) {
    EXPECT_EQ(ar->operand(0)->opcode(), HloOpcode::kCopy);
    EXPECT_EQ(ar->operand(0)->shape().layout().memory_space(), 1);
  }
}

class AllReduceTest
    : public CollectiveOpsWithFlagsBase,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {
 public:
  struct InputsOutputs {
    std::vector<Literal> inputs;
    std::vector<Literal> expected_outputs;

    [[nodiscard]] std::vector<std::vector<Literal*>> InputLiteralPtrs() {
      std::vector<std::vector<Literal*>> result;
      for (auto& input : inputs) {
        result.push_back(std::vector<Literal*>{&input});
      }
      return result;
    }
  };

  AllReduceTest()
      : CollectiveOpsWithFlagsBase(/*enable_async=*/std::get<0>(GetParam()),
                                   /*enable_p2p_memcpy=*/false,
                                   /*memory_size=*/32 * kMB,
                                   /*collectives_memory_size=*/0) {}

  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    if (!IsAmpereAndHigher()) {
      GTEST_SKIP() << "Test requires Ampere or newer architecture since it's "
                      "using triton.";
    }
  }

 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();

    opts.set_xla_gpu_unsupported_use_all_reduce_one_shot_kernel(
        std::get<1>(GetParam()));

    return opts;
  }

  static absl::StatusOr<InputsOutputs> BuildTestInputsOutputs(
      const HloModule& module, int64_t num_replicas, int64_t num_iterations) {
    std::vector<Array<float>> inputs;
    std::vector<Literal> input_literals;
    const int64_t num_elements =
        module.entry_computation()->root_instruction()->shape().dimensions()[0];
    for (int i = 0; i < num_replicas; ++i) {
      auto& input = inputs.emplace_back(Array<float>({num_elements}));
      input.FillRandom(1.0f, 10.0f, /*seed=*/i);
      input_literals.push_back(LiteralUtil::CreateFromArray(input));
    }
    std::vector<Array<float>> expected_outputs(num_replicas,
                                               Array<float>({num_elements}));
    std::vector<Literal> expected_output_literals;
    const HloInstruction* const instr =
        FindInstruction(&module, HloOpcode::kAllReduce);
    if (instr == nullptr) {
      return absl::InvalidArgumentError(
          "Instruction 'all-reduce' not found in module.");
    }
    const std::vector<ReplicaGroup>& replica_groups =
        instr->device_list().replica_groups();
    // Map each device to set of replica groups it belongs to.
    std::vector<std::vector<int64_t>> device_to_groups(num_replicas);
    for (const auto& replica_group : replica_groups) {
      const auto& replica_ids = replica_group.replica_ids();
      for (int64_t replica : replica_group.replica_ids()) {
        CHECK_EQ(device_to_groups[replica].size(), 0);
        device_to_groups[replica].assign(replica_ids.begin(),
                                         replica_ids.end());
      }
    }
    // Sum inputs from each replica group
    for (int i = 0; i < num_replicas; ++i) {
      expected_outputs[i].Each(
          [&](absl::Span<const int64_t> indices, float* val) {
            for (const int64_t replica : device_to_groups[i]) {
              *val += inputs[replica](indices);
            }
            // Each iteration after the first,the output is doubled.
            *val *= std::pow(device_to_groups[i].size(), num_iterations - 1);
          });
    }
    for (auto& expected_output : expected_outputs) {
      expected_output_literals.push_back(
          LiteralUtil::CreateFromArray(expected_output));
    }
    return InputsOutputs{std::move(input_literals),
                         std::move(expected_output_literals)};
  }
};

TEST_P(AllReduceTest, AsyncAllReduce_F32_2GPUs) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY test_computation {
    param_0 = f32[65536] parameter(0)
    ROOT all-reduce = f32[65536] all-reduce(param_0), to_apply=apply_op, replica_groups={{0,1}}
  }
  )";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  int64_t num_elements =
      module->entry_computation()->root_instruction()->shape().dimensions()[0];

  Array<float> input1({num_elements}), input2({num_elements});
  input1.FillRandom(1.0f, 10.0f, /*seed=*/0);
  input2.FillRandom(1.0f, 10.0f, /*seed=*/1);
  Array<float> expected_output({num_elements});
  expected_output.Each([&](absl::Span<const int64_t> indices, float* val) {
    *val = input1(indices) + input2(indices);
  });

  Literal input_literal1 = LiteralUtil::CreateFromArray(input1);
  Literal input_literal2 = LiteralUtil::CreateFromArray(input2);
  Literal expected_output_literal =
      LiteralUtil::CreateFromArray(expected_output);

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        std::vector<std::vector<Literal*>>{{&input_literal1},
                                                           {&input_literal2}}));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[1]));
}

TEST_P(AllReduceTest, AsyncAllReduceInsideWhile_F32_2GPUs) {
  const int64_t kNumElements = 32;
  const int64_t kNumIterations = 3;
  const absl::string_view kReplicaGroups = "{0,1}";
  const auto kModuleStr = absl::StrFormat(
      R"(
  HloModule test

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  while_condition {
    limit = s32[] constant(%1$d)
    params = (s32[], f32[%2$d]{0}) parameter(0)
    loop_counter = get-tuple-element(params), index=0
    ROOT result = pred[] compare(loop_counter, limit), direction=LT
  }

  while_body {
    params = (s32[], f32[%2$d]{0}) parameter(0)
    loop_counter = get-tuple-element(params), index=0
    tensor = get-tuple-element(params), index=1
    out0 = f32[%2$d] all-reduce(tensor), to_apply=apply_op,
      replica_groups={%3$s}
    new_loop_counter = s32[] add(loop_counter, s32[] constant(1))
    ROOT result = (s32[], f32[%2$d]{0}) tuple(new_loop_counter, out0)
  }

  ENTRY test_computation {
    param_0 = f32[%2$d] parameter(0)
    while_init = (s32[], f32[%2$d]{0}) tuple(s32[] constant(0), param_0)
    while_result = (s32[], f32[%2$d]{0})
      while(while_init), condition=while_condition, body=while_body
    ROOT result = get-tuple-element(while_result), index=1
  }
  )",
      kNumIterations, kNumElements, kReplicaGroups);

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  int64_t num_elements =
      module->entry_computation()->root_instruction()->shape().dimensions()[0];

  Array<float> input1({num_elements}), input2({num_elements});
  input1.FillRandom(1.0f, 10.0f, /*seed=*/0);
  input2.FillRandom(1.0f, 10.0f, /*seed=*/1);
  Array<float> expected_output({num_elements});
  expected_output.Each([&](absl::Span<const int64_t> indices, float* val) {
    *val =
        (input1(indices) + input2(indices)) * std::pow(2, kNumIterations - 1);
  });

  Literal input_literal1 = LiteralUtil::CreateFromArray(input1);
  Literal input_literal2 = LiteralUtil::CreateFromArray(input2);
  Literal expected_output_literal =
      LiteralUtil::CreateFromArray(expected_output);

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        std::vector<std::vector<Literal*>>{{&input_literal1},
                                                           {&input_literal2}}));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[1]));
}

TEST_P(AllReduceTest, AsyncAllReduce_BF16_2GPUs) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  apply_op {
    x = bf16[] parameter(0)
    y = bf16[] parameter(1)
    ROOT apply_op = bf16[] add(x, y)
  }

  ENTRY test_computation {
    param_0 = bf16[65536] parameter(0)
    ROOT all-reduce = bf16[65536] all-reduce(param_0), to_apply=apply_op, replica_groups={{0,1}}
  }
  )";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  int64_t num_elements =
      module->entry_computation()->root_instruction()->shape().dimensions()[0];

  Array<bfloat16> input1({num_elements}), input2({num_elements});
  input1.FillRandom(static_cast<bfloat16>(1.0f), 10.0f, /*seed=*/0);
  input2.FillRandom(static_cast<bfloat16>(1.0f), 10.0f, /*seed=*/1);
  Array<bfloat16> expected_output({num_elements});
  expected_output.Each([&](absl::Span<const int64_t> indices, bfloat16* val) {
    *val = input1(indices) + input2(indices);
  });

  Literal input_literal1 = LiteralUtil::CreateFromArray(input1);
  Literal input_literal2 = LiteralUtil::CreateFromArray(input2);
  Literal expected_output_literal =
      LiteralUtil::CreateFromArray(expected_output);

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        std::vector<std::vector<Literal*>>{{&input_literal1},
                                                           {&input_literal2}}));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[1]));
}

void VerifyAllReduceType(const HloModule* module, PrimitiveType expected_type) {
  bool found = false;
  for (auto* comp : module->computations()) {
    for (auto* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kAllReduce ||
          instr->opcode() == HloOpcode::kAllReduceStart) {
        PrimitiveType actual_type = instr->operand(0)->shape().element_type();
        ASSERT_EQ(actual_type, expected_type)
            << "Expected AllReduce type " << PrimitiveType_Name(expected_type)
            << " but got " << PrimitiveType_Name(actual_type);
        found = true;
      }
    }
  }
  ASSERT_TRUE(found) << "No AllReduce found in module";
}

// FP8 vs FP16 training step comparison.
TEST_P(AllReduceTest, AsyncAllReduce_F8E4M3FN_TrainingStep_2GPUs) {
  if (!Capability().IsCuda() ||
      !Capability().cuda_compute_capability()->IsAtLeast(9, 0)) {
    GTEST_SKIP() << "FP8 requires CUDA with Hopper or newer architecture.";
  }

  // FP16 baseline
  const absl::string_view kF16ModuleStr = R"(
  HloModule f16_training_step
  add_f16 { x = f16[] parameter(0)  y = f16[] parameter(1)  ROOT add = f16[] add(x, y) }
  ENTRY training_step {
    activations = f16[32,64] parameter(0)
    weights = f16[64,128] parameter(1)
    upstream_grad = f16[32,128] parameter(2)
    lr = f16[] parameter(3)
    output = f16[32,128] dot(activations, weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    activations_t = f16[64,32] transpose(activations), dimensions={1,0}
    weight_grad = f16[64,128] dot(activations_t, upstream_grad), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    weight_grad_allreduce = f16[64,128] all-reduce(weight_grad), to_apply=add_f16, replica_groups={{0,1}}
    two = f16[] constant(2)
    two_bcast = f16[64,128] broadcast(two), dimensions={}
    weight_grad_avg = f16[64,128] divide(weight_grad_allreduce, two_bcast)
    lr_bcast = f16[64,128] broadcast(lr), dimensions={}
    weight_update = f16[64,128] multiply(lr_bcast, weight_grad_avg)
    new_weights = f16[64,128] subtract(weights, weight_update)
    ROOT result = (f16[32,128], f16[64,128]) tuple(output, new_weights)
  })";

  // FP8 version
  const absl::string_view kF8ModuleStr = R"(
  HloModule fp8_training_step
  add_f8 { x = f8e4m3fn[] parameter(0)  y = f8e4m3fn[] parameter(1)  ROOT add = f8e4m3fn[] add(x, y) }
  ENTRY training_step {
    activations = f16[32,64] parameter(0)
    weights = f16[64,128] parameter(1)
    upstream_grad = f16[32,128] parameter(2)
    lr = f16[] parameter(3)
    output = f16[32,128] dot(activations, weights), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    activations_t = f16[64,32] transpose(activations), dimensions={1,0}
    weight_grad = f16[64,128] dot(activations_t, upstream_grad), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    weight_grad_f8 = f8e4m3fn[64,128] convert(weight_grad)
    weight_grad_allreduce_f8 = f8e4m3fn[64,128] all-reduce(weight_grad_f8), to_apply=add_f8, replica_groups={{0,1}}
    weight_grad_allreduce = f16[64,128] convert(weight_grad_allreduce_f8)
    two = f16[] constant(2)
    two_bcast = f16[64,128] broadcast(two), dimensions={}
    weight_grad_avg = f16[64,128] divide(weight_grad_allreduce, two_bcast)
    lr_bcast = f16[64,128] broadcast(lr), dimensions={}
    weight_update = f16[64,128] multiply(lr_bcast, weight_grad_avg)
    new_weights = f16[64,128] subtract(weights, weight_update)
    ROOT result = (f16[32,128], f16[64,128]) tuple(output, new_weights)
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  Array<Eigen::half> activations1({32, 64}), activations2({32, 64});
  activations1.FillRandom(Eigen::half(0.1f), 0.5f, /*seed=*/0);
  activations2.FillRandom(Eigen::half(0.1f), 0.5f, /*seed=*/1);
  Array<Eigen::half> weights({64, 128});
  weights.FillRandom(Eigen::half(0.1f), 0.3f, /*seed=*/42);
  Array<Eigen::half> upstream_grad1({32, 128}), upstream_grad2({32, 128});
  upstream_grad1.FillRandom(Eigen::half(0.01f), 0.1f, /*seed=*/100);
  upstream_grad2.FillRandom(Eigen::half(0.01f), 0.1f, /*seed=*/101);

  Literal lr = LiteralUtil::CreateR0<Eigen::half>(Eigen::half(0.01f));
  Literal activations_lit1 = LiteralUtil::CreateFromArray(activations1);
  Literal activations_lit2 = LiteralUtil::CreateFromArray(activations2);
  Literal weights_lit = LiteralUtil::CreateFromArray(weights);
  Literal upstream_grad_lit1 = LiteralUtil::CreateFromArray(upstream_grad1);
  Literal upstream_grad_lit2 = LiteralUtil::CreateFromArray(upstream_grad2);

  TF_ASSERT_OK_AND_ASSIGN(auto f16_module, ParseAndReturnVerifiedModule(
                                               kF16ModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult f16_result,
      ExecuteReplicated(
          std::move(f16_module),
          std::vector<std::vector<Literal*>>{
              {&activations_lit1, &weights_lit, &upstream_grad_lit1, &lr},
              {&activations_lit2, &weights_lit, &upstream_grad_lit2, &lr}}));
  // Verify FP16 all-reduce type in optimized module
  VerifyAllReduceType(f16_result.optimized_module, F16);

  TF_ASSERT_OK_AND_ASSIGN(
      auto f8_module, ParseAndReturnVerifiedModule(kF8ModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult f8_result,
      ExecuteReplicated(
          std::move(f8_module),
          std::vector<std::vector<Literal*>>{
              {&activations_lit1, &weights_lit, &upstream_grad_lit1, &lr},
              {&activations_lit2, &weights_lit, &upstream_grad_lit2, &lr}}));
  // Verify FP8 all-reduce type in optimized module
  VerifyAllReduceType(f8_result.optimized_module, F8E4M3FN);

  ASSERT_EQ(f16_result.results.size(), kNumReplicas);
  ASSERT_EQ(f8_result.results.size(), kNumReplicas);

  std::vector<Literal> f16_r0 = f16_result.results[0].DecomposeTuple();
  std::vector<Literal> f8_r0 = f8_result.results[0].DecomposeTuple();
  std::vector<Literal> f8_r1 = f8_result.results[1].DecomposeTuple();

  // Forward outputs should match exactly (no FP8 in forward path)
  EXPECT_TRUE(LiteralTestUtil::Equal(f16_r0[0], f8_r0[0]));

  // FP8 vs FP16 weight comparison: should be close but not identical
  EXPECT_TRUE(
      LiteralTestUtil::Near(f16_r0[1], f8_r0[1], ErrorSpec{1e-2, 1e-2}));

  // Numerical precision check: FP8 should produce measurably different results
  // than FP16. FP8 e4m3 has ~6% relative error (2^-4), FP16 has ~0.1% (2^-10).
  TF_ASSERT_OK_AND_ASSIGN(Literal f16_f32, f16_r0[1].Convert(F32));
  TF_ASSERT_OK_AND_ASSIGN(Literal f8_f32, f8_r0[1].Convert(F32));
  absl::Span<const float> f16_data = f16_f32.data<float>();
  absl::Span<const float> f8_data = f8_f32.data<float>();
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < f16_data.size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(f16_data[i] - f8_data[i]));
  }
  // Expect measurable difference (> FP16 noise floor of ~0.1%)
  EXPECT_GT(max_abs_diff, 1e-3f);
}

// Test that FP8 all-reduce fails on pre-Hopper GPUs.
TEST_P(AllReduceTest, AsyncAllReduce_F8E4M3FN_FailsOnPreHopper) {
  if (!Capability().IsCuda()) {
    GTEST_SKIP() << "Test requires CUDA.";
  }
  if (Capability().cuda_compute_capability()->IsAtLeast(9, 0)) {
    GTEST_SKIP() << "Test requires pre-Hopper GPU (compute capability < 9.0).";
  }

  const absl::string_view kF8ModuleStr = R"(
  HloModule fp8_allreduce_test
  add_f8 { x = f8e4m3fn[] parameter(0)  y = f8e4m3fn[] parameter(1)  ROOT add = f8e4m3fn[] add(x, y) }
  ENTRY test_computation {
    param_0 = f16[64,128] parameter(0)
    param_f8 = f8e4m3fn[64,128] convert(param_0)
    allreduce_f8 = f8e4m3fn[64,128] all-reduce(param_f8), to_apply=add_f8, replica_groups={{0,1}}
    ROOT result = f16[64,128] convert(allreduce_f8)
  })";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kF8ModuleStr, kNumReplicas));

  Array<Eigen::half> input1({64, 128}), input2({64, 128});
  input1.FillRandom(Eigen::half(0.1f), 0.5f, /*seed=*/0);
  input2.FillRandom(Eigen::half(0.1f), 0.5f, /*seed=*/1);
  Literal input_literal1 = LiteralUtil::CreateFromArray(input1);
  Literal input_literal2 = LiteralUtil::CreateFromArray(input2);

  auto result = ExecuteReplicated(
      std::move(module),
      std::vector<std::vector<Literal*>>{{&input_literal1}, {&input_literal2}});

  EXPECT_FALSE(result.ok())
      << "FP8 all-reduce should fail on pre-Hopper GPUs, but succeeded.";
  // NCCL returns ncclInvalidArgument for FP8 reductions on pre-sm90 GPUs.
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr("FP8 reduction support begins with sm90"));
}

TEST_P(AllReduceTest, AsyncAllReduce_PRED_2GPUs) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  apply_op {
    x = pred[] parameter(0)
    y = pred[] parameter(1)
    ROOT apply_op = pred[] or(x, y)
  }

  ENTRY test_computation {
    param_0 = pred[65536] parameter(0)
    ROOT all-reduce = pred[65536] all-reduce(param_0), to_apply=apply_op, replica_groups={{0,1}}
  }
  )";

  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  int64_t num_elements =
      module->entry_computation()->root_instruction()->shape().dimensions()[0];

  Array<bool> input1({num_elements}), input2({num_elements});
  input1.FillRandomBool(/*seed=*/0);
  input2.FillRandomBool(/*seed=*/1);
  Array<bool> expected_output({num_elements});
  expected_output.Each([&](absl::Span<const int64_t> indices, bool* val) {
    *val = input1(indices) | input2(indices);
  });

  Literal input_literal1 = LiteralUtil::CreateFromArray(input1);
  Literal input_literal2 = LiteralUtil::CreateFromArray(input2);
  Literal expected_output_literal =
      LiteralUtil::CreateFromArray(expected_output);

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        std::vector<std::vector<Literal*>>{{&input_literal1},
                                                           {&input_literal2}}));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected_output_literal, results[1]));
}

TEST_P(AllReduceTest, AsyncAllReduce_8GPUs_AllReplicasOneGroup) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY test_computation {
    param_0 = f32[65536] parameter(0)
    ROOT all-reduce = f32[65536] all-reduce(param_0), to_apply=apply_op,
      replica_groups={{0,1,2,3,4,5,6,7}}
  }
  )";

  const int64_t kNumReplicas = 8;
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));
  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      BuildTestInputsOutputs(*module, kNumReplicas, /*num_iterations=*/1));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()))
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    // NB: nccl accumulation order can be different from expected calculations
    // leading to differences in the results (floating point imprecision).
    ASSERT_TRUE(LiteralTestUtil::Near(test_io.expected_outputs[i], results[i],
                                      ErrorSpec{1e-4}))
        << "ExpectedOutput != Result at index " << i;
  }
}

TEST_P(AllReduceTest, AsyncAllReduce_8GPUs_2ReplicasPerGroup) {
  const int64_t kNumElements = 65536;
  const int64_t kNumIterations = 3;
  const auto kModuleStr = absl::StrFormat(
      R"(
  HloModule test

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  while_condition {
    limit = s32[] constant(%1$d)
    params = (s32[], f32[%2$d]{0}) parameter(0)
    loop_counter = get-tuple-element(params), index=0
    ROOT result = pred[] compare(loop_counter, limit), direction=LT
  }

  while_body {
    params = (s32[], f32[%2$d]{0}) parameter(0)
    loop_counter = get-tuple-element(params), index=0
    tensor = get-tuple-element(params), index=1
    out0 = f32[%2$d] all-reduce(tensor), to_apply=apply_op,
      replica_groups={{0,4},{1,5},{2,6},{3,7}}
    new_loop_counter = s32[] add(loop_counter, s32[] constant(1))
    ROOT result = (s32[], f32[%2$d]{0}) tuple(new_loop_counter, out0)
  }

  ENTRY test_computation {
    param_0 = f32[%2$d] parameter(0)
    while_init = (s32[], f32[%2$d]{0}) tuple(s32[] constant(0), param_0)
    while_result = (s32[], f32[%2$d]{0})
      while(while_init), condition=while_condition, body=while_body
    ROOT result = get-tuple-element(while_result), index=1
  }
  )",
      kNumIterations, kNumElements);

  const int64_t kNumReplicas = 8;
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleStr, kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(
      InputsOutputs test_io,
      BuildTestInputsOutputs(*module, kNumReplicas, kNumIterations));

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result,
      ExecuteReplicated(std::move(module),
                        /*arguments=*/test_io.InputLiteralPtrs()));
  const std::vector<Literal>& results = execution_result.results;
  ASSERT_EQ(results.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    ASSERT_TRUE(LiteralTestUtil::Equal(test_io.expected_outputs[i], results[i]))
        << "ExpectedOutput != Result at index " << i;
  }
}

TEST_F(CollectiveOpsTestE2E, OptimizedSubByteAllGatherOnDim0OutputIsCorrect) {
  constexpr int kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(auto unoptimized_module,
                          ParseAndReturnVerifiedModule(R"(
    HloModule m, replica_count=2

    e {
      a = s4[2,4]{1,0:E(4)} constant({{0,1,2,3},{4,5,5,4}})
      b = s4[4,4]{1,0:E(4)} all-gather(a), dimensions={0}
    })",
                                                       kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(unoptimized_module)));

  const HloModule* module = execution_result.optimized_module;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Bitcast(m::AllGatherDone().WithShape(S8, {4, 2}))));

  const Literal expected_result =
      LiteralUtil::CreateR2<s4>({{s4(0), s4(1), s4(2), s4(3)},
                                 {s4(4), s4(5), s4(5), s4(4)},
                                 {s4(0), s4(1), s4(2), s4(3)},
                                 {s4(4), s4(5), s4(5), s4(4)}});

  const std::vector<Literal>& result = execution_result.results;
  ASSERT_EQ(result.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_result, result[i]))
        << "Results differ at replica " << i;
  }
}

TEST_F(CollectiveOpsTestE2E, OptimizedSubByteAllGatherOnDim1OutputIsCorrect) {
  constexpr int kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "Test requires at least " << kNumReplicas << " devices ("
      << hlo_runner_->device_count() << " available)";

  TF_ASSERT_OK_AND_ASSIGN(auto unoptimized_module,
                          ParseAndReturnVerifiedModule(R"(
    HloModule m, replica_count=2

    e {
      a = s4[4,2]{1,0:E(4)} constant({{0,1},{2,3},{4,5},{5,4}})
      b = s4[4,4]{1,0:E(4)} all-gather(a), dimensions={1}
    })",
                                                       kNumReplicas));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionResult execution_result,
                          ExecuteReplicated(std::move(unoptimized_module)));

  const HloModule* module = execution_result.optimized_module;
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Fusion(
                        m::Bitcast(m::AllGatherDone().WithShape(S8, {2, 4})))));
  EXPECT_THAT(root->fused_expression_root(),
              GmockMatch(m::Transpose(m::Parameter())));

  const Literal expected_result =
      LiteralUtil::CreateR2<s4>({{s4(0), s4(1), s4(0), s4(1)},
                                 {s4(2), s4(3), s4(2), s4(3)},
                                 {s4(4), s4(5), s4(4), s4(5)},
                                 {s4(5), s4(4), s4(5), s4(4)}});

  const std::vector<Literal>& result = execution_result.results;
  ASSERT_EQ(result.size(), kNumReplicas);
  for (int i = 0; i < kNumReplicas; ++i) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_result, result[i]))
        << "Results differ at replica " << i;
  }
}

TEST_F(CollectiveOpsTestE2E, AllGatherOnChangedDimensionIsCorrect) {
  const int64_t kNumReplicas = 2;
  ASSERT_GE(hlo_runner_->device_count(), kNumReplicas)
      << "The test requires at least " << kNumReplicas << " devices";

  TF_ASSERT_OK_AND_ASSIGN(auto unoptimized_module,
                          ParseAndReturnVerifiedModule(R"(
  HloModule m, replica_count=2
  e {
    a = u32[2,2,3] constant({{{0,1,2},{3,4,5}},{{6,7,8},{9,10,11}}})
    g = u32[2,4,3] all-gather(a), dimensions={1}
  })"));
  TF_ASSERT_OK_AND_ASSIGN(auto executable, hlo_runner_->CreateExecutable(
                                               std::move(unoptimized_module),
                                               /*run_hlo_passes=*/true));
  TF_ASSERT_OK_AND_ASSIGN(const HloModule* module,
                          hlo_runner_->HloModuleFromWrapped(executable.get()));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  EXPECT_THAT(root, GmockMatch(m::Fusion(m::AllGatherDone(
                        m::AllGatherStart(m::Bitcast(m::Constant()))))));
  EXPECT_THAT(root->fused_expression_root(),
              GmockMatch(m::Transpose(m::Bitcast(m::Parameter()))));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  ASSERT_EQ(results.size(), kNumReplicas);
  Literal expected = LiteralUtil::CreateR3<uint32_t>(
      {{{0, 1, 2}, {3, 4, 5}, {0, 1, 2}, {3, 4, 5}},
       {{6, 7, 8}, {9, 10, 11}, {6, 7, 8}, {9, 10, 11}}});
  for (const Literal& result : results) {
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllReduceTest, AllReduceTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool()),
    [](const ::testing::TestParamInfo<std::tuple<bool, bool>>& info) {
      return absl::StrCat(GetAsyncTestName(std::get<0>(info.param)), "_",
                          std::get<1>(info.param) ? "one_shot" : "nccl");
    });

TEST_F(CollectiveOpsTestE2E, MultipleModuleDifferentDeviceGroupsShouldRun) {
  const absl::string_view kModuleStr_1 = R"(
  HloModule test

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY test_computation {
    param_0 = f32[8] parameter(0)
    ROOT all-reduce = f32[8] all-reduce(param_0), to_apply=apply_op, replica_groups={{0,1}}
  }
  )";
  const absl::string_view kModuleStr_2 = R"(
  HloModule test

  apply_op {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT apply_op = f32[] add(x, y)
  }

  ENTRY test_computation {
    param_0 = f32[8] parameter(0)
    all-reduce.1 = f32[8] all-reduce(param_0), to_apply=apply_op, replica_groups={{0,1}, {2,3}}
    all-reduce.2 = f32[8] all-reduce(all-reduce.1), to_apply=apply_op, replica_groups={{0,1}, {2,3}}
    all-reduce.3 = f32[8] all-reduce(all-reduce.2), to_apply=apply_op, replica_groups={{0,1}, {2,3}}
    ROOT all-reduce.4 = f32[8] all-reduce(all-reduce.3), to_apply=apply_op, replica_groups={{0,1,2,3}}
  }
  )";

  const int64_t kNumReplicas_1 = 2;
  const int64_t kNumReplicas_2 = 4;
  if (hlo_runner_->device_count() < kNumReplicas_2) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas_2 << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }

  HloModuleConfig config_1 =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas_1);
  HloModuleConfig config_2 =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas_2);

  TF_ASSERT_OK_AND_ASSIGN(auto module_1,
                          ParseAndReturnVerifiedModule(kModuleStr_1, config_1));

  TF_ASSERT_OK_AND_ASSIGN(auto module_2,
                          ParseAndReturnVerifiedModule(kModuleStr_2, config_2));

  int64_t num_elements_1 = ShapeUtil::ElementsIn(
      module_1->entry_computation()->parameter_instructions()[0]->shape());

  int64_t num_elements_2 = ShapeUtil::ElementsIn(
      module_2->entry_computation()->parameter_instructions()[0]->shape());

  Array<float> input1_1({num_elements_1}), input1_2({num_elements_1});
  input1_1.FillRandom(1.0f, 10.0f, /*seed=*/0);
  input1_2.FillRandom(1.0f, 10.0f, /*seed=*/1);

  Literal input_literal1_1 = LiteralUtil::CreateFromArray(input1_1);
  Literal input_literal1_2 = LiteralUtil::CreateFromArray(input1_2);

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result_1,
      ExecuteReplicated(std::move(module_1),
                        std::vector<std::vector<Literal*>>{
                            {&input_literal1_1}, {&input_literal1_2}}));

  Array<float> input2_1({num_elements_2}), input2_2({num_elements_2}),
      input2_3({num_elements_2}), input2_4({num_elements_2});
  input2_1.FillRandom(1.0f, 10.0f, /*seed=*/0);
  input2_2.FillRandom(1.0f, 10.0f, /*seed=*/1);
  input2_3.FillRandom(1.0f, 10.0f, /*seed=*/2);
  input2_4.FillRandom(1.0f, 10.0f, /*seed=*/3);

  Literal input_literal2_1 = LiteralUtil::CreateFromArray(input2_1);
  Literal input_literal2_2 = LiteralUtil::CreateFromArray(input2_2);
  Literal input_literal2_3 = LiteralUtil::CreateFromArray(input2_3);
  Literal input_literal2_4 = LiteralUtil::CreateFromArray(input2_4);

  TF_ASSERT_OK_AND_ASSIGN(
      ExecutionResult execution_result_2,
      ExecuteReplicated(std::move(module_2), std::vector<std::vector<Literal*>>{
                                                 {&input_literal2_1},
                                                 {&input_literal2_2},
                                                 {&input_literal2_3},
                                                 {&input_literal2_4}}));
}
}  // namespace
}  // namespace xla
