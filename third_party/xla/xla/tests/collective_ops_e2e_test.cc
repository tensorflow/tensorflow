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

#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/test_utils.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;
using ::testing::NotNull;

// Makes a DeviceAssignment device#i to replica_id #i.
DeviceAssignment MakeDeviceAssn(int64_t num_replicas) {
  DeviceAssignment assn(/*replica_count=*/num_replicas,
                        /*computation_count=*/1);
  for (int64_t i = 0; i < num_replicas; ++i) {
    assn(i, 0) = i;
  }
  return assn;
}

class CollectiveOpsTestE2E : public HloTestBase {
 public:
  CollectiveOpsTestE2E() {
    replacements_[kF8E4M3DatatypePlaceholder] =
        IsCuda() ? "f8e4m3fn" : "f8e4m3fnuz";
    replacements_[kF8E5M2DatatypePlaceholder] =
        IsCuda() ? "f8e5m2" : "f8e5m2fnuz";
  }

  bool IsCuda() {
    return std::holds_alternative<se::CudaComputeCapability>(Capability());
  }

  const se::GpuComputeCapability& Capability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability();
  }

  bool HasFp8Support() {
    if (IsCuda()) {
      return std::get<se::CudaComputeCapability>(Capability()).IsAtLeast(8, 9);
    }
    return std::get<se::RocmComputeCapability>(Capability())
               .has_fp8_support() &&
           GetDebugOptionsForTest().xla_gpu_enable_cublaslt();
  }

  void CollectiveOpsVerifyF8Matmul(absl::string_view hlo_text,
                                   const DebugOptions& options) {
    if (!HasFp8Support()) {
      return;
    }
    const int64_t kNumReplicas = 1;
    const int64_t kNumPartitions = 4;

    HloModuleConfig config =
        GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
    config.set_debug_options(options);
    config.set_num_partitions(kNumPartitions);
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_text, config));

    TF_ASSERT_OK_AND_ASSIGN(auto executable,
                            CreateExecutable(std::move(module),
                                             /*run_hlo_passes=*/true));
    EXPECT_TRUE(executable->has_module());
    std::vector<HloInstruction*> gemm_ops =
        FindInstructions(&executable->module(), HloOpcode::kCustomCall);
    for (HloInstruction* gemm_op : gemm_ops) {
      EXPECT_EQ(gemm_op->custom_call_target(), "__cublas$lt$matmul$f8");
    }
  }

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(Executable* executable,
                                                         int64_t num_replicas) {
    DeviceAssignment device_assignment = MakeDeviceAssn(num_replicas);
    return HloTestBase::ExecuteReplicated(
        /*executable_provider*/ [&](int64_t) { return executable; },
        /*argument_count_provider*/ [](int64_t) { return 0; },
        /*argument_provider*/ [](int64_t, int64_t) { return nullptr; },
        num_replicas, /*run_hlo_passes=*/false, &device_assignment);
  }

 protected:
  absl::flat_hash_map<absl::string_view, absl::string_view> replacements_;

 private:
  static constexpr const char* kF8E4M3DatatypePlaceholder{"<<F8E4M3>>"};
  static constexpr const char* kF8E5M2DatatypePlaceholder{"<<F8E5M2>>"};
};

// E2E tests for collective ops. These will generally verify some HLO transform
// for collectives (for example, sync -> async conversion) and correct
// execution of the transformed HLO.

// E2E test for async collectives. Tested with both async collective enabled
// and disabled. Verify that async collective is generated when enabled
// in the end-to-end compilation for GPU's and that the execution produces
// correct result.
class AsyncCollectiveOps : public CollectiveOpsTestE2E,
                           public ::testing::WithParamInterface<bool> {
 public:
  AsyncCollectiveOps() : num_devices_(backend().device_count()) {
    VLOG(1) << "Running with " << num_devices_ << " devices";
  }

 protected:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();

    // Enable or disable all async collectives based on test parameter.
    bool enable_async = GetParam();
    if (!enable_async) {
      for (auto option :
           {DebugOptions::NOOP, DebugOptions::ALLREDUCE,
            DebugOptions::ALLGATHER, DebugOptions::REDUCESCATTER,
            DebugOptions::COLLECTIVEBROADCAST, DebugOptions::ALLTOALL,
            DebugOptions::COLLECTIVEPERMUTE}) {
        debug_options.add_xla_gpu_disable_async_collectives(option);
      }
    }
    debug_options.add_xla_disable_hlo_passes(
        "gpu-convert-async-collectives-to-sync");
    return debug_options;
  }

  absl::StatusOr<std::unique_ptr<Executable>> CreateExecutable(
      absl::string_view hlo_string, int64_t num_replicas) {
    HloModuleConfig config =
        GetModuleConfigForTest(/*replica_count=*/num_replicas);

    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnVerifiedModule(hlo_string, config));
    return HloTestBase::CreateExecutable(std::move(module),
                                         /*run_hlo_passes=*/true);
  }

  bool IsAsync(const HloInstruction* inst) {
    return !inst->backend_config<gpu::GpuBackendConfig>()
                .value()
                .collective_backend_config()
                .is_sync();
  }

  const int64_t num_devices_;
};

XLA_TEST_P(AsyncCollectiveOps, AsyncAllReduce) {
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);
  const bool enable_async_all_reduce = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CreateExecutable(kModuleStr, kNumReplicas));
  EXPECT_TRUE(executable->has_module());

  HloInstruction* all_reduce_start =
      FindInstruction(&executable->module(), HloOpcode::kAllReduceStart);
  HloInstruction* all_reduce_done =
      FindInstruction(&executable->module(), HloOpcode::kAllReduceDone);
  EXPECT_THAT(all_reduce_start, NotNull());
  EXPECT_THAT(all_reduce_done, NotNull());
  EXPECT_EQ(IsAsync(all_reduce_start), enable_async_all_reduce);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  ASSERT_EQ(results.size(), kNumReplicas);
  // sum [0, num_devices)
  const uint32_t expected = kNumReplicas * (kNumReplicas - 1) / 2;
  for (int i = 0; i < kNumReplicas; ++i) {
    LiteralTestUtil::ExpectR0Equal<uint32_t>(expected, results[i]);
  }
}

XLA_TEST_P(AsyncCollectiveOps, AsyncAllGather) {
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);
  const bool enable_async_all_gather = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CreateExecutable(kModuleStr, kNumReplicas));

  EXPECT_TRUE(executable->has_module());
  HloInstruction* all_gather_start =
      FindInstruction(&executable->module(), HloOpcode::kAllGatherStart);
  HloInstruction* all_gather_done =
      FindInstruction(&executable->module(), HloOpcode::kAllGatherDone);
  EXPECT_THAT(all_gather_start, NotNull());
  EXPECT_THAT(all_gather_done, NotNull());
  EXPECT_EQ(IsAsync(all_gather_start), enable_async_all_gather);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));

  ASSERT_EQ(results.size(), kNumReplicas);
  for (const Literal& result : results) {
    LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, result);
  }
}

XLA_TEST_P(AsyncCollectiveOps, AsyncAllGatherMixedTypes) {
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);
  const bool enable_async_all_gather = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CreateExecutable(kModuleStr, kNumReplicas));
  EXPECT_TRUE(executable->has_module());
  HloInstruction* all_gather_start =
      FindInstruction(&executable->module(), HloOpcode::kAllGatherStart);
  HloInstruction* all_gather_done =
      FindInstruction(&executable->module(), HloOpcode::kAllGatherDone);
  EXPECT_THAT(all_gather_start, NotNull());
  EXPECT_THAT(all_gather_done, NotNull());
  EXPECT_EQ(IsAsync(all_gather_start), enable_async_all_gather);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));

  ASSERT_EQ(results.size(), kNumReplicas);
  for (Literal& result : results) {
    std::vector<Literal> results = result.DecomposeTuple();
    LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, results[0]);
    LiteralTestUtil::ExpectR1Equal<float>({10.0, 15.0, 11.0, 16.0}, results[1]);
  }
}

XLA_TEST_P(AsyncCollectiveOps, AsyncCollectiveBroadcast) {
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);
  const bool enable_async_collective_broadcast = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CreateExecutable(kModuleStr, kNumReplicas));
  EXPECT_TRUE(executable->has_module());
  HloInstruction* cb_start =
      FindInstruction(&executable->module(), HloOpcode::kAsyncStart);
  HloInstruction* cb_done =
      FindInstruction(&executable->module(), HloOpcode::kAsyncDone);
  EXPECT_THAT(cb_start, NotNull());
  EXPECT_THAT(cb_done, NotNull());
  EXPECT_EQ(IsAsync(cb_start), enable_async_collective_broadcast);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 11}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 11}, results[1]);
}

XLA_TEST_P(AsyncCollectiveOps, AsyncCollectivePermute) {
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);
  const bool enable_async_collective_permute = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CreateExecutable(kModuleStr, kNumReplicas));
  EXPECT_TRUE(executable->has_module());
  HloInstruction* cp_start = FindInstruction(
      &executable->module(), HloOpcode::kCollectivePermuteStart);
  HloInstruction* cp_done =
      FindInstruction(&executable->module(), HloOpcode::kCollectivePermuteDone);
  EXPECT_THAT(cp_start, NotNull());
  EXPECT_THAT(cp_done, NotNull());
  EXPECT_EQ(IsAsync(cp_start), enable_async_collective_permute);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 11}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 10}, results[1]);
}

XLA_TEST_P(AsyncCollectiveOps, AsyncReduceScatter) {
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);
  const bool enable_async_reduce_scatter = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CreateExecutable(kModuleStr, kNumReplicas));
  EXPECT_TRUE(executable->has_module());
  HloInstruction* rs_start =
      FindInstruction(&executable->module(), HloOpcode::kAsyncStart);
  HloInstruction* rs_done =
      FindInstruction(&executable->module(), HloOpcode::kAsyncDone);
  ASSERT_THAT(rs_start, NotNull());
  ASSERT_THAT(rs_done, NotNull());
  HloAsyncInstruction* rs_start_async = Cast<HloAsyncInstruction>(rs_start);
  EXPECT_EQ(rs_start_async->async_wrapped_opcode(), HloOpcode::kReduceScatter);
  EXPECT_EQ(IsAsync(rs_start), enable_async_reduce_scatter);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 13, 15, 17}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({19, 21, 23, 25}, results[1]);
}

XLA_TEST_P(AsyncCollectiveOps, AsyncAllToAllWithSplitDim) {
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);
  const bool enable_async_all_to_all = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CreateExecutable(kModuleStr, kNumReplicas));
  EXPECT_TRUE(executable->has_module());

  HloInstruction* a2a_start =
      FindInstruction(&executable->module(), HloOpcode::kAsyncStart);
  HloInstruction* a2a_done =
      FindInstruction(&executable->module(), HloOpcode::kAsyncDone);
  ASSERT_THAT(a2a_start, NotNull());
  ASSERT_THAT(a2a_done, NotNull());
  HloAsyncInstruction* a2a_start_async = Cast<HloAsyncInstruction>(a2a_start);
  EXPECT_EQ(a2a_start_async->async_wrapped_opcode(), HloOpcode::kAllToAll);
  EXPECT_EQ(IsAsync(a2a_start), enable_async_all_to_all);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 11}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({15, 16}, results[1]);
}

TEST_F(CollectiveOpsTestE2E, AsyncAllToAllMemCpy) {
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

  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_use_memcpy_local_p2p(true);
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true));
  ASSERT_TRUE(executable->has_module());
  HloModule* executable_module = &executable->module();

  // Verify that the all-to-all is not decomposed into a tuple all-to-all.
  const HloInstruction* all_to_all =
      FindInstruction(executable_module, HloOpcode::kAllToAll);
  EXPECT_THAT(all_to_all, op::Shape("u32[2, 2]"));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({20, 25, 21, 26}, results[1]);
}

XLA_TEST_P(AsyncCollectiveOps, AsyncAllToAllWithoutSplitDim) {
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);
  const bool enable_async_all_to_all = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CreateExecutable(kModuleStr, kNumReplicas));
  EXPECT_TRUE(executable->has_module());
  HloInstruction* a2a_start =
      FindInstruction(&executable->module(), HloOpcode::kAsyncStart);
  HloInstruction* a2a_done =
      FindInstruction(&executable->module(), HloOpcode::kAsyncDone);
  ASSERT_THAT(a2a_start, NotNull());
  ASSERT_THAT(a2a_done, NotNull());
  HloAsyncInstruction* a2a_start_async = Cast<HloAsyncInstruction>(a2a_start);
  EXPECT_EQ(a2a_start_async->async_wrapped_opcode(), HloOpcode::kAllToAll);
  EXPECT_EQ(IsAsync(a2a_start_async), enable_async_all_to_all);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({40, 60, 44, 64}, results[1]);
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  auto opts = GetDebugOptionsForTest();
  opts.set_xla_gpu_enable_cublaslt(GetParam());
  VLOG(0) << "Running with CUBLAS enabled: " << opts.xla_gpu_enable_cublaslt();
  config.set_debug_options(opts);

  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));
  DeviceAssignment assn(/*replica_count=*/kNumReplicas,
                        /*computation_count=*/1);
  for (int64_t i = 0; i < kNumReplicas; ++i) {
    assn(i, 0) = i;
  }

  auto fake_arguments = xla::MakeFakeArguments(module.get()).value();
  std::vector<Literal*> fake_ptrs(fake_arguments.size());
  for (int i = 0; i < fake_arguments.size(); i++) {
    fake_ptrs[i] = &fake_arguments[i];
  }
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          HloTestBase::ExecuteReplicated(
                              std::move(module), fake_ptrs, kNumReplicas, &assn,
                              true /*run_hlo_passes*/, true /*use-threads*/));
  ASSERT_EQ(results.size(), kNumReplicas);

  auto& ref_runner = HloTestBase::reference_runner_;
  TF_ASSERT_OK_AND_ASSIGN(
      auto ref_module, ParseAndReturnVerifiedModule(kModuleSingleStr, config));
  TF_ASSERT_OK_AND_ASSIGN(
      auto ref_exec, ref_runner.CreateExecutable(std::move(ref_module), true));

  ErrorSpec error_spec{1e-5, 1e-5};
  fake_ptrs.push_back(nullptr);
  for (int i = 0; i < kNumReplicas; i++) {
    auto replica_id =
        LiteralUtil::CreateFullWithDescendingLayout<uint32_t>({}, i);
    fake_ptrs.back() = &replica_id;
    TF_ASSERT_OK_AND_ASSIGN(
        auto res, ref_runner.ExecuteWithExecutable(ref_exec.get(), fake_ptrs));
    EXPECT_TRUE(LiteralTestUtil::Near(res, results[i], error_spec));
  }
}

INSTANTIATE_TEST_SUITE_P(AsyncCollectiveOps, AsyncCollectiveOps,
                         ::testing::Bool());

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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);

  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_while_loop_reduce_scatter_code_motion(true);
  HloModuleConfig config;
  config.set_debug_options(debug_options);
  config.set_replica_count(kNumReplicas);
  config.set_num_partitions(1);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true));
  ASSERT_TRUE(executable->has_module());
  HloModule* executable_module = &executable->module();

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

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true));
  ASSERT_TRUE(executable->has_module());
  HloModule* executable_module = &executable->module();

  // Verify that the all-to-all is not decomposed into a tuple all-to-all.
  const HloInstruction* all_to_all =
      FindInstruction(executable_module, HloOpcode::kAllToAll);
  EXPECT_THAT(all_to_all, op::Shape("u32[2, 2]"));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({20, 25, 21, 26}, results[1]);
}

// E2E tests comparing the results of windowed einsum and non-windowed cases.
class CollectiveOpsTestE2EWindowedNonWindowed : public CollectiveOpsTestE2E {
 public:
  void CollectiveOpsCompareWindowedNonWindowed(
      absl::string_view hlo_text, bool disable_dot_merger = false) {
    const int64_t kNumReplicas = 1;
    const int64_t kNumPartitions = 4;
    SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas * kNumPartitions);

    HloModuleConfig config =
        GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
    auto opts = GetDebugOptionsForTest();
    opts.set_xla_gpu_threshold_for_windowed_einsum_mib(0);
    opts.set_xla_gpu_multi_streamed_windowed_einsum(true);
    opts.set_xla_gpu_graph_min_graph_size(200);
    opts.set_xla_gpu_enable_triton_gemm(false);
    if (disable_dot_merger) {
      opts.add_xla_disable_hlo_passes("dot-merger");
    }
    config.set_debug_options(opts);
    config.set_num_partitions(kNumPartitions);
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    DeviceAssignment assn(/*replica_count=*/kNumReplicas,
                          /*computation_count=*/kNumPartitions);
    config.set_replica_count(kNumReplicas);
    for (int64_t i = 0; i < kNumPartitions; ++i) {
      assn(0, i) = i;
    }

    auto fake_arguments = xla::MakeFakeArguments(module.get()).value();
    std::vector<Literal*> fake_ptrs(fake_arguments.size());
    for (int i = 0; i < fake_arguments.size(); i++) {
      fake_ptrs[i] = &fake_arguments[i];
    }

    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<Literal> results,
        HloTestBase::ExecuteReplicated(
            std::move(module), fake_ptrs, kNumPartitions, &assn,
            true /*run_hlo_passes*/, true /*use-threads*/));
    ASSERT_EQ(results.size(), kNumPartitions);
    HloModuleConfig ref_config =
        GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
    auto ref_opts = GetDebugOptionsForTest();
    ref_opts.set_xla_gpu_graph_min_graph_size(200);
    ref_opts.set_xla_gpu_enable_triton_gemm(false);
    if (disable_dot_merger) {
      ref_opts.add_xla_disable_hlo_passes("dot-merger");
    }
    ref_config.set_debug_options(ref_opts);
    ref_config.set_num_partitions(kNumPartitions);
    TF_ASSERT_OK_AND_ASSIGN(auto ref_module,
                            ParseAndReturnVerifiedModule(hlo_text, ref_config));
    auto fake_ref_arguments = xla::MakeFakeArguments(ref_module.get()).value();
    std::vector<Literal*> ref_fake_ptrs(fake_ref_arguments.size());
    for (int i = 0; i < fake_ref_arguments.size(); i++) {
      ref_fake_ptrs[i] = &fake_ref_arguments[i];
    }

    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<Literal> ref_results,
        HloTestBase::ExecuteReplicated(
            std::move(ref_module), ref_fake_ptrs, kNumPartitions, &assn,
            true /*run_hlo_passes*/, true /*use-threads*/));
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
  CollectiveOpsCompareWindowedNonWindowed(
      absl::StrReplaceAll(kModuleReplicatedStr, replacements_),
      /*disable_dot_merger=*/true);

  // Verify the creation of FP8 GEMM Custom Calls on Hopper and newer
  // architectures.
  DebugOptions opts = GetDebugOptionsForTest();
  opts.set_xla_gpu_threshold_for_windowed_einsum_mib(0);
  opts.set_xla_gpu_multi_streamed_windowed_einsum(true);
  opts.set_xla_gpu_graph_min_graph_size(200);
  opts.set_xla_gpu_enable_triton_gemm(false);
  opts.add_xla_disable_hlo_passes("dot-merger");
  CollectiveOpsVerifyF8Matmul(
      absl::StrReplaceAll(kModuleReplicatedStr, replacements_), opts);
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

  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr);
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

  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr);
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

  CollectiveOpsCompareWindowedNonWindowed(kModuleReplicatedStr);
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

  const int64_t kNumReplicas = 1;
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  auto opts = GetDebugOptionsForTest();
  opts.set_xla_gpu_enable_pipelined_collectives(true);
  opts.set_xla_gpu_enable_triton_gemm(false);
  CollectiveOpsVerifyF8Matmul(
      absl::StrReplaceAll(kModuleReplicatedStr, replacements_), opts);
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas * kNumPartitions);

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.set_num_partitions(kNumPartitions);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CreateExecutable(std::move(module),
                                           /*run_hlo_passes=*/true));
  EXPECT_TRUE(executable->has_module());
  HloInstruction* all_to_all =
      FindInstruction(&executable->module(), HloOpcode::kAllToAll);
  EXPECT_THAT(all_to_all, NotNull());
  EXPECT_EQ(all_to_all->shape().element_type(), BF16);

  // Execute the test on 2 partitions.
  TF_ASSERT_OK_AND_ASSIGN(
      module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));
  DeviceAssignment assignment(/*replica_count=*/kNumReplicas,
                              /*computation_count=*/kNumPartitions);
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    assignment(0, i) = i;
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      HloTestBase::ExecuteReplicated(std::move(module), {}, kNumPartitions,
                                     &assignment, /*run_hlo_passes=*/true,
                                     /*use_threads=*/true));
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas * kNumPartitions);

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.set_num_partitions(kNumPartitions);

  // Verify that the element type of the all-to-all has been changed to BF16.
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CreateExecutable(std::move(module),
                                           /*run_hlo_passes=*/true));
  EXPECT_TRUE(executable->has_module());
  HloInstruction* all_to_all =
      FindInstruction(&executable->module(), HloOpcode::kAllToAll);
  EXPECT_THAT(all_to_all, NotNull());
  EXPECT_EQ(all_to_all->shape().element_type(), BF16);

  // Execute the test on 2 partitions.
  TF_ASSERT_OK_AND_ASSIGN(
      module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));
  DeviceAssignment assignment(/*replica_count=*/kNumReplicas,
                              /*computation_count=*/kNumPartitions);
  for (int64_t i = 0; i < kNumPartitions; ++i) {
    assignment(0, i) = i;
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      HloTestBase::ExecuteReplicated(std::move(module), {}, kNumPartitions,
                                     &assignment, /*run_hlo_passes=*/true,
                                     /*use_threads=*/true));
  ASSERT_EQ(results.size(), kNumPartitions);
  LiteralTestUtil::ExpectR1Equal<float>({4., 4.}, results[0]);
  LiteralTestUtil::ExpectR1Equal<float>({8., 8.}, results[1]);
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
  SKIP_TEST_IF_NUM_DEVICES_LESS_THAN(kNumReplicas);
  const int64_t kNumPartitions = 4;

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);

  auto opts = GetDebugOptionsForTest();
  opts.set_xla_experimental_ignore_channel_id(true);
  config.set_debug_options(opts);

  config.set_num_partitions(kNumPartitions);
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(kModuleReplicatedStr, config));

  TF_ASSERT_OK_AND_ASSIGN(auto executable,
                          CreateExecutable(std::move(module),
                                           /*run_hlo_passes=*/true));
  EXPECT_TRUE(executable->has_module());
}

}  // namespace
}  // namespace xla
