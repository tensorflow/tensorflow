/* Copyright 2026 The OpenXLA Authors.

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

// AOT compatibility of collective scheduling structure on 2 GPUs: async
// collectives on multiple communication streams, and collectives nested in
// WhileThunk and ConditionalThunk bodies.

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/literal.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/aot_compatibility/gpu/collective_aot_test_base.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"

namespace xla {
namespace aot_compatibility {
namespace {

using ::testing::ValuesIn;

constexpr absl::string_view kTargetBase = "async_collectives_aot_test_2gpu";
constexpr int64_t kNumReplicas = 2;

class AsyncCollectivesAotTest : public CollectiveAotTestBase {};

// Two independent async collectives in flight at once on separate
// communication streams (AsyncStartThunk communication_stream_id 0 and 1).
TEST_P(AsyncCollectivesAotTest, MultiStream) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] add(lhs, rhs)
    }

    ENTRY main {
      id = u32[] replica-id()
      one = u32[] constant(1)
      id1 = u32[] add(id, one)
      v = u32[2] broadcast(id1), dimensions={}
      ar = u32[2] all-reduce(v), replica_groups={{0,1}}, to_apply=add_u32
      ag = u32[4] all-gather(v), dimensions={0}, replica_groups={{0,1}}
      ROOT out = (u32[2], u32[4]) tuple(ar, ag)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  KeepCollectivesAsync(debug_options);
  debug_options.set_xla_gpu_experimental_parallel_collective_overlap_limit(2);
  debug_options.set_xla_gpu_experimental_enable_collective_multi_streaming(
      true);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  for (int64_t replica = 0; replica < kNumReplicas; ++replica) {
    std::vector<Literal> outputs = results[replica].DecomposeTuple();
    LiteralTestUtil::ExpectR1Equal<uint32_t>({3, 3}, outputs[0]);
    LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2}, outputs[1]);
  }
}

// Async collective-permute and all-reduce inside a WhileThunk body.
TEST_P(AsyncCollectivesAotTest, CollectivesInWhileLoop) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] add(lhs, rhs)
    }

    body {
      param = (u32[], u32[2], u32[2]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      cp_data = u32[2] get-tuple-element(param), index=1
      ar_data = u32[2] get-tuple-element(param), index=2
      replica = u32[] replica-id()
      replica_b = u32[2] broadcast(replica), dimensions={}
      cp_in = u32[2] add(cp_data, replica_b)
      cp = u32[2] collective-permute(cp_in), source_target_pairs={{0,1},{1,0}}
      one = u32[] constant(1)
      one_b = u32[2] broadcast(one), dimensions={}
      ar_in = u32[2] add(ar_data, one_b)
      ar = u32[2] all-reduce(ar_in), replica_groups={{0,1}}, to_apply=add_u32
      i_next = u32[] add(i, one)
      ROOT next = (u32[], u32[2], u32[2]) tuple(i_next, cp, ar)
    }

    cond {
      param = (u32[], u32[2], u32[2]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      limit = u32[] constant(4)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      replica = u32[] replica-id()
      ten = u32[] constant(10)
      init_cp = u32[] add(replica, ten)
      cp_b = u32[2] broadcast(init_cp), dimensions={}
      ar_b = u32[2] broadcast(replica), dimensions={}
      zero = u32[] constant(0)
      init = (u32[], u32[2], u32[2]) tuple(zero, cp_b, ar_b)
      loop = (u32[], u32[2], u32[2]) while(init), condition=cond, body=body
      cp_out = u32[2] get-tuple-element(loop), index=1
      ar_out = u32[2] get-tuple-element(loop), index=2
      ROOT out = (u32[2], u32[2]) tuple(cp_out, ar_out)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  KeepCollectivesAsync(config.mutable_debug_options());

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // collective-permute: each iteration adds the replica id, then swaps.
  //   init r0 = 10, r1 = 11 -> (12, 10) -> (11, 12) -> (13, 11) -> (12, 13).
  // all-reduce: each iteration adds 1, then sums over both replicas.
  //   init r0 = 0, r1 = 1 -> 3 -> 8 -> 18 -> 38.
  std::vector<Literal> r0 = results[0].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({12, 12}, r0[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({38, 38}, r0[1]);
  std::vector<Literal> r1 = results[1].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({13, 13}, r1[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({38, 38}, r1[1]);
}

// SendThunk and RecvThunk inside a WhileThunk body, across 2 partitions. Only
// partition 0 sends, so the recv of partition 1 is matched and the recv of
// partition 0 has no source and zeroes its buffer.
//
// Not covered: the pipelined form (xla_gpu_enable_pipelined_p2p with a
// decomposed collective-permute cycle) crashes the compiler in
// P2PSchedulePreparation on 2 devices.
TEST_P(AsyncCollectivesAotTest, SendRecvInWhileLoop) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    body {
      param = (u32[], u32[2], u32[2]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = u32[2] get-tuple-element(param), index=1
      acc = u32[2] get-tuple-element(param), index=2
      after_all = token[] after-all()
      data_cpy = u32[2] copy(data)
      send_ctx = (u32[2], u32[], token[]) send(data_cpy, after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=1
      recv_ctx = (u32[2], u32[], token[]) recv(after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=2, control-predecessors={data_cpy}
      send_done = token[] send-done(send_ctx), channel_id=1
      recv_done = (u32[2], token[]) recv-done(recv_ctx), channel_id=2
      received = u32[2] get-tuple-element(recv_done), index=0
      acc_next = u32[2] add(acc, received)
      one = u32[] constant(1)
      i_next = u32[] add(i, one)
      ROOT next = (u32[], u32[2], u32[2]) tuple(i_next, data, acc_next)
    }

    cond {
      param = (u32[], u32[2], u32[2]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      limit = u32[] constant(2)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      pid = u32[] partition-id()
      one = u32[] constant(1)
      pid1 = u32[] add(pid, one)
      data = u32[2] broadcast(pid1), dimensions={}
      zero = u32[] constant(0)
      acc = u32[2] broadcast(zero), dimensions={}
      init = (u32[], u32[2], u32[2]) tuple(zero, data, acc)
      loop = (u32[], u32[2], u32[2]) while(init), condition=cond, body=body
      ROOT out = u32[2] get-tuple-element(loop), index=2
    }
  )hlo";
  constexpr int64_t kNumPartitions = 2;
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/1, kNumPartitions);

  // Passes off: the send/recv pairs are hand-written per partition, as in
  // collective_pipeline_parallelism_test SendRecvLoop2Devices.
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config,
                                     /*run_hlo_passes=*/false));
  // Partition 1 receives partition 0's data {1, 1} in each of 2 iterations.
  LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 0}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 2}, results[1]);
}

// Collectives in both branches of a ConditionalThunk. Two conditionals with
// negated predicates make both branches execute. The predicate comes from an
// all-reduce, so all replicas take the same branch.
TEST_P(AsyncCollectivesAotTest, CollectivesInConditional) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] add(lhs, rhs)
    }

    gather_branch {
      p = u32[2] parameter(0)
      ROOT ag = u32[4] all-gather(p), dimensions={0}, replica_groups={{0,1}}
    }

    permute_branch {
      p = u32[2] parameter(0)
      cp = u32[2] collective-permute(p), source_target_pairs={{0,1},{1,0}}
      ROOT out = u32[4] concatenate(cp, cp), dimensions={0}
    }

    ENTRY main {
      id = u32[] replica-id()
      one = u32[] constant(1)
      id1 = u32[] add(id, one)
      v = u32[2] broadcast(id1), dimensions={}
      id_sum = u32[] all-reduce(id), replica_groups={{0,1}}, to_apply=add_u32
      take_gather = pred[] compare(id_sum, one), direction=EQ
      take_permute = pred[] not(take_gather)
      gathered = u32[4] conditional(take_gather, v, v),
                        true_computation=gather_branch,
                        false_computation=permute_branch
      permuted = u32[4] conditional(take_permute, v, v),
                        true_computation=gather_branch,
                        false_computation=permute_branch
      ROOT out = (u32[4], u32[4]) tuple(gathered, permuted)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  KeepCollectivesAsync(config.mutable_debug_options());

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // id_sum = 0 + 1 = 1, so `gathered` takes the all-gather branch and
  // `permuted` the collective-permute branch. v = {id + 1, id + 1}.
  std::vector<Literal> r0 = results[0].DecomposeTuple();
  std::vector<Literal> r1 = results[1].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2}, r0[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2}, r1[0]);
  // Each replica receives the other's v, concatenated twice.
  LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 2, 2, 2}, r0[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 1, 1}, r1[1]);
}

// Reduce-scatter, all-gather and all-to-all chained inside a WhileThunk body.
TEST_P(AsyncCollectivesAotTest, MoreCollectivesInWhileLoop) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] add(lhs, rhs)
    }

    body {
      param = (u32[], u32[4], u32[4]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = u32[4] get-tuple-element(param), index=1
      base = u32[4] get-tuple-element(param), index=2
      rs = u32[2] reduce-scatter(data), replica_groups={{0,1}}, dimensions={0},
                  to_apply=add_u32
      ag = u32[4] all-gather(rs), replica_groups={{0,1}}, dimensions={0}
      a2a = u32[4] all-to-all(ag), replica_groups={{0,1}}, dimensions={0}
      data_next = u32[4] add(a2a, base)
      one = u32[] constant(1)
      i_next = u32[] add(i, one)
      ROOT next = (u32[], u32[4], u32[4]) tuple(i_next, data_next, base)
    }

    cond {
      param = (u32[], u32[4], u32[4]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      limit = u32[] constant(3)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      id = u32[] replica-id()
      ten = u32[] constant(10)
      id10 = u32[] multiply(id, ten)
      id10_b = u32[4] broadcast(id10), dimensions={}
      io = u32[4] iota(), iota_dimension=0
      base = u32[4] add(io, id10_b)
      zero = u32[] constant(0)
      init = (u32[], u32[4], u32[4]) tuple(zero, base, base)
      loop = (u32[], u32[4], u32[4]) while(init), condition=cond, body=body
      ROOT out = u32[4] get-tuple-element(loop), index=1
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  KeepCollectivesAsync(config.mutable_debug_options());

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // base: r0 = {0, 1, 2, 3}, r1 = {10, 11, 12, 13}. Per iteration with
  // s = data_r0 + data_r1: the all-gather rebuilds s on both replicas, the
  // all-to-all gives r0 {s0, s1, s0, s1} and r1 {s2, s3, s2, s3}, then base
  // is added.
  //   it 1  s = {10, 12, 14, 16}, r0 = {10, 13, 12, 15}, r1 = {24, 27, 26, 29}
  //   it 2  s = {34, 40, 38, 44}, r0 = {34, 41, 36, 43}, r1 = {48, 55, 50, 57}
  //   it 3  s = {82, 96, 86, 100}, r0 = {82, 97, 84, 99},
  //         r1 = {96, 111, 98, 113}
  LiteralTestUtil::ExpectR1Equal<uint32_t>({82, 97, 84, 99}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({96, 111, 98, 113}, results[1]);
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, AsyncCollectivesAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, AsyncCollectivesAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
