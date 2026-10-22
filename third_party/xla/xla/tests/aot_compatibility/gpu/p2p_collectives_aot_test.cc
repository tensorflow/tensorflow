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

// AOT compatibility of the point-to-point collective thunks on 2 GPUs:
// CollectivePermuteThunk, SendThunk, RecvThunk and CollectiveGroupThunk.

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/service/device_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/aot_compatibility/gpu/collective_aot_test_base.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla.pb.h"

namespace xla {
namespace aot_compatibility {
namespace {

using ::testing::ValuesIn;

constexpr absl::string_view kTargetBase = "p2p_collectives_aot_test_2gpu";
constexpr int64_t kNumReplicas = 2;

// Single-operand, combined multi-operand and one-way (non-cycle) permutes of
// v0 = {id + 1, id + 1} and v1 = 10 * v0. Each permute is a separate output,
// so swapped result buffers of the combined permute (cp_m0, cp_m1) fail; see
// ExpectCollectivePermutes. cp_single and cp_m0 permute the same values.
constexpr absl::string_view kCollectivePermutesHlo = R"hlo(
  HloModule test
  ENTRY main {
    id = u32[] replica-id()
    one = u32[] constant(1)
    ten = u32[] constant(10)
    id1 = u32[] add(id, one)
    v0 = u32[2] broadcast(id1), dimensions={}
    t10 = u32[2] broadcast(ten), dimensions={}
    v1 = u32[2] multiply(v0, t10)

    cp_single = u32[2] collective-permute(v0),
                       source_target_pairs={{1,0},{0,1}}
    cp_multi = (u32[2], u32[2]) collective-permute(v0, v1),
                                source_target_pairs={{1,0},{0,1}}
    cp_m0 = u32[2] get-tuple-element(cp_multi), index=0
    cp_m1 = u32[2] get-tuple-element(cp_multi), index=1
    cp_oneway = u32[2] collective-permute(v0), source_target_pairs={{0,1}}

    ROOT out = (u32[2], u32[2], u32[2], u32[2])
               tuple(cp_single, cp_m0, cp_m1, cp_oneway)
  }
)hlo";

// Checks outputs 0-3 (cp_single, cp_m0, cp_m1, cp_oneway) of the decomposed
// per-device results of kCollectivePermutesHlo or of the same permutes across
// partitions. Each device receives the other's v0 and v1; the one-way permute
// only reaches device 1, so device 0 receives zeros.
void ExpectCollectivePermutes(const std::vector<Literal>& r0,
                              const std::vector<Literal>& r1) {
  LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 2}, r0[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 2}, r0[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({20, 20}, r0[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 0}, r0[3]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1}, r1[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1}, r1[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 10}, r1[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1}, r1[3]);
}

class P2PCollectivesAotTest : public CollectiveAotTestBase {};

// COLLECTIVES_PRIVATE_MEMORY with SourceTargetConnectedComponents, plus the
// degenerate lowerings: empty pairs (MemzeroThunk) and identity pairs (copy).
TEST_P(P2PCollectivesAotTest, CollectivePermuteVariants) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      id = u32[] replica-id()
      one = u32[] constant(1)
      ten = u32[] constant(10)
      id1 = u32[] add(id, one)
      v0 = u32[2] broadcast(id1), dimensions={}
      t10 = u32[2] broadcast(ten), dimensions={}
      v1 = u32[2] multiply(v0, t10)

      cp_single = u32[2] collective-permute(v0),
                         source_target_pairs={{1,0},{0,1}}
      cp_multi = (u32[2], u32[2]) collective-permute(v0, v1),
                                  source_target_pairs={{1,0},{0,1}}
      cp_m0 = u32[2] get-tuple-element(cp_multi), index=0
      cp_m1 = u32[2] get-tuple-element(cp_multi), index=1
      cp_oneway = u32[2] collective-permute(v0), source_target_pairs={{0,1}}
      cp_empty = u32[2] collective-permute(v0), source_target_pairs={}
      cp_id = u32[2] collective-permute(v0), source_target_pairs={{0,0},{1,1}}

      ROOT out = (u32[2], u32[2], u32[2], u32[2], u32[2], u32[2])
                 tuple(cp_single, cp_m0, cp_m1, cp_oneway, cp_empty, cp_id)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_collective_permute_mode(
      DebugOptions::COLLECTIVES_PRIVATE_MEMORY);
  debug_options.set_xla_gpu_collective_permute_connected_components(true);
  debug_options.add_xla_disable_hlo_passes("collective-permute-combiner");

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  std::vector<Literal> r0 = results[0].DecomposeTuple();
  std::vector<Literal> r1 = results[1].DecomposeTuple();
  ExpectCollectivePermutes(r0, r1);
  // cp_empty zeroes its output; cp_id keeps each replica's v0 = id + 1.
  LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 0}, r0[4]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 0}, r1[4]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1}, r0[5]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 2}, r1[5]);
}

// Runs twice: the peer-memory barrier only synchronizes from the second
// execution on.
TEST_P(P2PCollectivesAotTest, CollectivePermutePeerMemory) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_collective_permute_mode(
      DebugOptions::COLLECTIVES_PEER_MEMORY);
  debug_options.add_xla_disable_hlo_passes("collective-permute-combiner");

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<Literal>> runs,
      RunReplicatedRepeatedly(kCollectivePermutesHlo, config, /*num_runs=*/2));
  for (std::vector<Literal>& results : runs) {
    ExpectCollectivePermutes(results[0].DecomposeTuple(),
                             results[1].DecomposeTuple());
  }
}

// COLLECTIVES_SYMMETRIC_MEMORY over NCCL symmetric buffers
// (CollectiveConfig::use_symmetric_buffer). Runs twice for the same reason as
// CollectivePermutePeerMemory.
TEST_P(P2PCollectivesAotTest, CollectivePermuteSymmetricMemory) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_collective_permute_mode(
      DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY);
  debug_options.set_xla_gpu_experimental_nccl_window_symmetric(true);
  EnableNcclSymmetricBuffers(debug_options);
  debug_options.add_xla_disable_hlo_passes("collective-permute-combiner");

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<Literal>> runs,
      RunReplicatedRepeatedly(kCollectivePermutesHlo, config, /*num_runs=*/2));
  for (std::vector<Literal>& results : runs) {
    ExpectCollectivePermutes(results[0].DecomposeTuple(),
                             results[1].DecomposeTuple());
  }
}

// Non-identity RemapSourceTargetToCliqueRanks: a reversed DeviceAssignment
// makes the clique RankId differ from the logical replica ID.
TEST_P(P2PCollectivesAotTest, CollectivePermuteRankRemap) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      id = u32[] replica-id()
      id_f = f32[] convert(id)
      ten = f32[] constant(10.0)
      base = f32[] multiply(id_f, ten)
      b2 = f32[2] broadcast(base), dimensions={}
      c12 = f32[2] constant({1.0, 2.0})
      c34 = f32[2] constant({3.0, 4.0})
      x = f32[2] add(b2, c12)
      y = f32[2] add(b2, c34)
      cp_cycle = f32[2] collective-permute(x), source_target_pairs={{0,1},{1,0}}
      cp_oneway = f32[2] collective-permute(y), source_target_pairs={{0,1}}
      ROOT out = f32[2] add(cp_cycle, cp_oneway)
    }
  )hlo";
  DeviceAssignment device_assignment(kNumReplicas, /*computation_count=*/1);
  device_assignment(0, 0) = 1;
  device_assignment(1, 0) = 0;
  HloModuleConfig config = GetModuleConfigForTest(
      kNumReplicas, /*num_partitions=*/1, device_assignment);
  config.mutable_debug_options()
      .set_xla_gpu_collective_permute_connected_components(true);

  // Passes off: the module runs as written, so the golden holds exactly these
  // two permutes.
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config,
                                     /*run_hlo_passes=*/false));
  // r0: cp_cycle(r1.x = {11, 12}) + cp_oneway(0).
  // r1: cp_cycle(r0.x = {1, 2}) + cp_oneway(r0.y = {3, 4}).
  LiteralTestUtil::ExpectR1Near<float>({11.0, 12.0}, results[0],
                                       ErrorSpec{1e-4, 1e-4});
  LiteralTestUtil::ExpectR1Near<float>({4.0, 6.0}, results[1],
                                       ErrorSpec{1e-4, 1e-4});
}

// Element types other than u32/f32, one permute per type, plus a combined
// permute whose operands have different element types
// (CollectiveConfig::operand_element_type with several entries).
TEST_P(P2PCollectivesAotTest, CollectivePermuteElementTypes) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      id = u32[] replica-id()
      one = u32[] constant(1)
      id1 = u32[] add(id, one)
      id1_f = f32[] convert(id1)
      bf16_s = bf16[] convert(id1_f)
      bf16_v = bf16[2] broadcast(bf16_s), dimensions={}
      f16_s = f16[] convert(id1_f)
      f16_v = f16[2] broadcast(f16_s), dimensions={}
      s8_s = s8[] convert(id1)
      s8_v = s8[2] broadcast(s8_s), dimensions={}
      f64_s = f64[] convert(id1_f)
      f64_v = f64[2] broadcast(f64_s), dimensions={}
      c64_s = c64[] complex(id1_f, id1_f)
      c64_v = c64[2] broadcast(c64_s), dimensions={}
      zero = u32[] constant(0)
      is_r0 = pred[] compare(id, zero), direction=EQ
      pred_v = pred[2] broadcast(is_r0), dimensions={}

      cp_bf16 = bf16[2] collective-permute(bf16_v),
                        source_target_pairs={{0,1},{1,0}}
      cp_s8 = s8[2] collective-permute(s8_v), source_target_pairs={{0,1},{1,0}}
      cp_f64 = f64[2] collective-permute(f64_v),
                      source_target_pairs={{0,1},{1,0}}
      cp_c64 = c64[2] collective-permute(c64_v),
                      source_target_pairs={{0,1},{1,0}}
      cp_pred = pred[2] collective-permute(pred_v),
                        source_target_pairs={{0,1},{1,0}}
      cp_mixed = (f16[2], s8[2]) collective-permute(f16_v, s8_v),
                                 source_target_pairs={{0,1},{1,0}}
      mixed_f16 = f16[2] get-tuple-element(cp_mixed), index=0
      mixed_s8 = s8[2] get-tuple-element(cp_mixed), index=1
      ROOT out = (bf16[2], s8[2], f64[2], c64[2], pred[2], f16[2], s8[2])
                 tuple(cp_bf16, cp_s8, cp_f64, cp_c64, cp_pred, mixed_f16,
                       mixed_s8)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  config.mutable_debug_options().add_xla_disable_hlo_passes(
      "collective-permute-combiner");

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // Each replica receives the other's values: id + 1 and (id == 0).
  for (int64_t replica = 0; replica < kNumReplicas; ++replica) {
    const float peer = static_cast<float>((1 - replica) + 1);
    const bool peer_is_r0 = replica == 1;
    std::vector<Literal> outputs = results[replica].DecomposeTuple();
    LiteralTestUtil::ExpectR1Equal<bfloat16>(
        {static_cast<bfloat16>(peer), static_cast<bfloat16>(peer)}, outputs[0]);
    LiteralTestUtil::ExpectR1Equal<int8_t>(
        {static_cast<int8_t>(peer), static_cast<int8_t>(peer)}, outputs[1]);
    LiteralTestUtil::ExpectR1Equal<double>({peer, peer}, outputs[2]);
    LiteralTestUtil::ExpectR1Equal<complex64>(
        {complex64(peer, peer), complex64(peer, peer)}, outputs[3]);
    LiteralTestUtil::ExpectR1Equal<bool>({peer_is_r0, peer_is_r0}, outputs[4]);
    LiteralTestUtil::ExpectR1Equal<half>(
        {static_cast<half>(peer), static_cast<half>(peer)}, outputs[5]);
    LiteralTestUtil::ExpectR1Equal<int8_t>(
        {static_cast<int8_t>(peer), static_cast<int8_t>(peer)}, outputs[6]);
  }
}

// COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION: permutes with a channel_id across
// 2 partitions of 1 replica.
TEST_P(P2PCollectivesAotTest, CollectivePermuteCrossPartition) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      pid = u32[] partition-id()
      one = u32[] constant(1)
      ten = u32[] constant(10)
      pid1 = u32[] add(pid, one)
      v0 = u32[2] broadcast(pid1), dimensions={}
      t10 = u32[2] broadcast(ten), dimensions={}
      v1 = u32[2] multiply(v0, t10)

      cp_single = u32[2] collective-permute(v0), channel_id=1,
                         source_target_pairs={{1,0},{0,1}}
      cp_multi = (u32[2], u32[2]) collective-permute(v0, v1), channel_id=2,
                                  source_target_pairs={{1,0},{0,1}}
      cp_m0 = u32[2] get-tuple-element(cp_multi), index=0
      cp_m1 = u32[2] get-tuple-element(cp_multi), index=1
      cp_oneway = u32[2] collective-permute(v0), channel_id=3,
                         source_target_pairs={{0,1}}

      ROOT out = (u32[2], u32[2], u32[2], u32[2])
                 tuple(cp_single, cp_m0, cp_m1, cp_oneway)
    }
  )hlo";
  constexpr int64_t kNumPartitions = 2;
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/1, kNumPartitions);

  // Passes off: the HLO is already written per partition, as in the XLA e2e
  // cross-partition tests.
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config,
                                     /*run_hlo_passes=*/false));
  // Same permutes as kCollectivePermutesHlo with partition-id for id.
  ExpectCollectivePermutes(results[0].DecomposeTuple(),
                           results[1].DecomposeTuple());
}

// Concurrent pipelined send/recv streams (_xla_send_recv_pipeline).
TEST_P(P2PCollectivesAotTest, SendRecvConcurrentPipelines) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test, is_scheduled=true

    ENTRY main {
      c0 = u32[] constant(0)
      f0 = f32[2] constant({1.0, 2.0})
      f1 = f32[2] constant({3.0, 4.0})
      id = u32[] replica-id()
      cond = pred[] compare(id, c0), direction=EQ
      cond2 = pred[2] broadcast(cond), dimensions={}
      send_buf0 = f32[2] select(cond2, f0, f1)
      send_buf1 = f32[2] select(cond2, f1, f0)

      tok0 = after-all()
      tok1 = after-all()

      recv0 = (f32[2], u32[], token[]) recv(tok0), channel_id=0,
        frontend_attributes={
          _xla_send_recv_source_target_pairs={{0,1}},
          _xla_send_recv_pipeline="0"
        }
      send0 = (f32[2], u32[], token[]) send(send_buf0, tok0), channel_id=0,
        frontend_attributes={
          _xla_send_recv_source_target_pairs={{0,1}},
          _xla_send_recv_pipeline="0"
        }

      recv1 = (f32[2], u32[], token[]) recv(tok1), channel_id=0,
        frontend_attributes={
          _xla_send_recv_source_target_pairs={{1,0}},
          _xla_send_recv_pipeline="1"
        }
      send1 = (f32[2], u32[], token[]) send(send_buf1, tok1), channel_id=0,
        frontend_attributes={
          _xla_send_recv_source_target_pairs={{1,0}},
          _xla_send_recv_pipeline="1"
        }

      recv_done0 = (f32[2], token[]) recv-done(recv0), channel_id=0,
        frontend_attributes={_xla_send_recv_pipeline="0"}
      send_done0 = token[] send-done(send0), channel_id=0,
        frontend_attributes={_xla_send_recv_pipeline="0"}

      recv_done1 = (f32[2], token[]) recv-done(recv1), channel_id=0,
        frontend_attributes={_xla_send_recv_pipeline="1"}
      send_done1 = token[] send-done(send1), channel_id=0,
        frontend_attributes={_xla_send_recv_pipeline="1"}

      d0 = f32[2] get-tuple-element(recv_done0), index=0
      d1 = f32[2] get-tuple-element(recv_done1), index=0
      ROOT sum = f32[2] add(d0, d1)
    }
  )hlo";
  // Passes off: the module is already scheduled.
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, GetModuleConfigForTest(kNumReplicas),
                                     /*run_hlo_passes=*/false));
  // Pipeline 0 delivers r0.send_buf0 = {1, 2} to r1 and pipeline 1 delivers
  // r1.send_buf1 = {1, 2} to r0; the non-receiving side of each reads zeros.
  LiteralTestUtil::ExpectR1Near<float>({1.0, 2.0}, results[0],
                                       ErrorSpec{1e-5, 1e-5});
  LiteralTestUtil::ExpectR1Near<float>({1.0, 2.0}, results[1],
                                       ErrorSpec{1e-5, 1e-5});
}

// SendThunk and RecvThunk inside an async CollectiveGroupThunk with one-way
// pairs: the non-sender returns early (!target_id) and the non-receiver zeroes
// its output (!source_id).
TEST_P(P2PCollectivesAotTest, GroupedOneWaySendRecv) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    wrapped_send_recv {
      p0 = f32[2] parameter(0)
      t0 = token[] parameter(1)
      send1 = (f32[2], u32[], token[]) send(p0, t0), channel_id=0,
        frontend_attributes={_xla_send_recv_source_target_pairs="{{0,1}}"}
      t1 = token[] parameter(2)
      recv1 = (f32[2], u32[], token[]) recv(t1), channel_id=0,
        frontend_attributes={_xla_send_recv_source_target_pairs="{{0,1}}"}
      ROOT out = ((f32[2], u32[], token[]), (f32[2], u32[], token[]))
                 tuple(send1, recv1)
    }

    ENTRY main {
      data1 = f32[2] constant({10.0, 15.0})
      tok1 = token[] after-all()
      tok2 = token[] after-all()
      grp_start = ((f32[2], token[], token[]),
                   ((f32[2], u32[], token[]), (f32[2], u32[], token[])))
                  async-start(data1, tok1, tok2),
                  calls=wrapped_send_recv,
                  frontend_attributes={_collectives_group=""}
      grp_done = ((f32[2], u32[], token[]), (f32[2], u32[], token[]))
                 async-done(grp_start), calls=wrapped_send_recv,
                 frontend_attributes={_collectives_group=""}
      r1_ctx = (f32[2], u32[], token[]) get-tuple-element(grp_done), index=1
      ROOT r1 = f32[2] get-tuple-element(r1_ctx), index=0
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  KeepCollectivesAsync(config.mutable_debug_options());

  // Passes off: the hand-written _collectives_group computation runs as is.
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config,
                                     /*run_hlo_passes=*/false));
  LiteralTestUtil::ExpectR1Near<float>({0.0, 0.0}, results[0],
                                       ErrorSpec{1e-4, 1e-4});
  LiteralTestUtil::ExpectR1Near<float>({10.0, 15.0}, results[1],
                                       ErrorSpec{1e-4, 1e-4});
}

// CollectiveGroupThunk with 0 (all degenerate), 1 and several communicators.
TEST_P(P2PCollectivesAotTest, CollectiveGroup) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test, is_scheduled=true

    add_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] add(lhs, rhs)
    }

    group_degen {
      p0 = u32[2] parameter(0)
      ar0 = u32[2] all-reduce(p0), replica_groups={{0},{1}},
                       to_apply=add_u32
      ar1 = u32[2] all-reduce(p0), replica_groups={{0},{1}},
                       to_apply=add_u32
      ROOT t = (u32[2], u32[2]) tuple(ar0, ar1)
    }

    group_single_comm {
      p0 = u32[2] parameter(0)
      p1 = u32[2] parameter(1)
      cp0 = u32[2] collective-permute(p0), source_target_pairs={{0,1}}
      cp1 = u32[2] collective-permute(p1), source_target_pairs={{1,0}}
      ROOT t = (u32[2], u32[2]) tuple(cp0, cp1)
    }

    group_multi_comm {
      p0 = u32[2] parameter(0)
      p1 = u32[2] parameter(1)
      ar = u32[2] all-reduce(p0), replica_groups={{0,1}}, to_apply=add_u32
      cp = u32[2] collective-permute(p1), source_target_pairs={{0,1},{1,0}}
      degen = u32[2] all-reduce(p0), replica_groups={{0},{1}}, to_apply=add_u32
      ROOT t = (u32[2], u32[2], u32[2]) tuple(ar, cp, degen)
    }

    ENTRY main {
      id = u32[] replica-id()
      one = u32[] constant(1)
      ten = u32[] constant(10)
      id1 = u32[] add(id, one)
      v0 = u32[2] broadcast(id1), dimensions={}
      t10 = u32[2] broadcast(ten), dimensions={}
      v1 = u32[2] multiply(v0, t10)

      s_degen = ((u32[2]), (u32[2], u32[2])) async-start(v0), calls=group_degen,
        frontend_attributes={_collectives_group=""}
      d_degen = (u32[2], u32[2]) async-done(s_degen), calls=group_degen,
        frontend_attributes={_collectives_group=""}
      dg0 = u32[2] get-tuple-element(d_degen), index=0
      dg1 = u32[2] get-tuple-element(d_degen), index=1

      s_single = ((u32[2], u32[2]), (u32[2], u32[2])) async-start(v0, v1),
        calls=group_single_comm, frontend_attributes={_collectives_group=""}
      d_single = (u32[2], u32[2]) async-done(s_single),
        calls=group_single_comm, frontend_attributes={_collectives_group=""}
      sc0 = u32[2] get-tuple-element(d_single), index=0
      sc1 = u32[2] get-tuple-element(d_single), index=1

      s_multi = ((u32[2], u32[2]), (u32[2], u32[2], u32[2]))
        async-start(v0, v1), calls=group_multi_comm,
        frontend_attributes={_collectives_group=""}
      d_multi = (u32[2], u32[2], u32[2]) async-done(s_multi),
        calls=group_multi_comm, frontend_attributes={_collectives_group=""}
      mc0 = u32[2] get-tuple-element(d_multi), index=0
      mc1 = u32[2] get-tuple-element(d_multi), index=1
      mc2 = u32[2] get-tuple-element(d_multi), index=2

      sum_dg = u32[2] add(dg0, dg1)
      sum0 = u32[2] add(sum_dg, sc0)
      sum1 = u32[2] add(sum0, sc1)
      sum2 = u32[2] add(sum1, mc0)
      sum3 = u32[2] add(sum2, mc1)
      ROOT out = u32[2] add(sum3, mc2)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  KeepCollectivesAsync(config.mutable_debug_options());

  // Passes off: the module is already scheduled.
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config,
                                     /*run_hlo_passes=*/false));
  // r0: degen(1 + 1) + single(0 + 20) + multi(ar 3 + cp 20 + degen 1) = 46.
  // r1: degen(2 + 2) + single(1 + 0) + multi(ar 3 + cp 10 + degen 2) = 20.
  LiteralTestUtil::ExpectR1Equal<uint32_t>({46, 46}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({20, 20}, results[1]);
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, P2PCollectivesAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, P2PCollectivesAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
