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

// AOT compatibility of the non-reducing dense collective thunks on 2 GPUs:
// AllGatherThunk, AllToAllThunk and CollectiveBroadcastThunk.

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
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

constexpr absl::string_view kTargetBase =
    "data_movement_collectives_aot_test_2gpu";
constexpr int64_t kNumReplicas = 2;

// All-gather, array all-to-all (has_split_dimension), tuple all-to-all and
// collective-broadcast with a static and a dynamic root, all in one module.
constexpr absl::string_view kDataMovementHlo = R"hlo(
  HloModule test
  ENTRY main {
    id = u32[] replica-id()
    one = u32[] constant(1)
    id1 = u32[] add(id, one)
    v0 = u32[2] broadcast(id1), dimensions={}
    ag = u32[4] all-gather(v0), dimensions={0}, replica_groups={{0,1}}

    ten = u32[] constant(10)
    base = u32[] multiply(id, ten)
    b4 = u32[4] broadcast(base), dimensions={}
    iota4 = u32[4] iota(), iota_dimension=0
    in4 = u32[4] add(b4, iota4)
    a2a_split = u32[4] all-to-all(in4), dimensions={0}, replica_groups={{0,1}}

    b2 = u32[2] broadcast(base), dimensions={}
    c12 = u32[2] constant({1, 2})
    c34 = u32[2] constant({3, 4})
    t0_in = u32[2] add(b2, c12)
    t1_in = u32[2] add(b2, c34)
    a2a_tuple = (u32[2], u32[2]) all-to-all(t0_in, t1_in),
                                 replica_groups={{0,1}}
    a2a_t0 = u32[2] get-tuple-element(a2a_tuple), index=0
    a2a_t1 = u32[2] get-tuple-element(a2a_tuple), index=1

    cb_static = u32[2] collective-broadcast(t0_in), replica_groups={{1, 0}}
    root_idx = s32[1] constant({1})
    cb_dyn = u32[2] collective-broadcast(t1_in, root_idx),
                    replica_groups={{0, 1}}, has_dynamic_root=true
    ROOT out = (u32[4], u32[4], u32[2], u32[2], u32[2], u32[2])
               tuple(ag, a2a_split, a2a_t0, a2a_t1, cb_static, cb_dyn)
  }
)hlo";

// Expected results of kDataMovementHlo. With base = 10 * id:
//   ag        = {1, 1, 2, 2} on both replicas.
//   a2a_split: r0 = {0, 1, 10, 11}, r1 = {2, 3, 12, 13}.
//   a2a_tuple: r0 receives both t0_in (a2a_t0 = {1, 2}, a2a_t1 = {11, 12}),
//              r1 both t1_in (a2a_t0 = {3, 4}, a2a_t1 = {13, 14}).
//   cb_static and cb_dyn both broadcast from replica 1: {11, 12} and {13, 14}.
void ExpectDataMovementResults(std::vector<Literal> results) {
  std::vector<Literal> r0 = results[0].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2}, r0[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 1, 10, 11}, r0[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 2}, r0[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 12}, r0[3]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 12}, r0[4]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({13, 14}, r0[5]);

  std::vector<Literal> r1 = results[1].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2}, r1[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 3, 12, 13}, r1[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({3, 4}, r1[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({13, 14}, r1[3]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 12}, r1[4]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({13, 14}, r1[5]);
}

class DataMovementCollectivesAotTest : public CollectiveAotTestBase {};

// Default (COLLECTIVES_PRIVATE_MEMORY, NCCL) lowering, kept asynchronous.
TEST_P(DataMovementCollectivesAotTest, AllGatherAllToAllAndBroadcastAsync) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  KeepCollectivesAsync(config.mutable_debug_options());

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kDataMovementHlo, config));
  ExpectDataMovementResults(std::move(results));
}

// AllToAllThunk::p2p_memcpy_enabled: local peer-to-peer memcpy instead of NCCL.
TEST_P(DataMovementCollectivesAotTest, MemcpyLocalP2P) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  config.mutable_debug_options().set_xla_gpu_use_memcpy_local_p2p(true);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kDataMovementHlo, config));
  ExpectDataMovementResults(std::move(results));
}

// COLLECTIVES_SYMMETRIC_MEMORY all-gather and use_symmetric_buffer on every
// collective of the module.
TEST_P(DataMovementCollectivesAotTest, SymmetricMemory) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_all_gather_mode(
      DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY);
  debug_options.set_xla_gpu_experimental_nccl_window_symmetric(true);
  EnableNcclSymmetricBuffers(debug_options);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kDataMovementHlo, config));
  ExpectDataMovementResults(std::move(results));
}

// Serializes collectives_mode = COLLECTIVES_PEER_MEMORY on AllGatherThunk. The
// runtime has no peer-memory all-gather yet and runs NCCL. ag_minor gathers
// along a minor dimension, which the compiler canonicalizes into a major-dim
// all-gather plus a transpose fusion.
TEST_P(DataMovementCollectivesAotTest, AllGatherPeerMemory) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      id = u32[] replica-id()
      ten = u32[] constant(10)
      base_u = u32[] multiply(id, ten)
      base = f32[] convert(base_u)
      b2 = f32[2] broadcast(base), dimensions={}
      iota2 = f32[2] iota(), iota_dimension=0
      in = f32[2] add(b2, iota2)
      ag_major = f32[4] all-gather(in), dimensions={0}, replica_groups={{0,1}}
      in_2d = f32[2,1] reshape(in)
      ag_minor = f32[2,2] all-gather(in_2d), dimensions={1},
                          replica_groups={{0,1}}
      ag_minor_1d = f32[4] reshape(ag_minor)
      ROOT out = (f32[4], f32[4]) tuple(ag_major, ag_minor_1d)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  config.mutable_debug_options().set_xla_gpu_all_gather_mode(
      DebugOptions::COLLECTIVES_PEER_MEMORY);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // in: r0 = {0, 1}, r1 = {10, 11}.
  for (int64_t replica = 0; replica < kNumReplicas; ++replica) {
    std::vector<Literal> outputs = results[replica].DecomposeTuple();
    LiteralTestUtil::ExpectR1Near<float>({0, 1, 10, 11}, outputs[0],
                                         ErrorSpec{1e-5, 1e-5});
    LiteralTestUtil::ExpectR1Near<float>({0, 10, 1, 11}, outputs[1],
                                         ErrorSpec{1e-5, 1e-5});
  }
}

// Element types other than u32/f32, plus a variadic mixed-type all-gather with
// implicit (empty) replica groups, carried over from collective_ops_aot_test.
TEST_P(DataMovementCollectivesAotTest, ElementTypes) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      id = u32[] replica-id()
      one = u32[] constant(1)
      id1 = u32[] add(id, one)
      id1_f = f32[] convert(id1)
      bf16_s = bf16[] convert(id1_f)
      bf16_v = bf16[2] broadcast(bf16_s), dimensions={}
      ag_bf16 = bf16[4] all-gather(bf16_v), dimensions={0},
                        replica_groups={{0,1}}

      ten = u32[] constant(10)
      base = u32[] multiply(id, ten)
      b4 = u32[4] broadcast(base), dimensions={}
      iota4 = u32[4] iota(), iota_dimension=0
      in4 = u32[4] add(b4, iota4)
      in4_s8 = s8[4] convert(in4)
      a2a_s8 = s8[4] all-to-all(in4_s8), dimensions={0}, replica_groups={{0,1}}
      in4_f = f32[4] convert(in4)
      in4_c64 = c64[4] complex(in4_f, in4_f)
      a2a_c64 = c64[4] all-to-all(in4_c64), dimensions={0},
                       replica_groups={{0,1}}

      f16_s = f16[] convert(id1_f)
      f16_v = f16[2] broadcast(f16_s), dimensions={}
      a2a_f16 = (f16[2], f16[2]) all-to-all(f16_v, f16_v),
                                 replica_groups={{0,1}}
      a2a_f16_0 = f16[2] get-tuple-element(a2a_f16), index=0
      a2a_f16_1 = f16[2] get-tuple-element(a2a_f16), index=1

      f64_s = f64[] convert(id1_f)
      f64_v = f64[2] broadcast(f64_s), dimensions={}
      cb_f64 = f64[2] collective-broadcast(f64_v), replica_groups={{1, 0}}

      u32_v = u32[2,1] broadcast(id1), dimensions={}
      f32_v = f32[2,1] convert(u32_v)
      ag_mixed = (u32[2,2], f32[2,2]) all-gather(u32_v, f32_v), dimensions={1}
      ag_mixed_u32_2d = u32[2,2] get-tuple-element(ag_mixed), index=0
      ag_mixed_f32_2d = f32[2,2] get-tuple-element(ag_mixed), index=1
      ag_mixed_u32 = u32[4] reshape(ag_mixed_u32_2d)
      ag_mixed_f32 = f32[4] reshape(ag_mixed_f32_2d)
      ROOT out = (bf16[4], s8[4], c64[4], f16[2], f16[2], f64[2], u32[4],
                  f32[4])
                 tuple(ag_bf16, a2a_s8, a2a_c64, a2a_f16_0, a2a_f16_1, cb_f64,
                       ag_mixed_u32, ag_mixed_f32)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  const bfloat16 bf1(1.0f), bf2(2.0f);
  const half h1(1.0f), h2(2.0f);
  // DecomposeTuple moves the elements out, so decompose each result once.
  std::vector<Literal> r0 = results[0].DecomposeTuple();
  std::vector<Literal> r1 = results[1].DecomposeTuple();
  for (const std::vector<Literal>* outputs : {&r0, &r1}) {
    LiteralTestUtil::ExpectR1Equal<bfloat16>({bf1, bf1, bf2, bf2},
                                             (*outputs)[0]);
    // Output j of the tuple all-to-all is operand <own rank> of replica j, and
    // both operands of replica j hold j + 1.
    LiteralTestUtil::ExpectR1Equal<half>({h1, h1}, (*outputs)[3]);
    LiteralTestUtil::ExpectR1Equal<half>({h2, h2}, (*outputs)[4]);
    // Broadcast from replica 1: id + 1 = 2.
    LiteralTestUtil::ExpectR1Equal<double>({2.0, 2.0}, (*outputs)[5]);
    // Variadic all-gather over all replicas along the minor dimension, as in
    // the legacy AllGatherMixedTypes: each [2, 2] row holds {1, 2}.
    LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 2, 1, 2}, (*outputs)[6]);
    LiteralTestUtil::ExpectR1Near<float>({1, 2, 1, 2}, (*outputs)[7],
                                         ErrorSpec{1e-5, 1e-5});
  }
  LiteralTestUtil::ExpectR1Equal<int8_t>({0, 1, 10, 11}, r0[1]);
  LiteralTestUtil::ExpectR1Equal<complex64>(
      {complex64(0, 0), complex64(1, 1), complex64(10, 10), complex64(11, 11)},
      r0[2]);
  LiteralTestUtil::ExpectR1Equal<int8_t>({2, 3, 12, 13}, r1[1]);
  LiteralTestUtil::ExpectR1Equal<complex64>(
      {complex64(2, 2), complex64(3, 3), complex64(12, 12), complex64(13, 13)},
      r1[2]);
}

// Group modes across 2 partitions of 1 replica: COLLECTIVE_OP_GROUP_MODE_
// FLATTENED_ID and CROSS_REPLICA_AND_PARTITION all-gathers, CROSS_PARTITION
// all-to-all and collective-broadcast.
TEST_P(DataMovementCollectivesAotTest, CrossPartition) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      pid = u32[] partition-id()
      one = u32[] constant(1)
      pid1 = u32[] add(pid, one)
      v = u32[2] broadcast(pid1), dimensions={}
      iota2 = u32[2] iota(), iota_dimension=0
      w = u32[2] add(v, iota2)
      ag_flat = u32[4] all-gather(v), channel_id=1, dimensions={0},
                       replica_groups={{0,1}}, use_global_device_ids=true
      ag_xrp = u32[4] all-gather(v), channel_id=2, dimensions={0},
                      replica_groups={{0}}
      a2a = u32[2] all-to-all(w), channel_id=3, dimensions={0},
                   replica_groups={{0,1}}
      cb = u32[2] collective-broadcast(v), channel_id=4, replica_groups={{1,0}}
      ROOT out = (u32[4], u32[4], u32[2], u32[2]) tuple(ag_flat, ag_xrp, a2a, cb)
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
  // v: p0 = {1, 1}, p1 = {2, 2}; w = v + {0, 1}: p0 = {1, 2}, p1 = {2, 3}.
  std::vector<Literal> p0 = results[0].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2}, p0[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2}, p0[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 2}, p0[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 2}, p0[3]);
  std::vector<Literal> p1 = results[1].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2}, p1[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({1, 1, 2, 2}, p1[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 3}, p1[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 2}, p1[3]);
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, DataMovementCollectivesAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, DataMovementCollectivesAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
