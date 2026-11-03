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

// AOT compatibility of the reducing collective thunks on 2 GPUs:
// AllReduceThunk, ReduceScatterThunk, CollectiveReduceThunk and
// PartitionIdThunk.

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
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

constexpr absl::string_view kTargetBase = "reduction_collectives_aot_test_2gpu";
constexpr int64_t kNumReplicas = 2;

class ReductionCollectivesAotTest : public CollectiveAotTestBase {
 protected:
  // Lowers all-reduce through NCCL in every case. CollectiveKernelThunk has its
  // own target, collective_kernels_aot_test.
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        CollectiveAotTestBase::GetDebugOptionsForTest();
    DisableCollectiveKernels(debug_options);
    return debug_options;
  }
};

// Combined (tuple) all-reduce, every u32 reduction kind, a FLATTENED_ID
// all-reduce and a reduce-scatter, kept asynchronous and uncombined.
TEST_P(ReductionCollectivesAotTest, AllReduceAndReduceScatterAsync) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] add(lhs, rhs)
    }

    mul_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] multiply(lhs, rhs)
    }

    min_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] minimum(lhs, rhs)
    }

    max_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] maximum(lhs, rhs)
    }

    ENTRY main {
      id = u32[] replica-id()
      two = u32[] constant(2)
      v0 = u32[] add(id, two)
      vec = u32[2] broadcast(v0), dimensions={}
      vec_two = u32[2] broadcast(two), dimensions={}
      vec2 = u32[2] multiply(vec, vec_two)

      ar_sum = (u32[2], u32[2]) all-reduce(vec, vec2), replica_groups={{0,1}},
                                to_apply=add_u32
      ar_sum0 = u32[2] get-tuple-element(ar_sum), index=0
      ar_sum1 = u32[2] get-tuple-element(ar_sum), index=1

      ar_prod = u32[2] all-reduce(vec), channel_id=1,
                       use_global_device_ids=true, replica_groups={{0,1}},
                       to_apply=mul_u32
      ar_min = u32[2] all-reduce(vec), replica_groups={{0,1}}, to_apply=min_u32
      ar_max = u32[2] all-reduce(vec), replica_groups={{0,1}}, to_apply=max_u32

      c0 = u32[4] constant({10, 2, 30, 4})
      c1 = u32[4] constant({5, 20, 15, 40})
      zero = u32[] constant(0)
      is_r0 = pred[] compare(id, zero), direction=EQ
      pb = pred[4] broadcast(is_r0), dimensions={}
      rs_in = u32[4] select(pb, c0, c1)
      rs_max = u32[2] reduce-scatter(rs_in), replica_groups={{0,1}},
                      dimensions={0}, to_apply=max_u32

      ROOT out = (u32[2], u32[2], u32[2], u32[2], u32[2], u32[2])
                 tuple(ar_sum0, ar_sum1, ar_prod, ar_min, ar_max, rs_max)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  KeepCollectivesAsync(debug_options);
  debug_options.add_xla_disable_hlo_passes("gpu-all-reduce-combiner");
  debug_options.add_xla_disable_hlo_passes("gpu-reduce-scatter-combiner");
  debug_options.add_xla_disable_hlo_passes("all-reduce-contiguous");

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // vec: r0 = 2, r1 = 3, so sum (5), product (6), min (2) and max (3) all
  // differ and a swapped reduction kind fails. vec2 = 2 * vec.
  std::vector<Literal> r0 = results[0].DecomposeTuple();
  std::vector<Literal> r1 = results[1].DecomposeTuple();
  for (const std::vector<Literal>* outputs : {&r0, &r1}) {
    LiteralTestUtil::ExpectR1Equal<uint32_t>({5, 5}, (*outputs)[0]);
    LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 10}, (*outputs)[1]);
    LiteralTestUtil::ExpectR1Equal<uint32_t>({6, 6}, (*outputs)[2]);
    LiteralTestUtil::ExpectR1Equal<uint32_t>({2, 2}, (*outputs)[3]);
    LiteralTestUtil::ExpectR1Equal<uint32_t>({3, 3}, (*outputs)[4]);
  }
  // rs_max = max(c0, c1) = {10, 20 | 30, 40}.
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 20}, r0[5]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({30, 40}, r1[5]);
}

// Element types and reduction kinds beyond u32, one all-reduce or
// reduce-scatter each, plus a u32 reduce-scatter with implicit (empty) replica
// groups, carried over from collective_ops_aot_test.
TEST_P(ReductionCollectivesAotTest, ElementTypes) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] add(lhs, rhs)
    }

    add_bf16 {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT res = bf16[] add(lhs, rhs)
    }

    max_f16 {
      lhs = f16[] parameter(0)
      rhs = f16[] parameter(1)
      ROOT res = f16[] maximum(lhs, rhs)
    }

    min_s8 {
      lhs = s8[] parameter(0)
      rhs = s8[] parameter(1)
      ROOT res = s8[] minimum(lhs, rhs)
    }

    mul_u8 {
      lhs = u8[] parameter(0)
      rhs = u8[] parameter(1)
      ROOT res = u8[] multiply(lhs, rhs)
    }

    add_s64 {
      lhs = s64[] parameter(0)
      rhs = s64[] parameter(1)
      ROOT res = s64[] add(lhs, rhs)
    }

    add_f64 {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT res = f64[] add(lhs, rhs)
    }

    or_pred {
      lhs = pred[] parameter(0)
      rhs = pred[] parameter(1)
      ROOT res = pred[] or(lhs, rhs)
    }

    and_pred {
      lhs = pred[] parameter(0)
      rhs = pred[] parameter(1)
      ROOT res = pred[] and(lhs, rhs)
    }

    add_c64 {
      lhs = c64[] parameter(0)
      rhs = c64[] parameter(1)
      ROOT res = c64[] add(lhs, rhs)
    }

    ENTRY main {
      id = u32[] replica-id()
      one = u32[] constant(1)
      id1 = u32[] add(id, one)
      id1_f = f32[] convert(id1)

      bf16_s = bf16[] convert(id1_f)
      bf16_v = bf16[2] broadcast(bf16_s), dimensions={}
      ar_bf16 = bf16[2] all-reduce(bf16_v), replica_groups={{0,1}},
                        to_apply=add_bf16

      one_f = f32[] constant(1)
      id2_f = f32[] add(id1_f, one_f)
      f16_s = f16[] convert(id2_f)
      f16_v = f16[2] broadcast(f16_s), dimensions={}
      ar_f16 = f16[2] all-reduce(f16_v), replica_groups={{0,1}},
                      to_apply=max_f16

      s8_s = s8[] convert(id1)
      s8_neg = s8[] negate(s8_s)
      s8_v = s8[2] broadcast(s8_neg), dimensions={}
      ar_s8 = s8[2] all-reduce(s8_v), replica_groups={{0,1}}, to_apply=min_s8

      three = u32[] constant(3)
      u8_base = u32[] add(id1, three)
      u8_s = u8[] convert(u8_base)
      u8_v = u8[2] broadcast(u8_s), dimensions={}
      ar_u8 = u8[2] all-reduce(u8_v), replica_groups={{0,1}}, to_apply=mul_u8

      s64_s = s64[] convert(id1)
      two_pow_32 = s64[] constant(4294967296)
      s64_big = s64[] multiply(s64_s, two_pow_32)
      s64_v = s64[2] broadcast(s64_big), dimensions={}
      ar_s64 = s64[2] all-reduce(s64_v), replica_groups={{0,1}},
                      to_apply=add_s64

      f64_s = f64[] convert(id1_f)
      f64_v = f64[2] broadcast(f64_s), dimensions={}
      ar_f64 = f64[2] all-reduce(f64_v), replica_groups={{0,1}},
                      to_apply=add_f64

      zero = u32[] constant(0)
      is_r0 = pred[] compare(id, zero), direction=EQ
      pred_v = pred[2] broadcast(is_r0), dimensions={}
      ar_or = pred[2] all-reduce(pred_v), replica_groups={{0,1}},
                      to_apply=or_pred
      ar_and = pred[2] all-reduce(pred_v), replica_groups={{0,1}},
                       to_apply=and_pred

      neg_f = f32[] negate(id1_f)
      c64_s = c64[] complex(id1_f, neg_f)
      c64_v = c64[2] broadcast(c64_s), dimensions={}
      ar_c64 = c64[2] all-reduce(c64_v), replica_groups={{0,1}},
                      to_apply=add_c64

      iota4 = f32[4] iota(), iota_dimension=0
      b4 = f32[4] broadcast(id1_f), dimensions={}
      rs_in_f = f32[4] multiply(iota4, b4)
      rs_in = bf16[4] convert(rs_in_f)
      rs_bf16 = bf16[2] reduce-scatter(rs_in), replica_groups={{0,1}},
                        dimensions={0}, to_apply=add_bf16

      iota4_u = u32[4] iota(), iota_dimension=0
      b4_u = u32[4] broadcast(id1), dimensions={}
      rs_all_in = u32[4] multiply(iota4_u, b4_u)
      rs_all = u32[2] reduce-scatter(rs_all_in), replica_groups={},
                      dimensions={0}, to_apply=add_u32

      ROOT out = (bf16[2], f16[2], s8[2], u8[2], s64[2], f64[2], pred[2],
                  pred[2], c64[2], bf16[2], u32[2])
                 tuple(ar_bf16, ar_f16, ar_s8, ar_u8, ar_s64, ar_f64, ar_or,
                       ar_and, ar_c64, rs_bf16, rs_all)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // DecomposeTuple moves the elements out, so decompose each result once.
  std::vector<Literal> r0 = results[0].DecomposeTuple();
  std::vector<Literal> r1 = results[1].DecomposeTuple();
  // Per replica id1 = id + 1: r0 = 1, r1 = 2.
  for (const std::vector<Literal>* outputs : {&r0, &r1}) {
    LiteralTestUtil::ExpectR1Equal<bfloat16>({bfloat16(3.0f), bfloat16(3.0f)},
                                             (*outputs)[0]);
    // max(2, 3): the f16 input is id1 + 1, so max differs from the product.
    LiteralTestUtil::ExpectR1Equal<half>({half(3.0f), half(3.0f)},
                                         (*outputs)[1]);
    LiteralTestUtil::ExpectR1Equal<int8_t>({-2, -2}, (*outputs)[2]);
    // (1 + 3) * (2 + 3) = 20.
    LiteralTestUtil::ExpectR1Equal<uint8_t>({20, 20}, (*outputs)[3]);
    // (1 + 2) * 2^32: exercises the upper 32 bits.
    LiteralTestUtil::ExpectR1Equal<int64_t>({12884901888, 12884901888},
                                            (*outputs)[4]);
    LiteralTestUtil::ExpectR1Equal<double>({3.0, 3.0}, (*outputs)[5]);
    // Only replica 0 holds true.
    LiteralTestUtil::ExpectR1Equal<bool>({true, true}, (*outputs)[6]);
    LiteralTestUtil::ExpectR1Equal<bool>({false, false}, (*outputs)[7]);
    LiteralTestUtil::ExpectR1Equal<complex64>(
        {complex64(3, -3), complex64(3, -3)}, (*outputs)[8]);
  }
  // rs_in = iota * id1: r0 = {0, 1, 2, 3}, r1 = {0, 2, 4, 6}; sum =
  // {0, 3, 6, 9}, scattered in halves.
  LiteralTestUtil::ExpectR1Equal<bfloat16>({bfloat16(0.0f), bfloat16(3.0f)},
                                           r0[9]);
  LiteralTestUtil::ExpectR1Equal<bfloat16>({bfloat16(6.0f), bfloat16(9.0f)},
                                           r1[9]);
  // rs_all is the same reduction in u32 over all replicas (empty groups).
  LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 3}, r0[10]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({6, 9}, r1[10]);
}

// CollectiveReduceThunk with a static root (first group member) and a dynamic
// root operand, kept asynchronous. Only the root's output is defined.
TEST_P(ReductionCollectivesAotTest, CollectiveReduce) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] add(lhs, rhs)
    }

    max_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT res = f32[] maximum(lhs, rhs)
    }

    ENTRY main {
      id = u32[] replica-id()
      one = u32[] constant(1)
      val = u32[] add(id, one)
      v = u32[2] broadcast(val), dimensions={}
      cr_static = u32[2] collective-reduce(v), replica_groups={{0,1}},
                         to_apply=add_u32

      val_f = f32[] convert(val)
      vf = f32[2] broadcast(val_f), dimensions={}
      root = s32[1] constant({1})
      cr_dynamic = f32[2] collective-reduce(vf, root), replica_groups={{0,1}},
                          to_apply=max_f32, has_dynamic_root=true
      ROOT out = (u32[2], f32[2]) tuple(cr_static, cr_dynamic)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_experimental_emit_collective_reduce(true);
  KeepCollectivesAsync(debug_options);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // Static root is replica 0 and receives 1 + 2; the dynamic root is rank 1
  // and receives max(1, 2).
  LiteralTestUtil::ExpectR1Equal<uint32_t>({3, 3},
                                           results[0].DecomposeTuple()[0]);
  LiteralTestUtil::ExpectR1Equal<float>({2.0f, 2.0f},
                                        results[1].DecomposeTuple()[1]);
}

// use_symmetric_buffer on all-reduce and reduce-scatter.
TEST_P(ReductionCollectivesAotTest, SymmetricBuffers) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] add(lhs, rhs)
    }

    ENTRY main {
      id = u32[] replica-id()
      ten = u32[] constant(10)
      base = u32[] multiply(id, ten)
      b4 = u32[4] broadcast(base), dimensions={}
      iota4 = u32[4] iota(), iota_dimension=0
      in = u32[4] add(b4, iota4)
      ar = u32[4] all-reduce(in), replica_groups={{0,1}}, to_apply=add_u32
      rs = u32[2] reduce-scatter(in), replica_groups={{0,1}}, dimensions={0},
                  to_apply=add_u32
      ROOT out = (u32[4], u32[2]) tuple(ar, rs)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  EnableNcclSymmetricBuffers(config.mutable_debug_options());

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // in: r0 = {0, 1, 2, 3}, r1 = {10, 11, 12, 13}; sum = {10, 12, 14, 16}.
  std::vector<Literal> r0 = results[0].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 12, 14, 16}, r0[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 12}, r0[1]);
  std::vector<Literal> r1 = results[1].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 12, 14, 16}, r1[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({14, 16}, r1[1]);
}

// PartitionIdThunk and the CROSS_REPLICA_AND_PARTITION and FLATTENED_ID group
// modes of all-reduce and reduce-scatter across 2 partitions of 1 replica.
TEST_P(ReductionCollectivesAotTest, CrossPartition) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] add(lhs, rhs)
    }

    max_u32 {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT res = u32[] maximum(lhs, rhs)
    }

    ENTRY main {
      pid = u32[] partition-id()
      one = u32[] constant(1)
      pid1 = u32[] add(pid, one)
      v = u32[2] broadcast(pid1), dimensions={}
      v2 = u32[2] add(v, v)
      ar_xrp = u32[2] all-reduce(v), channel_id=1, replica_groups={{0}},
                      to_apply=add_u32
      ar_flat = u32[2] all-reduce(v2), channel_id=2, replica_groups={{0,1}},
                       use_global_device_ids=true, to_apply=max_u32

      ten = u32[] constant(10)
      base = u32[] multiply(pid, ten)
      b4 = u32[4] broadcast(base), dimensions={}
      iota4 = u32[4] iota(), iota_dimension=0
      in = u32[4] add(b4, iota4)
      rs_xrp = u32[2] reduce-scatter(in), channel_id=3, replica_groups={{0}},
                      dimensions={0}, to_apply=add_u32
      rs_flat = u32[2] reduce-scatter(in), channel_id=4, replica_groups={{1,0}},
                       use_global_device_ids=true, dimensions={0},
                       to_apply=add_u32
      ROOT out = (u32[2], u32[2], u32[2], u32[2])
                 tuple(ar_xrp, ar_flat, rs_xrp, rs_flat)
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
  // v: p0 = 1, p1 = 2; v2 = 2 * v, so ar_flat's max (4) differs from the
  // product. in: p0 = {0, 1, 2, 3}, p1 = {10, 11, 12, 13}, so the
  // reduce-scatter sum is {10, 12, 14, 16}. rs_flat lists device 1 first, so
  // partition 1 is rank 0 and receives the first half.
  std::vector<Literal> p0 = results[0].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({3, 3}, p0[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({4, 4}, p0[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 12}, p0[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({14, 16}, p0[3]);
  std::vector<Literal> p1 = results[1].DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<uint32_t>({3, 3}, p1[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({4, 4}, p1[1]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({14, 16}, p1[2]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 12}, p1[3]);
}

// Element types beyond ElementTypes: c128 (two f64 halves in NCCL), u64 with
// the upper 32 bits set, and the 8-bit floats NCCL reduces natively on
// Hopper and newer.
TEST_P(ReductionCollectivesAotTest, MoreElementTypes) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_c128 {
      lhs = c128[] parameter(0)
      rhs = c128[] parameter(1)
      ROOT res = c128[] add(lhs, rhs)
    }

    add_u64 {
      lhs = u64[] parameter(0)
      rhs = u64[] parameter(1)
      ROOT res = u64[] add(lhs, rhs)
    }

    add_f8e4m3fn {
      lhs = f8e4m3fn[] parameter(0)
      rhs = f8e4m3fn[] parameter(1)
      ROOT res = f8e4m3fn[] add(lhs, rhs)
    }

    min_f8e5m2 {
      lhs = f8e5m2[] parameter(0)
      rhs = f8e5m2[] parameter(1)
      ROOT res = f8e5m2[] minimum(lhs, rhs)
    }

    ENTRY main {
      id = u32[] replica-id()
      one = u32[] constant(1)
      id1 = u32[] add(id, one)
      id1_f = f32[] convert(id1)

      id1_d = f64[] convert(id1_f)
      neg_d = f64[] negate(id1_d)
      c128_s = c128[] complex(id1_d, neg_d)
      c128_v = c128[2] broadcast(c128_s), dimensions={}
      ar_c128 = c128[2] all-reduce(c128_v), replica_groups={{0,1}},
                        to_apply=add_c128

      u64_s = u64[] convert(id1)
      two_pow_40 = u64[] constant(1099511627776)
      u64_big = u64[] multiply(u64_s, two_pow_40)
      u64_v = u64[2] broadcast(u64_big), dimensions={}
      ar_u64 = u64[2] all-reduce(u64_v), replica_groups={{0,1}},
                      to_apply=add_u64

      e4m3_s = f8e4m3fn[] convert(id1_f)
      e4m3_v = f8e4m3fn[4] broadcast(e4m3_s), dimensions={}
      ar_e4m3 = f8e4m3fn[4] all-reduce(e4m3_v), replica_groups={{0,1}},
                            to_apply=add_f8e4m3fn

      e5m2_s = f8e5m2[] convert(id1_f)
      e5m2_v = f8e5m2[4] broadcast(e5m2_s), dimensions={}
      ar_e5m2 = f8e5m2[4] all-reduce(e5m2_v), replica_groups={{0,1}},
                          to_apply=min_f8e5m2

      ROOT out = (c128[2], u64[2], f8e4m3fn[4], f8e5m2[4])
                 tuple(ar_c128, ar_u64, ar_e4m3, ar_e5m2)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // Per replica id1 = id + 1: r0 = 1, r1 = 2.
  std::vector<Literal> r0 = results[0].DecomposeTuple();
  std::vector<Literal> r1 = results[1].DecomposeTuple();
  for (const std::vector<Literal>* outputs : {&r0, &r1}) {
    LiteralTestUtil::ExpectR1Equal<complex128>(
        {complex128(3, -3), complex128(3, -3)}, (*outputs)[0]);
    // (1 + 2) * 2^40: exercises the upper 32 bits.
    LiteralTestUtil::ExpectR1Equal<uint64_t>({3298534883328, 3298534883328},
                                             (*outputs)[1]);
    const float8_e4m3fn e4m3_3(3.0f);
    LiteralTestUtil::ExpectR1Equal<float8_e4m3fn>(
        {e4m3_3, e4m3_3, e4m3_3, e4m3_3}, (*outputs)[2]);
    // min(1, 2), distinct from the sum, product and max.
    const float8_e5m2 e5m2_1(1.0f);
    LiteralTestUtil::ExpectR1Equal<float8_e5m2>(
        {e5m2_1, e5m2_1, e5m2_1, e5m2_1}, (*outputs)[3]);
  }
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, ReductionCollectivesAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, ReductionCollectivesAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
