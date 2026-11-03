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

// AOT compatibility of CollectiveKernelThunk (Triton one-shot and two-shot
// collective kernels) on 2 GPUs.

#include <cstdint>
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

constexpr absl::string_view kTargetBase = "collective_kernels_aot_test_2gpu";
constexpr int64_t kNumReplicas = 2;

class CollectiveKernelsAotTest : public CollectiveAotTestBase {
 protected:
  // Enables the Triton kernels for every collective type that has one. Ops the
  // kernels do not support silently fall back to NCCL, so the goldens must be
  // checked for collective_kernel_thunk when a case is added.
  static HloModuleConfig CollectiveKernelsConfig(HloModuleConfig config) {
    DebugOptions& debug_options = config.mutable_debug_options();
    debug_options.add_xla_gpu_experimental_use_collective_kernels(
        DebugOptions::COLLECTIVE_KERNEL_ALL_REDUCE);
    debug_options.add_xla_gpu_experimental_use_collective_kernels(
        DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER);
    debug_options.add_xla_gpu_experimental_use_collective_kernels(
        DebugOptions::COLLECTIVE_KERNEL_REDUCE_SCATTER);
    debug_options.add_xla_gpu_experimental_use_collective_kernels(
        DebugOptions::COLLECTIVE_KERNEL_ALL_TO_ALL);
    EnableNcclSymmetricBuffers(debug_options);
    debug_options.add_xla_disable_hlo_passes("algsimp");
    debug_options.add_xla_disable_hlo_passes("all-reduce-reassociate");
    debug_options.add_xla_disable_hlo_passes("gpu-all-reduce-combiner");
    debug_options.add_xla_disable_hlo_passes("gpu-reduce-scatter-combiner");
    debug_options.add_xla_disable_hlo_passes("all-reduce-contiguous");
    return config;
  }
};

// One-shot (512 B) and two-shot (2 MiB) f32 all-reduce and an all-gather. The
// all-reduce goes two-shot once input bytes times the number of ranks exceed
// 2 MiB.
TEST_P(CollectiveKernelsAotTest, AllReduceAndAllGather) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT res = f32[] add(lhs, rhs)
    }

    ENTRY main {
      id = u32[] replica-id()
      id_f = f32[] convert(id)
      one = f32[] constant(1.0)
      val = f32[] add(id_f, one)

      in_small = f32[128] broadcast(val), dimensions={}
      ar_oneshot = f32[128] all-reduce(in_small), replica_groups={{0,1}},
                            to_apply=add_f32
      ag_oneshot = f32[256] all-gather(in_small), dimensions={0},
                            replica_groups={{0,1}}

      in_large = f32[524288] broadcast(val), dimensions={}
      ar_twoshot = f32[524288] all-reduce(in_large), replica_groups={{0,1}},
                               to_apply=add_f32

      s_ar1 = f32[2] slice(ar_oneshot), slice={[0:2]}
      // One element from the half each rank reduces in the two-shot kernel.
      s_ar2_r0 = f32[1] slice(ar_twoshot), slice={[0:1]}
      s_ar2_r1 = f32[1] slice(ar_twoshot), slice={[524287:524288]}
      s_ar2 = f32[2] concatenate(s_ar2_r0, s_ar2_r1), dimensions={0}
      s_ag0 = f32[1] slice(ag_oneshot), slice={[0:1]}
      s_ag1 = f32[1] slice(ag_oneshot), slice={[128:129]}
      s_ag = f32[2] concatenate(s_ag0, s_ag1), dimensions={0}

      sum0 = f32[2] add(s_ar1, s_ar2)
      ROOT out = f32[2] add(sum0, s_ag)
    }
  )hlo";
  HloModuleConfig config =
      CollectiveKernelsConfig(GetModuleConfigForTest(kNumReplicas));

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // val: r0 = 1, r1 = 2. Both all-reduces give 3; the all-gather slices pick
  // r0's and r1's value: {3 + 3 + 1, 3 + 3 + 2}.
  for (int64_t replica = 0; replica < kNumReplicas; ++replica) {
    LiteralTestUtil::ExpectR1Near<float>({7.0, 8.0}, results[replica],
                                         ErrorSpec{1e-4, 1e-4});
  }
}

// Triton one-shot reduce-scatter.
TEST_P(CollectiveKernelsAotTest, ReduceScatter) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT res = f32[] add(lhs, rhs)
    }

    ENTRY main {
      id = u32[] replica-id()
      id_f = f32[] convert(id)
      thousand = f32[] constant(1000)
      base = f32[] multiply(id_f, thousand)
      b = f32[256] broadcast(base), dimensions={}
      io = f32[256] iota(), iota_dimension=0
      in = f32[256] add(b, io)
      rs = f32[128] reduce-scatter(in), dimensions={0}, replica_groups={{0,1}},
                    to_apply=add_f32
      ROOT out = f32[2] slice(rs), slice={[0:2]}
    }
  )hlo";
  HloModuleConfig config =
      CollectiveKernelsConfig(GetModuleConfigForTest(kNumReplicas));

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // in: r0 = iota, r1 = 1000 + iota; sum[i] = 1000 + 2 * i. Replica r gets
  // elements [128 * r, 128 * r + 128).
  LiteralTestUtil::ExpectR1Near<float>({1000, 1002}, results[0],
                                       ErrorSpec{1e-4, 1e-4});
  LiteralTestUtil::ExpectR1Near<float>({1256, 1258}, results[1],
                                       ErrorSpec{1e-4, 1e-4});
}

// Element types of the all-reduce kernel other than f32.
TEST_P(CollectiveKernelsAotTest, AllReduceElementTypes) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_bf16 {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT res = bf16[] add(lhs, rhs)
    }

    add_s32 {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT res = s32[] add(lhs, rhs)
    }

    ENTRY main {
      id = u32[] replica-id()
      one = u32[] constant(1)
      val = u32[] add(id, one)
      val_s = s32[] convert(val)
      val_bf16 = bf16[] convert(val)
      in_bf16 = bf16[256] broadcast(val_bf16), dimensions={}
      ar_bf16 = bf16[256] all-reduce(in_bf16), replica_groups={{0,1}},
                          to_apply=add_bf16
      in_s32 = s32[128] broadcast(val_s), dimensions={}
      ar_s32 = s32[128] all-reduce(in_s32), replica_groups={{0,1}},
                        to_apply=add_s32
      s_bf16 = bf16[2] slice(ar_bf16), slice={[0:2]}
      s_s32 = s32[2] slice(ar_s32), slice={[0:2]}
      ROOT out = (bf16[2], s32[2]) tuple(s_bf16, s_s32)
    }
  )hlo";
  HloModuleConfig config =
      CollectiveKernelsConfig(GetModuleConfigForTest(kNumReplicas));

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  for (int64_t replica = 0; replica < kNumReplicas; ++replica) {
    std::vector<Literal> outputs = results[replica].DecomposeTuple();
    LiteralTestUtil::ExpectR1Equal<bfloat16>({bfloat16(3.0f), bfloat16(3.0f)},
                                             outputs[0]);
    LiteralTestUtil::ExpectR1Equal<int32_t>({3, 3}, outputs[1]);
  }
}

// Kernel element types and reduction kinds beyond add on f32/bf16/s32:
// f64 multiply, s64 minimum (upper 32 bits set), f32 maximum and pred
// or/and. The inputs make sum, product, min and max all differ.
TEST_P(CollectiveKernelsAotTest, AllReduceTypesAndReductions) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    mul_f64 {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT res = f64[] multiply(lhs, rhs)
    }

    min_s64 {
      lhs = s64[] parameter(0)
      rhs = s64[] parameter(1)
      ROOT res = s64[] minimum(lhs, rhs)
    }

    max_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT res = f32[] maximum(lhs, rhs)
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

    ENTRY main {
      id = u32[] replica-id()
      two = u32[] constant(2)
      val = u32[] add(id, two)

      val_f64 = f64[] convert(val)
      in_f64 = f64[128] broadcast(val_f64), dimensions={}
      ar_f64 = f64[128] all-reduce(in_f64), replica_groups={{0,1}},
                        to_apply=mul_f64

      val_s64 = s64[] convert(val)
      two_pow_32 = s64[] constant(4294967296)
      big_s64 = s64[] multiply(val_s64, two_pow_32)
      in_s64 = s64[128] broadcast(big_s64), dimensions={}
      ar_s64 = s64[128] all-reduce(in_s64), replica_groups={{0,1}},
                        to_apply=min_s64

      val_f32 = f32[] convert(val)
      in_f32 = f32[128] broadcast(val_f32), dimensions={}
      ar_f32 = f32[128] all-reduce(in_f32), replica_groups={{0,1}},
                        to_apply=max_f32

      zero = u32[] constant(0)
      is_r0 = pred[] compare(id, zero), direction=EQ
      in_pred = pred[128] broadcast(is_r0), dimensions={}
      ar_or = pred[128] all-reduce(in_pred), replica_groups={{0,1}},
                        to_apply=or_pred
      ar_and = pred[128] all-reduce(in_pred), replica_groups={{0,1}},
                         to_apply=and_pred

      s_f64 = f64[2] slice(ar_f64), slice={[0:2]}
      s_s64 = s64[2] slice(ar_s64), slice={[0:2]}
      s_f32 = f32[2] slice(ar_f32), slice={[0:2]}
      s_or = pred[2] slice(ar_or), slice={[0:2]}
      s_and = pred[2] slice(ar_and), slice={[0:2]}
      ROOT out = (f64[2], s64[2], f32[2], pred[2], pred[2])
                 tuple(s_f64, s_s64, s_f32, s_or, s_and)
    }
  )hlo";
  HloModuleConfig config =
      CollectiveKernelsConfig(GetModuleConfigForTest(kNumReplicas));

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // val: r0 = 2, r1 = 3.
  for (int64_t replica = 0; replica < kNumReplicas; ++replica) {
    std::vector<Literal> outputs = results[replica].DecomposeTuple();
    LiteralTestUtil::ExpectR1Equal<double>({6.0, 6.0}, outputs[0]);
    // min(2, 3) * 2^32.
    LiteralTestUtil::ExpectR1Equal<int64_t>({8589934592, 8589934592},
                                            outputs[1]);
    LiteralTestUtil::ExpectR1Equal<float>({3.0f, 3.0f}, outputs[2]);
    // Only replica 0 holds true.
    LiteralTestUtil::ExpectR1Equal<bool>({true, true}, outputs[3]);
    LiteralTestUtil::ExpectR1Equal<bool>({false, false}, outputs[4]);
  }
}

// The configuration production compiles with: default debug options, which
// select the all-reduce kernel, no symmetric buffers and no disabled passes.
// Non-uniform inputs keep algsimp from shrinking the all-reduces, and the
// two-shot input depends on the one-shot result so the combiner cannot merge
// them.
TEST_P(CollectiveKernelsAotTest, ProductionConfig) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT res = f32[] add(lhs, rhs)
    }

    ENTRY main {
      id = u32[] replica-id()
      id_f = f32[] convert(id)
      thousand = f32[] constant(1000)
      base = f32[] multiply(id_f, thousand)

      base_small = f32[128] broadcast(base), dimensions={}
      iota_small = f32[128] iota(), iota_dimension=0
      in_small = f32[128] add(iota_small, base_small)
      ar_oneshot = f32[128] all-reduce(in_small), replica_groups={{0,1}},
                            to_apply=add_f32

      first = f32[1] slice(ar_oneshot), slice={[0:1]}
      first_s = f32[] reshape(first)
      offset = f32[] add(first_s, base)
      offset_large = f32[524288] broadcast(offset), dimensions={}
      iota_large = f32[524288] iota(), iota_dimension=0
      in_large = f32[524288] add(iota_large, offset_large)
      ar_twoshot = f32[524288] all-reduce(in_large), replica_groups={{0,1}},
                               to_apply=add_f32

      s_small = f32[2] slice(ar_oneshot), slice={[0:2]}
      // One element from the half each rank reduces in the two-shot kernel.
      s_large_r0 = f32[1] slice(ar_twoshot), slice={[0:1]}
      s_large_r1 = f32[1] slice(ar_twoshot), slice={[524287:524288]}
      s_large = f32[2] concatenate(s_large_r0, s_large_r1), dimensions={0}
      ROOT out = (f32[2], f32[2]) tuple(s_small, s_large)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // in_small = iota + 1000 * id, so ar_oneshot = 2 * iota + 1000 and its
  // first element is 1000. in_large = iota + 1000 + 1000 * id, so
  // ar_twoshot = 2 * iota + 3000.
  for (int64_t replica = 0; replica < kNumReplicas; ++replica) {
    std::vector<Literal> outputs = results[replica].DecomposeTuple();
    LiteralTestUtil::ExpectR1Near<float>({1000, 1002}, outputs[0],
                                         ErrorSpec{1e-4, 1e-4});
    LiteralTestUtil::ExpectR1Near<float>({3000, 1051574}, outputs[1],
                                         ErrorSpec{1e-4, 1e-4});
  }
}

// The all-reduce kernel inside a WhileThunk body: its scratch buffers and
// signal counters must stay consistent across iterations.
TEST_P(CollectiveKernelsAotTest, AllReduceInWhileLoop) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT res = f32[] add(lhs, rhs)
    }

    body {
      param = (u32[], f32[128]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      data = f32[128] get-tuple-element(param), index=1
      ar = f32[128] all-reduce(data), replica_groups={{0,1}}, to_apply=add_f32
      one = u32[] constant(1)
      i_next = u32[] add(i, one)
      ROOT next = (u32[], f32[128]) tuple(i_next, ar)
    }

    cond {
      param = (u32[], f32[128]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      limit = u32[] constant(4)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      id = u32[] replica-id()
      id_f = f32[] convert(id)
      id_b = f32[128] broadcast(id_f), dimensions={}
      io = f32[128] iota(), iota_dimension=0
      init_data = f32[128] add(io, id_b)
      zero = u32[] constant(0)
      init = (u32[], f32[128]) tuple(zero, init_data)
      loop = (u32[], f32[128]) while(init), condition=cond, body=body
      data_out = f32[128] get-tuple-element(loop), index=1
      ROOT out = f32[2] slice(data_out), slice={[0:2]}
    }
  )hlo";
  HloModuleConfig config =
      CollectiveKernelsConfig(GetModuleConfigForTest(kNumReplicas));

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // init = iota + id; each iteration sums both replicas:
  //   2 * iota + 1 -> 4 * iota + 2 -> 8 * iota + 4 -> 16 * iota + 8.
  for (int64_t replica = 0; replica < kNumReplicas; ++replica) {
    LiteralTestUtil::ExpectR1Near<float>({8, 24}, results[replica],
                                         ErrorSpec{1e-4, 1e-4});
  }
}

// Kernels with and without the host-provided invocation count argument in one
// executable, each launched once per loop iteration. The all-reduce (one-shot
// and two-shot), all-gather and all-to-all kernels still take the argument and
// ignore it; the reduce-scatter kernel does not take it. All of them read their
// barrier signal value from device memory (`device_sync_count` in
// CollectiveCodegenConfig), so it must advance on every launch. Once newer
// kernels drop the argument, this golden still holds kernels that expect it.
TEST_P(CollectiveKernelsAotTest, InvocationCountArgument) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    add_f32 {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT res = f32[] add(lhs, rhs)
    }

    body {
      param = (u32[], f32[256], f32[524288], f32[256]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      small = f32[256] get-tuple-element(param), index=1
      large = f32[524288] get-tuple-element(param), index=2
      a2a_in = f32[256] get-tuple-element(param), index=3
      ar_oneshot = f32[256] all-reduce(small), replica_groups={{0,1}},
                            to_apply=add_f32
      ar_twoshot = f32[524288] all-reduce(large), replica_groups={{0,1}},
                               to_apply=add_f32
      rs = f32[128] reduce-scatter(ar_oneshot), dimensions={0},
                    replica_groups={{0,1}}, to_apply=add_f32
      ag = f32[256] all-gather(rs), dimensions={0}, replica_groups={{0,1}}
      a2a = f32[256] all-to-all(a2a_in), dimensions={0},
                     replica_groups={{0,1}}
      one = u32[] constant(1)
      i_next = u32[] add(i, one)
      ROOT next = (u32[], f32[256], f32[524288], f32[256])
                  tuple(i_next, ag, ar_twoshot, a2a)
    }

    cond {
      param = (u32[], f32[256], f32[524288], f32[256]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      limit = u32[] constant(3)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      id = u32[] replica-id()
      id_f = f32[] convert(id)
      thousand = f32[] constant(1000)
      base = f32[] multiply(id_f, thousand)
      base_small = f32[256] broadcast(base), dimensions={}
      iota_small = f32[256] iota(), iota_dimension=0
      init_small = f32[256] add(iota_small, base_small)
      base_large = f32[524288] broadcast(base), dimensions={}
      iota_large = f32[524288] iota(), iota_dimension=0
      init_large = f32[524288] add(iota_large, base_large)
      zero = u32[] constant(0)
      init = (u32[], f32[256], f32[524288], f32[256])
             tuple(zero, init_small, init_large, init_small)
      loop = (u32[], f32[256], f32[524288], f32[256]) while(init),
                                                      condition=cond, body=body
      small_out = f32[256] get-tuple-element(loop), index=1
      large_out = f32[524288] get-tuple-element(loop), index=2
      a2a_out = f32[256] get-tuple-element(loop), index=3
      // First and last elements: the reduce-scatter, the two-shot kernel and
      // the all-to-all split the arrays between the ranks.
      small_first = f32[1] slice(small_out), slice={[0:1]}
      small_last = f32[1] slice(small_out), slice={[255:256]}
      large_first = f32[1] slice(large_out), slice={[0:1]}
      large_last = f32[1] slice(large_out), slice={[524287:524288]}
      a2a_first = f32[1] slice(a2a_out), slice={[0:1]}
      a2a_last = f32[1] slice(a2a_out), slice={[255:256]}
      ROOT out = f32[6] concatenate(small_first, small_last, large_first,
                                    large_last, a2a_first, a2a_last),
                                    dimensions={0}
    }
  )hlo";
  HloModuleConfig config =
      CollectiveKernelsConfig(GetModuleConfigForTest(kNumReplicas));

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  // init = iota + 1000 * id. Each iteration, small becomes twice its
  // all-reduce (the reduce-scatter adds up two identical all-reduce results
  // and the all-gather reassembles them), and large becomes its all-reduce:
  //   small: 4 * iota + 2000 -> 16 * iota + 8000 -> 64 * iota + 32000
  //   large: 2 * iota + 1000 -> 4 * iota + 2000 -> 8 * iota + 4000
  // The all-to-all swaps the second half of rank 0 with the first half of
  // rank 1; after an odd number of swaps rank 0 holds iota[0:128] and
  // 1000 + iota[0:128], rank 1 holds iota[128:256] and 1000 + iota[128:256].
  LiteralTestUtil::ExpectR1Near<float>({32000, 48320, 4000, 4198296, 0, 1127},
                                       results[0], ErrorSpec{1e-4, 1e-4});
  LiteralTestUtil::ExpectR1Near<float>({32000, 48320, 4000, 4198296, 128, 1255},
                                       results[1], ErrorSpec{1e-4, 1e-4});
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, CollectiveKernelsAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, CollectiveKernelsAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
