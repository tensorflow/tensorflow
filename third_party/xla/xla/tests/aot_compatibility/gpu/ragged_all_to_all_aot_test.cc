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

// AOT compatibility of RaggedAllToAllThunk on 2 GPUs. Its implementation is
// selected by debug options at compile time and serialized into the thunk, so
// each implementation is its own case.

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
#include "xla/xla.pb.h"

namespace xla {
namespace aot_compatibility {
namespace {

using ::testing::ValuesIn;

constexpr absl::string_view kTargetBase = "ragged_all_to_all_aot_test_2gpu";
constexpr int64_t kNumReplicas = 2;

// One single-row update per peer. Replica r holds input = 10 * r + {0, 1, 2, 3}
// and sends row 0 to replica 0 and row 2 to replica 1, both to output row
// 2 * r of the receiver.
constexpr absl::string_view kRaggedAllToAllHlo = R"hlo(
  HloModule test
  ENTRY main {
    io = f32[4] iota(), iota_dimension=0
    id = u32[] replica-id()
    ten = u32[] constant(10)
    id2 = u32[] multiply(id, ten)
    id3 = f32[] convert(id2)
    id4 = f32[4] broadcast(id3), dimensions={}
    input = f32[4] add(io, id4)
    output = f32[4] constant({-1, -1, -1, -1})
    send_sizes = s32[2] constant({1, 1})
    recv_sizes = s32[2] constant({1, 1})
    input_offsets = s32[2] constant({0, 2})
    step = u32[] constant(2)
    oof = u32[] multiply(id, step)
    oof2 = s32[] convert(oof)
    output_offsets = s32[2] broadcast(oof2), dimensions={}
    ROOT ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
      send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
  }
)hlo";

class RaggedAllToAllAotTest : public CollectiveAotTestBase {
 protected:
  void RunAndExpectRaggedAllToAll(const HloModuleConfig& config) {
    ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                         RunReplicated(kRaggedAllToAllHlo, config));
    LiteralTestUtil::ExpectR1Near<float>({0.0f, -1.0f, 10.0f, -1.0f},
                                         results[0], ErrorSpec{1e-5, 1e-5});
    LiteralTestUtil::ExpectR1Near<float>({2.0f, -1.0f, 12.0f, -1.0f},
                                         results[1], ErrorSpec{1e-5, 1e-5});
  }
};

// One-shot kernel synchronized with an NCCL-based multi-GPU barrier.
TEST_P(RaggedAllToAllAotTest, OneShotWithNcclBarrier) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(
      true);
  debug_options
      .set_xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl(true);
  RunAndExpectRaggedAllToAll(config);
}

// One-shot kernel synchronized with the local (single process) barrier.
TEST_P(RaggedAllToAllAotTest, OneShotLocalBarrier) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(
      true);
  debug_options
      .set_xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl(false);
  RunAndExpectRaggedAllToAll(config);
}

// NCCL send/recv fallback in COLLECTIVES_PRIVATE_MEMORY mode.
TEST_P(RaggedAllToAllAotTest, NcclSendRecvFallback) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(
      false);
  debug_options.set_xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback(
      true);
  debug_options.set_xla_gpu_ragged_all_to_all_mode(
      DebugOptions::COLLECTIVES_PRIVATE_MEMORY);
  debug_options
      .set_xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl(false);
  RunAndExpectRaggedAllToAll(config);
}

// NCCL send/recv on symmetric memory with use_symmetric_buffer set.
TEST_P(RaggedAllToAllAotTest, SymmetricMemory) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_ragged_all_to_all_mode(
      DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY);
  debug_options.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(
      false);
  debug_options.set_xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback(
      true);
  debug_options.set_xla_gpu_experimental_nccl_window_symmetric(true);
  EnableNcclSymmetricBuffers(debug_options);
  RunAndExpectRaggedAllToAll(config);
}

// Serializes collectives_mode = COLLECTIVES_PEER_MEMORY. The runtime has no
// peer-memory path yet and runs the default one-shot kernel, so this case
// guards the serialized mode, not a distinct implementation.
TEST_P(RaggedAllToAllAotTest, PeerMemory) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  config.mutable_debug_options().set_xla_gpu_ragged_all_to_all_mode(
      DebugOptions::COLLECTIVES_PEER_MEMORY);
  RunAndExpectRaggedAllToAll(config);
}

// Device-initiated communication kernel on symmetric memory.
TEST_P(RaggedAllToAllAotTest, DeviceKernel) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_experimental_ragged_all_to_all_use_device_kernel(
      true);
  debug_options.set_xla_gpu_ragged_all_to_all_mode(
      DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY);
  EnableNcclSymmetricBuffers(debug_options);
  debug_options
      .set_xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl(false);
  RunAndExpectRaggedAllToAll(config);
}

// RaggedAllToAllThunkProto.fast_interconnect_slice_size_override, which
// replaces the communicator's LSA size for the NCCL-barrier one-shot kernel.
// Both GPUs are in one slice, so 2 keeps the fast path.
TEST_P(RaggedAllToAllAotTest, FastInterconnectSliceSizeOverride) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel(
      true);
  debug_options
      .set_xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl(true);
  debug_options.set_xla_gpu_unsupported_override_fast_interconnect_slice_size(
      kNumReplicas);
  RunAndExpectRaggedAllToAll(config);
}

// Not covered: enable_gxl (xla_gpu_enable_gxl_ragged_all_to_all). The thunk
// connects to the host's GXL daemon at initialization, which only exists on
// some test hosts, so the result depends on the machine rather than on the
// serialized executable.

// Multiple updates per peer, multi-dimensional rows and s64 offsets/sizes.
//
// input[r, c] = 100 * id + 10 * r + c. Offsets and sizes are indexed by
// 2 * peer + update. Every replica sends
//   to replica 0: row 0 (update 0) and rows 2-3 (update 1),
//   to replica 1: rows 4-5 (update 0) and row 7 (update 1),
// and the sender with id s writes its 3 rows to rows 4 * s .. 4 * s + 2 of the
// receiver's output, leaving rows 3 and 7 at -1.
TEST_P(RaggedAllToAllAotTest, MultipleUpdatesMultiDimRows) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      id = u32[] replica-id()
      id_s = s32[] convert(id)
      hundred = s32[] constant(100)
      base = s32[] multiply(id_s, hundred)
      base2 = s32[8,2] broadcast(base), dimensions={}
      rows = s32[8,2] iota(), iota_dimension=0
      ten = s32[8,2] constant({{10,10},{10,10},{10,10},{10,10},
                              {10,10},{10,10},{10,10},{10,10}})
      rows10 = s32[8,2] multiply(rows, ten)
      cols = s32[8,2] iota(), iota_dimension=1
      in0 = s32[8,2] add(base2, rows10)
      in1 = s32[8,2] add(in0, cols)
      input = bf16[8,2] convert(in1)
      output = bf16[8,2] constant({{-1,-1},{-1,-1},{-1,-1},{-1,-1},
                                  {-1,-1},{-1,-1},{-1,-1},{-1,-1}})

      input_offsets = s64[4] constant({0, 2, 4, 7})
      send_sizes = s64[4] constant({1, 2, 2, 1})
      id64 = s64[] convert(id)
      four = s64[] constant(4)
      out_base = s64[] multiply(id64, four)
      out_base4 = s64[4] broadcast(out_base), dimensions={}
      out_delta = s64[4] constant({0, 1, 0, 2})
      output_offsets = s64[4] add(out_base4, out_delta)
      zero = u32[] constant(0)
      is_r0 = pred[] compare(id, zero), direction=EQ
      is_r0_4 = pred[4] broadcast(is_r0), dimensions={}
      recv_r0 = s64[4] constant({1, 2, 1, 2})
      recv_r1 = s64[4] constant({2, 1, 2, 1})
      recv_sizes = s64[4] select(is_r0_4, recv_r0, recv_r1)

      ra2a = bf16[8,2] ragged-all-to-all(input, output, input_offsets,
        send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
      ROOT out = f32[8,2] convert(ra2a)
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                       RunReplicated(kHlo, config));
  LiteralTestUtil::ExpectR2Equal<float>({{0, 1},
                                         {20, 21},
                                         {30, 31},
                                         {-1, -1},
                                         {100, 101},
                                         {120, 121},
                                         {130, 131},
                                         {-1, -1}},
                                        results[0]);
  LiteralTestUtil::ExpectR2Equal<float>({{40, 41},
                                         {50, 51},
                                         {70, 71},
                                         {-1, -1},
                                         {140, 141},
                                         {150, 151},
                                         {170, 171},
                                         {-1, -1}},
                                        results[1]);
}

// The default implementation kept asynchronous (AsyncStart/AsyncDone around
// the ragged-all-to-all thunk).
TEST_P(RaggedAllToAllAotTest, Async) {
  HloModuleConfig config = GetModuleConfigForTest(kNumReplicas);
  KeepCollectivesAsync(config.mutable_debug_options());
  RunAndExpectRaggedAllToAll(config);
}

// Singleton replica groups {{0},{1}}: every replica exchanges with itself
// only, yet the op stays a ragged-all-to-all thunk. Replica r copies its rows
// 1-2 to output rows 0-1.
TEST_P(RaggedAllToAllAotTest, SingletonGroups) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      io = f32[4] iota(), iota_dimension=0
      id = u32[] replica-id()
      ten = u32[] constant(10)
      id10 = u32[] multiply(id, ten)
      id10_f = f32[] convert(id10)
      base = f32[4] broadcast(id10_f), dimensions={}
      input = f32[4] add(io, base)
      output = f32[4] constant({-1, -1, -1, -1})
      input_offsets = s32[1] constant({1})
      send_sizes = s32[1] constant({2})
      output_offsets = s32[1] constant({0})
      recv_sizes = s32[1] constant({2})
      ROOT ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
        send_sizes, output_offsets, recv_sizes), replica_groups={{0},{1}}
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      RunReplicated(kHlo, GetModuleConfigForTest(kNumReplicas)));
  LiteralTestUtil::ExpectR1Near<float>({1.0f, 2.0f, -1.0f, -1.0f}, results[0],
                                       ErrorSpec{1e-5, 1e-5});
  LiteralTestUtil::ExpectR1Near<float>({11.0f, 12.0f, -1.0f, -1.0f}, results[1],
                                       ErrorSpec{1e-5, 1e-5});
}

// The default implementation inside a WhileThunk body: its barrier state must
// carry over across iterations. Each iteration runs the exchange of
// kRaggedAllToAllHlo on the carried data and adds 1.
TEST_P(RaggedAllToAllAotTest, InWhileLoop) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test

    body {
      param = (u32[], f32[4]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      input = f32[4] get-tuple-element(param), index=1
      output = f32[4] constant({-1, -1, -1, -1})
      send_sizes = s32[2] constant({1, 1})
      recv_sizes = s32[2] constant({1, 1})
      input_offsets = s32[2] constant({0, 2})
      id = u32[] replica-id()
      step = u32[] constant(2)
      oof = u32[] multiply(id, step)
      oof2 = s32[] convert(oof)
      output_offsets = s32[2] broadcast(oof2), dimensions={}
      ra2a = f32[4] ragged-all-to-all(input, output, input_offsets,
        send_sizes, output_offsets, recv_sizes), replica_groups={{0,1}}
      one_f = f32[] constant(1)
      one_b = f32[4] broadcast(one_f), dimensions={}
      data_next = f32[4] add(ra2a, one_b)
      one = u32[] constant(1)
      i_next = u32[] add(i, one)
      ROOT next = (u32[], f32[4]) tuple(i_next, data_next)
    }

    cond {
      param = (u32[], f32[4]) parameter(0)
      i = u32[] get-tuple-element(param), index=0
      limit = u32[] constant(3)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    ENTRY main {
      io = f32[4] iota(), iota_dimension=0
      id = u32[] replica-id()
      ten = u32[] constant(10)
      id10 = u32[] multiply(id, ten)
      id10_f = f32[] convert(id10)
      base = f32[4] broadcast(id10_f), dimensions={}
      input = f32[4] add(io, base)
      zero = u32[] constant(0)
      init = (u32[], f32[4]) tuple(zero, input)
      loop = (u32[], f32[4]) while(init), condition=cond, body=body
      ROOT out = f32[4] get-tuple-element(loop), index=1
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      RunReplicated(kHlo, GetModuleConfigForTest(kNumReplicas)));
  // Each iteration, replica 0 receives both replicas' row 0 in rows 0 and 2,
  // replica 1 both replicas' row 2; rows 1 and 3 stay -1, then +1:
  //   init  r0 = {0, 1, 2, 3},     r1 = {10, 11, 12, 13}
  //   it 1  r0 = {1, 0, 11, 0},    r1 = {3, 0, 13, 0}
  //   it 2  r0 = {2, 0, 4, 0},     r1 = {12, 0, 14, 0}
  //   it 3  r0 = {3, 0, 13, 0},    r1 = {5, 0, 15, 0}
  LiteralTestUtil::ExpectR1Near<float>({3.0f, 0.0f, 13.0f, 0.0f}, results[0],
                                       ErrorSpec{1e-5, 1e-5});
  LiteralTestUtil::ExpectR1Near<float>({5.0f, 0.0f, 15.0f, 0.0f}, results[1],
                                       ErrorSpec{1e-5, 1e-5});
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, RaggedAllToAllAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, RaggedAllToAllAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
