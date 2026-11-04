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

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

constexpr ErrorSpec kExactMatch{/*aabs=*/0, /*arel=*/0};

// End-to-end numerics tests for the TensorIR fusion emitter.
//
// A fusion is routed to the TensorIR emitter iff it is a `kCustom` fusion
// whose `fusion_backend_config.kind` is `__tensorir` and which carries a
// `tensor_ir_fusion_config` (see `HloFusionAnalysis::GetEmitterFusionKind` and
// `TensorIrFusion::Emit`). The tile sizes inside that config are optional; if
// they are omitted, the pipeline's own tile analysis picks them, which is what
// most tests below rely on.
//
// All tests run without HLO passes so that the optimizer does not rewrite the
// hand-written fusions and route them to a different emitter.
class TensorIrEmitterTest
    : public HloInterpreterReferenceMixin<GpuPjRtCodegenTest> {};

TEST_F(TensorIrEmitterTest, ElementwiseAddF32) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT add = f32[8,16] add(p0, p1)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT fusion = f32[8,16] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// The tile size in the backend config is expressed in *iteration space*
// terms, not in output-shape terms: layout propagation flattens the iteration
// space of an elementwise fusion to rank 1, so a rank-2 tile is rejected with
// "tile rank 2 vs iteration space rank 1".
TEST_F(TensorIrEmitterTest, ElementwiseAddF32WithExplicitTileSize) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[64,128] parameter(0)
  p1 = f32[64,128] parameter(1)
  ROOT add = f32[64,128] add(p0, p1)
}

ENTRY main {
  p0 = f32[64,128] parameter(0)
  p1 = f32[64,128] parameter(1)
  ROOT fusion = f32[64,128] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{"tile_size":[256]}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TensorIrEmitterTest, CopyChangesLayout) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[8,16]{1,0} parameter(0)
  neg = f32[8,16]{1,0} negate(p0)
  ROOT copy = f32[8,16]{0,1} copy(neg)
}

ENTRY main {
  p0 = f32[8,16]{1,0} parameter(0)
  ROOT fusion = f32[8,16]{0,1} fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TensorIrEmitterTest, MultiOutputElementwiseF32) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  add = f32[8,16] add(p0, p1)
  mul = f32[8,16] multiply(p0, p1)
  ROOT tuple = (f32[8,16], f32[8,16]) tuple(add, mul)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT fusion = (f32[8,16], f32[8,16]) fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TensorIrEmitterTest, MultiOutputWithDifferentTypes) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[8,16] parameter(0)
  neg = f32[8,16] negate(p0)
  cvt = s32[8,16] convert(p0)
  ROOT tuple = (f32[8,16], s32[8,16]) tuple(neg, cvt)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  ROOT fusion = (f32[8,16], s32[8,16]) fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TensorIrEmitterTest, MultiOutputElementwiseAndReduceF32) {
  constexpr absl::string_view kHloText = R"(
add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT sum = f32[] add(a, b)
}

fused_computation {
  p0 = f32[8,16] parameter(0)
  neg = f32[8,16] negate(p0)
  zero = f32[] constant(0)
  sum = f32[8] reduce(p0, zero), dimensions={1}, to_apply=add_f32
  ROOT tuple = (f32[8,16], f32[8]) tuple(neg, sum)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  ROOT fusion = (f32[8,16], f32[8]) fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TensorIrEmitterTest, ElementwiseMulMaxF32) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  mul = f32[8,16] multiply(p0, p1)
  ROOT max = f32[8,16] maximum(mul, p0)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT fusion = f32[8,16] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TensorIrEmitterTest, UnaryElementwiseF32) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[8,16] parameter(0)
  abs = f32[8,16] abs(p0)
  ROOT neg = f32[8,16] negate(abs)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  ROOT fusion = f32[8,16] fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TensorIrEmitterTest, ConvertS32ToF32) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = s32[8,16] parameter(0)
  ROOT convert = f32[8,16] convert(p0)
}

ENTRY main {
  p0 = s32[8,16] parameter(0)
  ROOT fusion = f32[8,16] fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TensorIrEmitterTest, ConvertF32ToF16) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[8,16] parameter(0)
  ROOT convert = f16[8,16] convert(p0)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  ROOT fusion = f16[8,16] fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// Integer elementwise arithmetic exercises the signless -> signed bridge in
// the TypeConverter: StableHLO uses signless `i32` while nv_tensor_ir requires
// signed `si32`.
TEST_F(TensorIrEmitterTest, ElementwiseS32) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = s32[8,16] parameter(0)
  p1 = s32[8,16] parameter(1)
  mul = s32[8,16] multiply(p0, p1)
  ROOT sub = s32[8,16] subtract(mul, p1)
}

ENTRY main {
  p0 = s32[8,16] parameter(0)
  p1 = s32[8,16] parameter(1)
  ROOT fusion = s32[8,16] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// Signed division and remainder must not be lowered to their unsigned
// counterparts, which is only observable for negative operands.
TEST_F(TensorIrEmitterTest, SignedDivideAndRemainderS32) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = s32[2,4] parameter(0)
  p1 = s32[2,4] parameter(1)
  div = s32[2,4] divide(p0, p1)
  rem = s32[2,4] remainder(p0, p1)
  ROOT add = s32[2,4] add(div, rem)
}

ENTRY main {
  p0 = s32[2,4] parameter(0)
  p1 = s32[2,4] parameter(1)
  ROOT fusion = s32[2,4] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  Literal lhs = LiteralUtil::CreateR2<int32_t>(
      {{-7, -1, 0, 1}, {7, -2147483648, 13, -13}});
  Literal rhs =
      LiteralUtil::CreateR2<int32_t>({{2, -3, 5, -1}, {-2, 3, -4, 4}});
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module), {&lhs, &rhs}, kExactMatch));
}

TEST_F(TensorIrEmitterTest, SelectAndClampF32) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  gt = pred[8,16] compare(p0, p1), direction=GT
  sel = f32[8,16] select(gt, p0, p1)
  ROOT res = f32[8,16] maximum(sel, p1)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT fusion = f32[8,16] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// This case is load-bearing, not decoration: float `NE` must lower to the
// *unordered* comparator `une`, because IEEE-754 makes `NaN != x` true for
// every `x`, including `NaN != NaN`. The emitter used to lower it to the
// ordered `one`, which returns false for NaN operands; that was a real bug we
// fixed. Feeding an actual NaN (plus +/-Inf and -0.0) through the kernel and
// comparing exactly is the only thing that catches a regression here; random
// inputs never produce a NaN.
TEST_F(TensorIrEmitterTest, CompareNeWithNaN) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[2,4] parameter(0)
  p1 = f32[2,4] parameter(1)
  ROOT ne = pred[2,4] compare(p0, p1), direction=NE
}

ENTRY main {
  p0 = f32[2,4] parameter(0)
  p1 = f32[2,4] parameter(1)
  ROOT fusion = pred[2,4] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  const float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float kInf = std::numeric_limits<float>::infinity();
  // Row 0 pairs a NaN against itself, a number, an infinity and -0.0; all four
  // comparisons must be true. Row 1 covers the ordered cases.
  Literal lhs = LiteralUtil::CreateR2<float>(
      {{kNaN, kNaN, kNaN, kNaN}, {1.0f, 1.0f, -0.0f, kInf}});
  Literal rhs = LiteralUtil::CreateR2<float>(
      {{kNaN, 1.0f, kInf, -0.0f}, {1.0f, 2.0f, 0.0f, kInf}});
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module), {&lhs, &rhs}, kExactMatch));
}

// The other float comparison directions must stay *ordered*: every comparison
// against a NaN other than `NE` is false.
TEST_F(TensorIrEmitterTest, CompareOrderedDirectionsWithNaN) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[2,4] parameter(0)
  p1 = f32[2,4] parameter(1)
  eq = pred[2,4] compare(p0, p1), direction=EQ
  lt = pred[2,4] compare(p0, p1), direction=LT
  ge = pred[2,4] compare(p0, p1), direction=GE
  or0 = pred[2,4] or(eq, lt)
  ROOT res = pred[2,4] or(or0, ge)
}

ENTRY main {
  p0 = f32[2,4] parameter(0)
  p1 = f32[2,4] parameter(1)
  ROOT fusion = pred[2,4] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  const float kNaN = std::numeric_limits<float>::quiet_NaN();
  Literal lhs = LiteralUtil::CreateR2<float>(
      {{kNaN, kNaN, 1.0f, 2.0f}, {-1.0f, 0.0f, 1.0f, 2.0f}});
  Literal rhs = LiteralUtil::CreateR2<float>(
      {{kNaN, 1.0f, kNaN, 2.0f}, {1.0f, -0.0f, 1.0f, 3.0f}});
  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module), {&lhs, &rhs}, kExactMatch));
}

// Reductions whose body is a plain `add`/`max` lower to TensorIR's built-in
// reduction modes.
TEST_F(TensorIrEmitterTest, ReduceAddF32) {
  constexpr absl::string_view kHloText = R"(
add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

fused_computation {
  p0 = f32[16,64] parameter(0)
  c0 = f32[] constant(0)
  ROOT reduce = f32[16] reduce(p0, c0), dimensions={1}, to_apply=add_f32
}

ENTRY main {
  p0 = f32[16,64] parameter(0)
  ROOT fusion = f32[16] fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  // The reference interpreter sums sequentially while the kernel sums in a
  // tree, so the results differ in the last few ulps.
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(TensorIrEmitterTest, ReduceMaxF32) {
  constexpr absl::string_view kHloText = R"(
max_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT max = f32[] maximum(lhs, rhs)
}

fused_computation {
  p0 = f32[16,64] parameter(0)
  c0 = f32[] constant(-inf)
  ROOT reduce = f32[16] reduce(p0, c0), dimensions={1}, to_apply=max_f32
}

ENTRY main {
  p0 = f32[16,64] parameter(0)
  ROOT fusion = f32[16] fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// A reduction body that is not a bare built-in mode (here: operands flipped)
// lowers to `reduce_ud`, i.e. a user-defined reduction region.
TEST_F(TensorIrEmitterTest, ReduceUserDefinedF32) {
  constexpr absl::string_view kHloText = R"(
add_flipped {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(rhs, lhs)
}

fused_computation {
  p0 = f32[16,64] parameter(0)
  c0 = f32[] constant(0)
  ROOT reduce = f32[16] reduce(p0, c0), dimensions={1}, to_apply=add_flipped
}

ENTRY main {
  p0 = f32[16,64] parameter(0)
  ROOT fusion = f32[16] fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{1e-5, 1e-5}));
}

// Integer reductions always lower to `reduce_ud`, and additionally exercise
// the signless -> signed bridge inside the reduction region.
TEST_F(TensorIrEmitterTest, ReduceUserDefinedS32) {
  constexpr absl::string_view kHloText = R"(
add_s32 {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

fused_computation {
  p0 = s32[16,64] parameter(0)
  c0 = s32[] constant(0)
  ROOT reduce = s32[16] reduce(p0, c0), dimensions={1}, to_apply=add_s32
}

ENTRY main {
  p0 = s32[16,64] parameter(0)
  ROOT fusion = s32[16] fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TensorIrEmitterTest, DotF32) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[64,32] parameter(0)
  p1 = f32[32,16] parameter(1)
  ROOT dot = f32[64,16] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  p0 = f32[64,32] parameter(0)
  p1 = f32[32,16] parameter(1)
  ROOT fusion = f32[64,16] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{1e-3, 1e-3}));
}

TEST_F(TensorIrEmitterTest, DotWithBatchF32) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[4,16,32] parameter(0)
  p1 = f32[4,32,16] parameter(1)
  ROOT dot = f32[4,16,16] dot(p0, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  p0 = f32[4,16,32] parameter(0)
  p1 = f32[4,32,16] parameter(1)
  ROOT fusion = f32[4,16,16] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{1e-3, 1e-3}));
}

TEST_F(TensorIrEmitterTest, BroadcastTransposeSlice) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[8,16] parameter(0)
  transpose = f32[16,8] transpose(p0), dimensions={1,0}
  slice = f32[8,4] slice(transpose), slice={[0:8], [0:4]}
  ROOT reshape = f32[32] reshape(slice)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  ROOT fusion = f32[32] fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// Non-default layouts must reach the kernel as stride attributes; see
// `tensor_ir::AttachBoundaryAttributes`. The first operand is column major.
TEST_F(TensorIrEmitterTest, NonDefaultOperandLayout) {
  constexpr absl::string_view kHloText = R"(
HloModule m, entry_computation_layout={(f32[8,16]{0,1},f32[8,16]{1,0})->f32[8,16]{1,0}}

fused_computation {
  p0 = f32[8,16]{0,1} parameter(0)
  p1 = f32[8,16]{1,0} parameter(1)
  ROOT add = f32[8,16]{1,0} add(p0, p1)
}

ENTRY main {
  p0 = f32[8,16]{0,1} parameter(0)
  p1 = f32[8,16]{1,0} parameter(1)
  ROOT fusion = f32[8,16]{1,0} fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// Same, but the fusion output is column major as well.
TEST_F(TensorIrEmitterTest, NonDefaultOutputLayout) {
  constexpr absl::string_view kHloText = R"(
HloModule m, entry_computation_layout={(f32[8,16]{1,0},f32[8,16]{1,0})->f32[8,16]{0,1}}

fused_computation {
  p0 = f32[8,16]{1,0} parameter(0)
  p1 = f32[8,16]{1,0} parameter(1)
  ROOT add = f32[8,16]{0,1} add(p0, p1)
}

ENTRY main {
  p0 = f32[8,16]{1,0} parameter(0)
  p1 = f32[8,16]{1,0} parameter(1)
  ROOT fusion = f32[8,16]{0,1} fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// Rank-0 tensors need care: CudaTile's layout propagation rejects a rank-0
// graph argument, graph result or constant, so the legalization pass rewrites
// all three to rank 1. See `PromoteRank0Tensors`.
TEST_F(TensorIrEmitterTest, ScalarParameterBroadcast) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[256] parameter(0)
  p1 = f32[] parameter(1)
  b = f32[256] broadcast(p1), dimensions={}
  ROOT a = f32[256] add(p0, b)
}

ENTRY main {
  p0 = f32[256] parameter(0)
  p1 = f32[] parameter(1)
  ROOT fusion = f32[256] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TensorIrEmitterTest, ScalarConstantBroadcast) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[256] parameter(0)
  c = f32[] constant(2)
  b = f32[256] broadcast(c), dimensions={}
  ROOT m = f32[256] multiply(p0, b)
}

ENTRY main {
  p0 = f32[256] parameter(0)
  ROOT fusion = f32[256] fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

// A full reduce, so the fusion root is rank 0.
TEST_F(TensorIrEmitterTest, ReduceToScalar) {
  constexpr absl::string_view kHloText = R"(
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

fused_computation {
  p0 = f32[256] parameter(0)
  c0 = f32[] constant(0)
  ROOT r = f32[] reduce(p0, c0), dimensions={0}, to_apply=add
}

ENTRY main {
  p0 = f32[256] parameter(0)
  ROOT fusion = f32[] fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{1e-5, 1e-5}));
}

// Acceptance criterion 4: the same kernels, executed from inside a command
// buffer rather than launched directly. The thunk test next door proves the
// conversion happens; this proves the recorded graph computes the same thing.
class TensorIrCommandBufferTest : public TensorIrEmitterTest {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = TensorIrEmitterTest::GetDebugOptionsForTest();
    debug_options.clear_xla_gpu_enable_command_buffer();
    debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
    // Otherwise a single-kernel graph is below the size threshold.
    debug_options.set_xla_gpu_graph_min_graph_size(1);
    return debug_options;
  }
};

TEST_F(TensorIrCommandBufferTest, ElementwiseAddF32) {
  constexpr absl::string_view kHloText = R"(
fused_computation {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT add = f32[8,16] add(p0, p1)
}

ENTRY main {
  p0 = f32[8,16] parameter(0)
  p1 = f32[8,16] parameter(1)
  ROOT fusion = f32[8,16] fusion(p0, p1), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, kExactMatch));
}

TEST_F(TensorIrCommandBufferTest, ReduceToScalar) {
  constexpr absl::string_view kHloText = R"(
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}

fused_computation {
  p0 = f32[256] parameter(0)
  c0 = f32[] constant(0)
  ROOT r = f32[] reduce(p0, c0), dimensions={0}, to_apply=add
}

ENTRY main {
  p0 = f32[256] parameter(0)
  ROOT fusion = f32[] fusion(p0), kind=kCustom, calls=fused_computation,
    backend_config={"fusion_backend_config":{"kind":"__tensorir","tensor_ir_fusion_config":{}}}
})";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla::gpu
