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

#include "xla/backends/gpu/transforms/dus_accumulator_zero_init_elimination.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"

namespace xla::gpu {
namespace {

class DusAccumulatorZeroInitEliminationTest
    : public HloHardwareIndependentTestBase {
 protected:
  void EnableFlag(HloModule* module, bool value = true) {
    auto opts = module->config().debug_options();
    opts.set_xla_gpu_enable_dus_accumulator_zero_init_elimination(value);
    module->mutable_config().set_debug_options(opts);
  }

  int64_t CountAllocateBuffers(HloModule* module) {
    int64_t n = 0;
    for (const HloComputation* c : module->computations()) {
      for (const HloInstruction* i : c->instructions()) {
        if (i->opcode() == HloOpcode::kCustomCall &&
            i->custom_call_target() == "AllocateBuffer") {
          ++n;
        }
      }
    }
    return n;
  }
};

// Scaffold A: single-slot raw-DUS scan with broadcast(0) init. Cases override
// only what differs from "BasicAscendingRawDus".
struct RawDusCase {
  absl::string_view name;
  bool expected_changed;
  int64_t expected_buffers;
  absl::string_view body_dus = R"(  z = s32[] constant(0)
  dus = bf16[4,8] dynamic-update-slice(acc, update, it, z))";
  absl::string_view init_expr = R"(  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] broadcast(c0), dimensions={})";
  absl::string_view while_extra =
      "";  // extra attribute on `w =`, e.g. sharding
  absl::string_view trip_limit = "4";
  absl::string_view iv_init = "z";     // ENTRY-level name feeding the IV's init
  absl::string_view entry_extra = "";  // entry pre-init declarations
  bool check_first_slot = false;  // assert init->operand(1) is AllocateBuffer

  constexpr RawDusCase(absl::string_view name, bool expected_changed,
                       int64_t expected_buffers)
      : name(name),
        expected_changed(expected_changed),
        expected_buffers(expected_buffers) {}

  constexpr RawDusCase WithBodyDus(absl::string_view val) const {
    RawDusCase c = *this;
    c.body_dus = val;
    return c;
  }
  constexpr RawDusCase WithInitExpr(absl::string_view val) const {
    RawDusCase c = *this;
    c.init_expr = val;
    return c;
  }
  constexpr RawDusCase WithWhileExtra(absl::string_view val) const {
    RawDusCase c = *this;
    c.while_extra = val;
    return c;
  }
  constexpr RawDusCase WithTripLimit(absl::string_view val) const {
    RawDusCase c = *this;
    c.trip_limit = val;
    return c;
  }
  constexpr RawDusCase WithIvInit(absl::string_view val) const {
    RawDusCase c = *this;
    c.iv_init = val;
    return c;
  }
  constexpr RawDusCase WithEntryExtra(absl::string_view val) const {
    RawDusCase c = *this;
    c.entry_extra = val;
    return c;
  }
  constexpr RawDusCase WithCheckFirstSlot(bool val) const {
    RawDusCase c = *this;
    c.check_first_slot = val;
    return c;
  }
};

std::string MakeRawDusHlo(const RawDusCase& c) {
  return absl::StrReplaceAll(
      R"(
HloModule m

body {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[4,8] get-tuple-element(p), index=1
  update = bf16[1,8] constant({ {1,1,1,1,1,1,1,1} })
$BODY_DUS
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[4,8]) tuple(next, dus)
}

cond {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant($TRIP_LIMIT)
  ROOT lt = pred[] compare(it, limit), direction=LT
}

ENTRY e {
  z = s32[] constant(0)
$ENTRY_EXTRA$INIT_EXPR
  init = (s32[], bf16[4,8]) tuple($IV_INIT, zero_init)
  w = (s32[], bf16[4,8]) while(init), condition=cond, body=body,$WHILE_EXTRA
      backend_config={"known_trip_count":{"n":"$TRIP_LIMIT"},
                      "known_init_step":{"init":"0","step":"1"},
                      "known_induction_variable":{"tuple_index":"0"}}
  ROOT r = bf16[4,8] get-tuple-element(w), index=1
}
)",
      {{"$BODY_DUS", c.body_dus},
       {"$INIT_EXPR", c.init_expr},
       {"$WHILE_EXTRA", c.while_extra},
       {"$TRIP_LIMIT", c.trip_limit},
       {"$IV_INIT", c.iv_init},
       {"$ENTRY_EXTRA", c.entry_extra}});
}

const RawDusCase kRawDusCases[] = {
    // Default scaffold; also asserts the init becomes AllocateBuffer.
    RawDusCase("BasicAscendingRawDus", true, 1).WithCheckFirstSlot(true),

    RawDusCase("DescendingDusSubIv", true, 1)
        .WithBodyDus(R"(  z = s32[] constant(0)
  three = s32[] constant(3)
  idx = s32[] subtract(three, it)
  dus = bf16[4,8] dynamic-update-slice(acc, update, idx, z))"),

    // sub(2, it) covers [-1, 2], not [0, 4) — must skip.
    RawDusCase("DescendingWithWrongConstant", false, 0)
        .WithBodyDus(R"(  z = s32[] constant(0)
  two = s32[] constant(2)
  idx = s32[] subtract(two, it)
  dus = bf16[4,8] dynamic-update-slice(acc, update, idx, z))"),

    // Second use of `acc` forces body-gte multi-user — must skip.
    RawDusCase("MultiUserBodyGteSkipped", false, 0)
        .WithBodyDus(R"(  z = s32[] constant(0)
  dus_inner = bf16[4,8] dynamic-update-slice(acc, update, it, z)
  dus = bf16[4,8] add(dus_inner, acc))"),

    RawDusCase("NonBroadcastInitSkipped", false, 0)
        .WithInitExpr("  zero_init = bf16[4,8] parameter(0)"),

    RawDusCase("NonZeroIvInitSkipped", false, 0)
        .WithTripLimit("5")
        .WithIvInit("one_init")
        .WithEntryExtra("  one_init = s32[] constant(1)\n"),

    RawDusCase("ShardedWhileSkipped", false, 0)
        .WithWhileExtra(
            "\n      sharding={{replicated}, {devices=[2,1]<=[2]}},"),

    // Init value is irrelevant when the slot is dead-input; -inf elides too.
    RawDusCase("BroadcastNegInfInitElidable", true, 1)
        .WithInitExpr(R"(  cneg = bf16[] constant(-inf)
  zero_init = bf16[4,8] broadcast(cneg), dimensions={})"),

    RawDusCase("TightTileDim1", true, 1).WithBodyDus(R"(  z = s32[] constant(0)
  two = s32[] constant(2)
  two_it = s32[] multiply(it, two)
  update_tile = bf16[4,2] constant({ {1,1}, {1,1}, {1,1}, {1,1} })
  dus = bf16[4,8] dynamic-update-slice(acc, update_tile, z, two_it))"),
};

class DusRawDusParam : public DusAccumulatorZeroInitEliminationTest,
                       public ::testing::WithParamInterface<RawDusCase> {};

TEST_P(DusRawDusParam, ParameterisedScan) {
  const RawDusCase& c = GetParam();
  SCOPED_TRACE(c.name);
  const std::string hlo = MakeRawDusHlo(c);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EnableFlag(module.get());
  ASSERT_OK_AND_ASSIGN(bool changed,
                       DusAccumulatorZeroInitElimination().Run(module.get()));
  EXPECT_EQ(changed, c.expected_changed);
  EXPECT_EQ(CountAllocateBuffers(module.get()), c.expected_buffers);

  if (c.check_first_slot) {
    const HloInstruction* w = hlo_query::GetFirstInstructionWithOpcode(
        *module->entry_computation(), HloOpcode::kWhile);
    ASSERT_NE(w, nullptr);
    const HloInstruction* init = w->operand(0);
    ASSERT_EQ(init->opcode(), HloOpcode::kTuple);
    const HloInstruction* slot1 = init->operand(1);
    EXPECT_EQ(slot1->opcode(), HloOpcode::kCustomCall);
    EXPECT_EQ(slot1->custom_call_target(), "AllocateBuffer");
  }
}

INSTANTIATE_TEST_SUITE_P(DusAccumulatorZeroInitEliminationTest, DusRawDusParam,
                         ::testing::ValuesIn(kRawDusCases),
                         [](const ::testing::TestParamInfo<RawDusCase>& info) {
                           return std::string(info.param.name);
                         });

// Scaffold B: single-slot fusion-DUS scan with broadcast(0) init. Each Case
// declares the fusions it calls and the body line invoking them.
struct FusionDusCase {
  absl::string_view name;
  bool expected_changed;
  int64_t expected_buffers;
  absl::string_view fusion_decls;  // required: pre-body fusion decls
  absl::string_view body_dus;      // required: body's `dus = ...` line
  absl::string_view update_shape = "bf16[1,8]";
  absl::string_view update_const = "{ {1,1,1,1,1,1,1,1} }";
  absl::string_view init_expr = R"(  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] broadcast(c0), dimensions={})";
  absl::string_view while_extra = "";

  constexpr FusionDusCase(absl::string_view name, bool expected_changed,
                          int64_t expected_buffers,
                          absl::string_view fusion_decls,
                          absl::string_view body_dus)
      : name(name),
        expected_changed(expected_changed),
        expected_buffers(expected_buffers),
        fusion_decls(fusion_decls),
        body_dus(body_dus) {}

  constexpr FusionDusCase WithUpdateShape(absl::string_view val) const {
    FusionDusCase c = *this;
    c.update_shape = val;
    return c;
  }
  constexpr FusionDusCase WithUpdateConst(absl::string_view val) const {
    FusionDusCase c = *this;
    c.update_const = val;
    return c;
  }
  constexpr FusionDusCase WithInitExpr(absl::string_view val) const {
    FusionDusCase c = *this;
    c.init_expr = val;
    return c;
  }
  constexpr FusionDusCase WithWhileExtra(absl::string_view val) const {
    FusionDusCase c = *this;
    c.while_extra = val;
    return c;
  }
};

std::string MakeFusionDusHlo(const FusionDusCase& c) {
  return absl::StrReplaceAll(
      R"(
HloModule m
$FUSION_DECLS
body {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[4,8] get-tuple-element(p), index=1
  update = $UPDATE_SHAPE constant($UPDATE_CONST)
$BODY_DUS
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[4,8]) tuple(next, dus)
}

cond {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT lt = pred[] compare(it, limit), direction=LT
}

ENTRY e {
  z = s32[] constant(0)
$INIT_EXPR
  init = (s32[], bf16[4,8]) tuple(z, zero_init)
  w = (s32[], bf16[4,8]) while(init), condition=cond, body=body,$WHILE_EXTRA
      backend_config={"known_trip_count":{"n":"4"},
                      "known_init_step":{"init":"0","step":"1"},
                      "known_induction_variable":{"tuple_index":"0"}}
  ROOT r = bf16[4,8] get-tuple-element(w), index=1
}
)",
      {{"$FUSION_DECLS", c.fusion_decls},
       {"$BODY_DUS", c.body_dus},
       {"$UPDATE_SHAPE", c.update_shape},
       {"$UPDATE_CONST", c.update_const},
       {"$INIT_EXPR", c.init_expr},
       {"$WHILE_EXTRA", c.while_extra}});
}

const FusionDusCase kFusionDusCases[] = {
    FusionDusCase("FusionWrappedDus", true, 1,
                  R"(
%fused_dus (p0: bf16[4,8], p1: s32[], p2: bf16[1,8]) -> bf16[4,8] {
  %p0 = bf16[4,8] parameter(0)
  %p1 = s32[] parameter(1)
  %p2 = bf16[1,8] parameter(2)
  %z = s32[] constant(0)
  ROOT %dus = bf16[4,8] dynamic-update-slice(%p0, %p2, %p1, %z)
}
)",
                  "  dus = bf16[4,8] fusion(acc, it, update), kind=kLoop, "
                  "calls=%fused_dus"),

    FusionDusCase("PlainLoopFusionWithDescendingDus", true, 1,
                  R"(
%fused_dus (p0: bf16[4,8], p1: bf16[1,8], p2: s32[]) -> bf16[4,8] {
  %p0 = bf16[4,8] parameter(0)
  %p1 = bf16[1,8] parameter(1)
  %p2 = s32[] parameter(2)
  %three = s32[] constant(3)
  %idx = s32[] subtract(%three, %p2)
  %z = s32[] constant(0)
  ROOT %dus = bf16[4,8] dynamic-update-slice(%p0, %p1, %idx, %z)
}
)",
                  "  dus = bf16[4,8] fusion(acc, update, it), kind=kLoop, "
                  "calls=%fused_dus"),

    FusionDusCase("KInputFusionRootedAtDus", true, 1,
                  R"(
%reduce_add (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %r = f32[] add(%a, %b)
}

%fused_dus (p0: bf16[4,8], p1: f32[2,1,8], p2: s32[]) -> bf16[4,8] {
  %p0 = bf16[4,8] parameter(0)
  %p1 = f32[2,1,8] parameter(1)
  %zc = f32[] constant(0)
  %red = f32[1,8] reduce(%p1, %zc), dimensions={0}, to_apply=%reduce_add
  %cvt = bf16[1,8] convert(%red)
  %p2 = s32[] parameter(2)
  %z = s32[] constant(0)
  ROOT %dus = bf16[4,8] dynamic-update-slice(%p0, %cvt, %p2, %z)
}
)",
                  "  dus = bf16[4,8] fusion(acc, update, it), kind=kInput, "
                  "calls=%fused_dus")
        .WithUpdateShape("f32[2,1,8]")
        .WithUpdateConst("{ { {1,1,1,1,1,1,1,1} }, { {2,2,2,2,2,2,2,2} } }"),

    FusionDusCase("PlainLoopFusionRootNotDusSkipped", false, 0,
                  R"(
%fused_add (p0: bf16[4,8], p1: bf16[4,8]) -> bf16[4,8] {
  %p0 = bf16[4,8] parameter(0)
  %p1 = bf16[4,8] parameter(1)
  ROOT %a = bf16[4,8] add(%p0, %p1)
}
)",
                  R"(  c1 = bf16[] constant(1)
  ones = bf16[4,8] broadcast(c1), dimensions={}
  dus = bf16[4,8] fusion(acc, ones), kind=kLoop, calls=%fused_add)"),

    // The fusion runs DUS on a bitcast-permuted accumulator and the body
    // bitcasts the fusion output back; the walker must peel both bitcasts
    // and still recognise the inner DUS as covering.
    FusionDusCase(
        "LayoutPermutedBitcastDusFusion", true, 1,
        R"(
%fused_layout_dus (p0: bf16[4,8], p1: bf16[1,8], p2: s32[]) -> bf16[8,4] {
  %p0 = bf16[4,8] parameter(0)
  %p1 = bf16[1,8] parameter(1)
  %p2 = s32[] parameter(2)
  %p0_perm = bf16[8,4] bitcast(%p0)
  %u_perm = bf16[8,1] bitcast(%p1)
  %z = s32[] constant(0)
  ROOT %dus = bf16[8,4] dynamic-update-slice(%p0_perm, %u_perm, %z, %p2)
}
)",
        R"(  fout = bf16[8,4] fusion(acc, update, it), kind=kLoop, calls=%fused_layout_dus
  dus = bf16[4,8] bitcast(fout))"),
};

class DusFusionDusParam : public DusAccumulatorZeroInitEliminationTest,
                          public ::testing::WithParamInterface<FusionDusCase> {
};

TEST_P(DusFusionDusParam, ParameterisedScan) {
  const FusionDusCase& c = GetParam();
  SCOPED_TRACE(c.name);
  const std::string hlo = MakeFusionDusHlo(c);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EnableFlag(module.get());
  ASSERT_OK_AND_ASSIGN(bool changed,
                       DusAccumulatorZeroInitElimination().Run(module.get()));
  EXPECT_EQ(changed, c.expected_changed);
  EXPECT_EQ(CountAllocateBuffers(module.get()), c.expected_buffers);
}

INSTANTIATE_TEST_SUITE_P(
    DusAccumulatorZeroInitEliminationTest, DusFusionDusParam,
    ::testing::ValuesIn(kFusionDusCases),
    [](const ::testing::TestParamInfo<FusionDusCase>& info) {
      return std::string(info.param.name);
    });

// Scaffold C: single-slot scan whose init is built by a fusion call (possibly
// via tuple/GTE). Body may be raw or fusion DUS.
struct FusionInitCase {
  absl::string_view name;
  bool expected_changed;
  int64_t expected_buffers;
  absl::string_view fusion_decls;  // pre-body fusion decls
  absl::string_view init_expr;     // declares `zero_init` (and any `tup = ...`)
  absl::string_view body_dus = R"(  z = s32[] constant(0)
  dus = bf16[4,8] dynamic-update-slice(acc, update, it, z))";

  constexpr FusionInitCase(absl::string_view name, bool expected_changed,
                           int64_t expected_buffers,
                           absl::string_view fusion_decls,
                           absl::string_view init_expr)
      : name(name),
        expected_changed(expected_changed),
        expected_buffers(expected_buffers),
        fusion_decls(fusion_decls),
        init_expr(init_expr) {}

  constexpr FusionInitCase WithBodyDus(absl::string_view val) const {
    FusionInitCase c = *this;
    c.body_dus = val;
    return c;
  }
};

std::string MakeFusionInitHlo(const FusionInitCase& c) {
  return absl::StrReplaceAll(
      R"(
HloModule m
$FUSION_DECLS
body {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[4,8] get-tuple-element(p), index=1
  update = bf16[1,8] constant({ {1,1,1,1,1,1,1,1} })
$BODY_DUS
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[4,8]) tuple(next, dus)
}

cond {
  p = (s32[], bf16[4,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT lt = pred[] compare(it, limit), direction=LT
}

ENTRY e {
  z = s32[] constant(0)
$INIT_EXPR
  init = (s32[], bf16[4,8]) tuple(z, zero_init)
  w = (s32[], bf16[4,8]) while(init), condition=cond, body=body,
      backend_config={"known_trip_count":{"n":"4"},
                      "known_init_step":{"init":"0","step":"1"},
                      "known_induction_variable":{"tuple_index":"0"}}
  ROOT r = bf16[4,8] get-tuple-element(w), index=1
}
)",
      {{"$FUSION_DECLS", c.fusion_decls},
       {"$INIT_EXPR", c.init_expr},
       {"$BODY_DUS", c.body_dus}});
}

const FusionInitCase kFusionInitCases[] = {
    FusionInitCase("BroadcastZeroFusionInit", true, 1,
                   R"(
%fused_broadcast () -> bf16[4,8] {
  %c0 = bf16[] constant(0)
  ROOT %b = bf16[4,8] broadcast(%c0), dimensions={}
}
)",
                   "  zero_init = bf16[4,8] fusion(), kind=kLoop, "
                   "calls=%fused_broadcast"),

    FusionInitCase(
        "TupleOutputBroadcastFusionInit", true, 1,
        R"(
%fused_broadcast_tuple () -> (bf16[4,8], bf16[4,16]) {
  %c0 = bf16[] constant(0)
  %b0 = bf16[4,8] broadcast(%c0), dimensions={}
  %b1 = bf16[4,16] broadcast(%c0), dimensions={}
  ROOT %t = (bf16[4,8], bf16[4,16]) tuple(%b0, %b1)
}
)",
        R"(  tup = (bf16[4,8], bf16[4,16]) fusion(), kind=kLoop, calls=%fused_broadcast_tuple
  zero_init = bf16[4,8] get-tuple-element(tup), index=0)"),

    FusionInitCase(
        "PlainLoopFusionWithTupleBroadcastInit", true, 1,
        R"(
%fused_broadcast_tuple () -> (bf16[4,8], bf16[4,8]) {
  %c0 = bf16[] constant(0)
  %b = bf16[4,8] broadcast(%c0), dimensions={}
  %cp = bf16[4,8] copy(%b)
  ROOT %t = (bf16[4,8], bf16[4,8]) tuple(%b, %cp)
}

%fused_dus (p0: bf16[4,8], p1: bf16[1,8], p2: s32[]) -> bf16[4,8] {
  %p0 = bf16[4,8] parameter(0)
  %p1 = bf16[1,8] parameter(1)
  %p2 = s32[] parameter(2)
  %z = s32[] constant(0)
  ROOT %dus = bf16[4,8] dynamic-update-slice(%p0, %p1, %p2, %z)
}
)",
        R"(  tup = (bf16[4,8], bf16[4,8]) fusion(), kind=kLoop, calls=%fused_broadcast_tuple
  zero_init = bf16[4,8] get-tuple-element(tup), index=1)")
        .WithBodyDus("  dus = bf16[4,8] fusion(acc, update, it), kind=kLoop, "
                     "calls=%fused_dus"),
};

class DusFusionInitParam
    : public DusAccumulatorZeroInitEliminationTest,
      public ::testing::WithParamInterface<FusionInitCase> {};

TEST_P(DusFusionInitParam, ParameterisedScan) {
  const FusionInitCase& c = GetParam();
  SCOPED_TRACE(c.name);
  const std::string hlo = MakeFusionInitHlo(c);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  EnableFlag(module.get());
  ASSERT_OK_AND_ASSIGN(bool changed,
                       DusAccumulatorZeroInitElimination().Run(module.get()));
  EXPECT_EQ(changed, c.expected_changed);
  EXPECT_EQ(CountAllocateBuffers(module.get()), c.expected_buffers);
}

INSTANTIATE_TEST_SUITE_P(
    DusAccumulatorZeroInitEliminationTest, DusFusionInitParam,
    ::testing::ValuesIn(kFusionInitCases),
    [](const ::testing::TestParamInfo<FusionInitCase>& info) {
      return std::string(info.param.name);
    });

// Cases that don't fit any scaffold — inline HLO for clarity.

TEST_F(DusAccumulatorZeroInitEliminationTest, MultiSlotScanEliminatesBoth) {
  constexpr absl::string_view kHlo = R"(
HloModule m

body {
  p = (s32[], bf16[4,8], bf16[4,16]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[4,8] get-tuple-element(p), index=1
  acc1 = bf16[4,16] get-tuple-element(p), index=2
  update = bf16[1,8] constant({ {1,1,1,1,1,1,1,1} })
  update1 = bf16[1,16] constant({ {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2} })
  z = s32[] constant(0)
  dus = bf16[4,8] dynamic-update-slice(acc, update, it, z)
  dus1 = bf16[4,16] dynamic-update-slice(acc1, update1, it, z)
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[4,8], bf16[4,16]) tuple(next, dus, dus1)
}

cond {
  p = (s32[], bf16[4,8], bf16[4,16]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT lt = pred[] compare(it, limit), direction=LT
}

ENTRY e {
  z = s32[] constant(0)
  c0 = bf16[] constant(0)
  zero_init = bf16[4,8] broadcast(c0), dimensions={}
  zi1 = bf16[4,16] broadcast(c0), dimensions={}
  init = (s32[], bf16[4,8], bf16[4,16]) tuple(z, zero_init, zi1)
  w = (s32[], bf16[4,8], bf16[4,16]) while(init), condition=cond, body=body,
      backend_config={"known_trip_count":{"n":"4"},
                      "known_init_step":{"init":"0","step":"1"},
                      "known_induction_variable":{"tuple_index":"0"}}
  ROOT r = bf16[4,8] get-tuple-element(w), index=1
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EnableFlag(module.get());
  ASSERT_OK_AND_ASSIGN(bool changed,
                       DusAccumulatorZeroInitElimination().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(CountAllocateBuffers(module.get()), 2);
}

// Trip count is 4 but acc is bf16[8,8] — DUS covers only half, so skip.
TEST_F(DusAccumulatorZeroInitEliminationTest, TripCountMismatchSkipped) {
  constexpr absl::string_view kHlo = R"(
HloModule m

body {
  p = (s32[], bf16[8,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  acc = bf16[8,8] get-tuple-element(p), index=1
  update = bf16[1,8] constant({ {1,1,1,1,1,1,1,1} })
  z = s32[] constant(0)
  dus = bf16[8,8] dynamic-update-slice(acc, update, it, z)
  one = s32[] constant(1)
  next = s32[] add(it, one)
  ROOT out = (s32[], bf16[8,8]) tuple(next, dus)
}

cond {
  p = (s32[], bf16[8,8]) parameter(0)
  it = s32[] get-tuple-element(p), index=0
  limit = s32[] constant(4)
  ROOT lt = pred[] compare(it, limit), direction=LT
}

ENTRY e {
  z = s32[] constant(0)
  c0 = bf16[] constant(0)
  zero_init = bf16[8,8] broadcast(c0), dimensions={}
  init = (s32[], bf16[8,8]) tuple(z, zero_init)
  w = (s32[], bf16[8,8]) while(init), condition=cond, body=body,
      backend_config={"known_trip_count":{"n":"4"},
                      "known_init_step":{"init":"0","step":"1"},
                      "known_induction_variable":{"tuple_index":"0"}}
  ROOT r = bf16[8,8] get-tuple-element(w), index=1
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EnableFlag(module.get());
  ASSERT_OK_AND_ASSIGN(bool changed,
                       DusAccumulatorZeroInitElimination().Run(module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(CountAllocateBuffers(module.get()), 0);
}

}  // namespace
}  // namespace xla::gpu
