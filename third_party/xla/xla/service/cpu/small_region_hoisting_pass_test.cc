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

#include "xla/service/cpu/small_region_hoisting_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class SmallRegionHoistingPassTest : public HloHardwareIndependentTestBase {
 protected:
  // Default 64KB smallness threshold (design doc §2.3); min_region_size left at
  // its default of 4 so the cost-model size floor is exercised by the tests.
  absl::StatusOr<bool> RunPass(HloModule* module,
                               int64_t small_buffer_access_size = 1 << 16) {
    return cpu::SmallRegionHoistingPass(small_buffer_access_size).Run(module);
  }

  // Returns the single `xla_cpu_small_call`-tagged call in the module, or
  // nullptr if there is not exactly one.
  const HloInstruction* SoleSmallCall(HloModule* module) {
    const HloInstruction* found = nullptr;
    for (const HloComputation* comp : module->computations()) {
      for (const HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() != HloOpcode::kCall) continue;
        std::optional<std::string> attr =
            instr->get_frontend_attribute("xla_cpu_small_call");
        if (attr == "true") {
          if (found != nullptr) return nullptr;  // more than one
          found = instr;
        }
      }
    }
    return found;
  }

  bool CalledComputationContains(const HloInstruction* call, HloOpcode opcode) {
    return absl::c_any_of(
        call->to_apply()->instructions(),
        [&](const HloInstruction* i) { return i->opcode() == opcode; });
  }

  // Asserts that every instruction with `opcode` survives the pass and is NOT
  // inside the called computation of any `xla_cpu_small_call` (i.e., it was not
  // pulled into a single-kernel region). Unavailable ops must always stay as
  // their own thunks.
  void ExpectUnavailableOpNotHoisted(HloModule* module, HloOpcode opcode) {
    absl::flat_hash_set<const HloComputation*> hoisted_bodies;
    for (const HloComputation* comp : module->computations()) {
      for (const HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() == HloOpcode::kCall &&
            instr->get_frontend_attribute("xla_cpu_small_call") == "true") {
          hoisted_bodies.insert(instr->to_apply());
        }
      }
    }
    int found = 0;
    for (const HloComputation* comp : module->computations()) {
      for (const HloInstruction* instr : comp->instructions()) {
        if (instr->opcode() != opcode) continue;
        ++found;
        EXPECT_FALSE(hoisted_bodies.contains(comp))
            << "op " << instr->name() << " was hoisted into a small_call body";
      }
    }
    EXPECT_GT(found, 0) << "expected at least one " << HloOpcodeString(opcode);
  }
};

// ---------------------------------------------------------------------------
// Core: straight-line elementwise/dot chain collapses to one region.
// ---------------------------------------------------------------------------

TEST_F(SmallRegionHoistingPassTest, StraightLineRegionHoisting) {
  // A frag-MWE-shaped chain: two small (matvec, bias-add, tanh) layers. Every
  // op is region-eligible and the aggregate footprint is tiny, so the whole
  // entry should hoist into a single small_call kernel.
  constexpr absl::string_view hlo_string = R"(
    HloModule frag_like

    ENTRY main (x: f32[16]) -> f32[16] {
      x = f32[16]{0} parameter(0)
      c = f32[] constant(0.5)
      w = f32[16,16]{1,0} broadcast(c), dimensions={}
      b = f32[16]{0} broadcast(c), dimensions={}
      d0 = f32[16]{0} dot(w, x), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      a0 = f32[16]{0} add(d0, b)
      t0 = f32[16]{0} tanh(a0)
      d1 = f32[16]{0} dot(w, t0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      a1 = f32[16]{0} add(d1, b)
      ROOT t1 = f32[16]{0} tanh(a1)
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call = SoleSmallCall(m.get());
  ASSERT_NE(call, nullptr);
  // The region's only external input is the entry parameter `x`.
  EXPECT_EQ(call->operand_count(), 1);
  EXPECT_EQ(call->to_apply()->num_parameters(), 1);
  // The call produces the entry result.
  EXPECT_EQ(m->entry_computation()->root_instruction(), call);
  // The chain (dots + tanh) lives inside the outlined computation.
  EXPECT_TRUE(CalledComputationContains(call, HloOpcode::kDot));
  EXPECT_TRUE(CalledComputationContains(call, HloOpcode::kTanh));
}

// ---------------------------------------------------------------------------
// Liveness: a value defined outside but used inside becomes a parameter; a
// value defined inside but used after the region becomes a result.
// ---------------------------------------------------------------------------

TEST_F(SmallRegionHoistingPassTest, RegionLivenessAtCustomCallBoundary) {
  // A custom-call is an unavailable op, so it splits the entry into two
  // hoistable regions. The first region's output feeds the custom-call; the
  // second region takes the custom-call result as a parameter.
  constexpr absl::string_view hlo_string = R"(
    HloModule split_region

    ENTRY main (x: f32[16]) -> f32[16] {
      x = f32[16]{0} parameter(0)
      c = f32[] constant(2.0)
      bc = f32[16]{0} broadcast(c), dimensions={}
      // Region A (>= 4 eligible ops).
      a0 = f32[16]{0} multiply(x, bc)
      a1 = f32[16]{0} add(a0, bc)
      a2 = f32[16]{0} tanh(a1)
      a3 = f32[16]{0} subtract(a2, bc)
      // Boundary.
      cc = f32[16]{0} custom-call(a3), custom_call_target="Barrier"
      // Region B (>= 4 eligible ops), a pure chain off the custom-call result.
      b0 = f32[16]{0} multiply(cc, cc)
      b1 = f32[16]{0} add(b0, cc)
      b2 = f32[16]{0} tanh(b1)
      ROOT b3 = f32[16]{0} subtract(b2, b0)
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_TRUE(changed);

  // Two regions hoisted, custom-call left in place between them.
  int small_calls = 0;
  for (const HloComputation* comp : m->computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kCall &&
          instr->get_frontend_attribute("xla_cpu_small_call") == "true") {
        ++small_calls;
      }
    }
  }
  EXPECT_EQ(small_calls, 2);

  const HloInstruction* cc = FindInstruction(m.get(), HloOpcode::kCustomCall);
  ASSERT_NE(cc, nullptr);
  // The custom-call's operand is region A's call; its user is region B's call.
  EXPECT_EQ(cc->operand(0)->opcode(), HloOpcode::kCall);
  ASSERT_EQ(cc->user_count(), 1);
  EXPECT_EQ(cc->users()[0]->opcode(), HloOpcode::kCall);
}

TEST_F(SmallRegionHoistingPassTest, NoHoistBelowMinRegionSize) {
  // Two eligible ops, no control flow: below the min_region_size floor, the
  // single-kernel win does not beat a couple of thunks, so leave it alone.
  constexpr absl::string_view hlo_string = R"(
    HloModule tiny

    ENTRY main (x: f32[16]) -> f32[16] {
      x = f32[16]{0} parameter(0)
      c = f32[] constant(1.0)
      bc = f32[16]{0} broadcast(c), dimensions={}
      ROOT a = f32[16]{0} add(x, bc)
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SmallRegionHoistingPassTest, NoHoistWhenNotSmall) {
  // A large straight-line region exceeds the bytes-accessed threshold: keep it
  // as per-op thunks so the intra-op pool can parallelize.
  constexpr absl::string_view hlo_string = R"(
    HloModule big

    ENTRY main (x: f32[100000]) -> f32[100000] {
      x = f32[100000]{0} parameter(0)
      c = f32[] constant(1.0)
      bc = f32[100000]{0} broadcast(c), dimensions={}
      a0 = f32[100000]{0} multiply(x, bc)
      a1 = f32[100000]{0} add(a0, bc)
      a2 = f32[100000]{0} tanh(a1)
      ROOT a3 = f32[100000]{0} subtract(a2, bc)
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get(), /*=*/1 << 16));
  EXPECT_FALSE(changed);
}

// ---------------------------------------------------------------------------
// Bug-shaped regression guards (design doc §2.4 tripwires).
// ---------------------------------------------------------------------------

// jax #37465: nearest-neighbor regularization — squared first/second
// differences accumulated. An unrolled slice/subtract/square/reduce chain.
TEST_F(SmallRegionHoistingPassTest, DiffRegularization_jax37465) {
  constexpr absl::string_view hlo_string = R"(
    HloModule diff_reg

    add_fn {
      a = f32[] parameter(0)
      b = f32[] parameter(1)
      ROOT s = f32[] add(a, b)
    }

    ENTRY main (x: f32[64]) -> f32[] {
      x = f32[64]{0} parameter(0)
      zero = f32[] constant(0)
      hi = f32[63]{0} slice(x), slice={[1:64]}
      lo = f32[63]{0} slice(x), slice={[0:63]}
      d = f32[63]{0} subtract(hi, lo)
      sq = f32[63]{0} multiply(d, d)
      sum1 = f32[] reduce(sq, zero), dimensions={0}, to_apply=add_fn
      hi2 = f32[62]{0} slice(d), slice={[1:63]}
      lo2 = f32[62]{0} slice(d), slice={[0:62]}
      d2 = f32[62]{0} subtract(hi2, lo2)
      sq2 = f32[62]{0} multiply(d2, d2)
      sum2 = f32[] reduce(sq2, zero), dimensions={0}, to_apply=add_fn
      ROOT total = f32[] add(sum1, sum2)
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call = SoleSmallCall(m.get());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->operand_count(), 1);  // only `x` crosses in
  EXPECT_EQ(m->entry_computation()->root_instruction(), call);
  EXPECT_TRUE(CalledComputationContains(call, HloOpcode::kReduce));
}

// jax #26145: differentiable simulator (jaxley) — many small matvec + bias +
// nonlinearity steps, dispatch-bound on CPU.
TEST_F(SmallRegionHoistingPassTest, DiffSimulatorStep_jax26145) {
  constexpr absl::string_view hlo_string = R"(
    HloModule jaxley_step

    ENTRY main (v: f32[8], g: f32[8]) -> f32[8] {
      v = f32[8]{0} parameter(0)
      g = f32[8]{0} parameter(1)
      e = f32[] constant(-0.07)
      eb = f32[8]{0} broadcast(e), dimensions={}
      // i = g * (v - e)
      dv = f32[8]{0} subtract(v, eb)
      i = f32[8]{0} multiply(g, dv)
      // small coupling matvec
      c = f32[] constant(0.1)
      a = f32[8,8]{1,0} broadcast(c), dimensions={}
      coup = f32[8]{0} dot(a, i), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      // explicit step v' = v - dt*(i + coup)
      tot = f32[8]{0} add(i, coup)
      dt = f32[] constant(0.025)
      dtb = f32[8]{0} broadcast(dt), dimensions={}
      step = f32[8]{0} multiply(dtb, tot)
      ROOT vnext = f32[8]{0} subtract(v, step)
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call = SoleSmallCall(m.get());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->operand_count(), 2);  // v and g cross in
  EXPECT_EQ(m->entry_computation()->root_instruction(), call);
  EXPECT_TRUE(CalledComputationContains(call, HloOpcode::kDot));
}

// jax #33666: diffrax ODE integration — a small Runge-Kutta-ish body inside a
// while loop. The whole loop should hoist (the control-flow case).
TEST_F(SmallRegionHoistingPassTest, OdeIntegrationLoop_jax33666) {
  constexpr absl::string_view hlo_string = R"(
    HloModule diffrax_ode

    body {
      state = (s32[], f32[8]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      one = s32[] constant(1)
      inext = s32[] add(i, one)
      y = f32[8]{0} get-tuple-element(state), index=1
      dt = f32[] constant(0.01)
      dtb = f32[8]{0} broadcast(dt), dimensions={}
      f = f32[8]{0} negate(y)
      k = f32[8]{0} multiply(dtb, f)
      ynext = f32[8]{0} add(y, k)
      ROOT next = (s32[], f32[8]) tuple(inext, ynext)
    }

    cond {
      state = (s32[], f32[8]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(100)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    ENTRY main (y0: f32[8]) -> f32[8] {
      zero = s32[] constant(0)
      y0 = f32[8]{0} parameter(0)
      init = (s32[], f32[8]) tuple(zero, y0)
      w = (s32[], f32[8]) while(init), condition=cond, body=body
      ROOT out = f32[8]{0} get-tuple-element(w), index=1
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* call = SoleSmallCall(m.get());
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->operand_count(), 1);  // only y0 crosses in
  // The while loop is now inside the hoisted region.
  EXPECT_TRUE(CalledComputationContains(call, HloOpcode::kWhile));
}

// ---------------------------------------------------------------------------
// Ported SmallWhileLoopHoistingPass coverage: a small while is one region
// shape, and the unavailable-op rejections must still hold.
// ---------------------------------------------------------------------------

TEST_F(SmallRegionHoistingPassTest, SmallWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule simple_while_loop

    while_body {
      counter = s32[] parameter(0)
      increment = s32[] constant(1)
      ROOT incremented_counter = s32[] add(counter, increment)
    }

    while_condition {
      counter = s32[] parameter(0)
      limit = s32[] constant(10)
      ROOT less_than = pred[] compare(counter, limit), direction=LT
    }

    ENTRY main {
      initial_counter = s32[] constant(0)
      ROOT while_loop = s32[] while(initial_counter), condition=while_condition, body=while_body
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get(), /*=*/1024));
  EXPECT_TRUE(changed);

  const HloInstruction* call = SoleSmallCall(m.get());
  ASSERT_NE(call, nullptr);
  EXPECT_TRUE(CalledComputationContains(call, HloOpcode::kWhile));
}

TEST_F(SmallRegionHoistingPassTest, NoBigWhileLoopHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule simple_while_loop

    reduce_fn {
      x = s32[] parameter(0)
      y = s32[] parameter(1)
      ROOT add = s32[] add(x, y)
    }

    while_body {
      counter = s32[] parameter(0)
      dummy_constant = s32[1000000] constant({...})
      element_reduce = s32[] reduce(dummy_constant, counter), dimensions={0}, to_apply=reduce_fn
      ROOT incremented_counter = s32[] add(counter, element_reduce)
    }

    while_condition {
      counter = s32[] parameter(0)
      limit = s32[] constant(10)
      ROOT less_than = pred[] compare(counter, limit), direction=LT
    }

    ENTRY main {
      initial_counter = s32[] constant(0)
      ROOT while_loop = s32[] while(initial_counter), condition=while_condition, body=while_body
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get(), /*=*/1024));
  EXPECT_FALSE(changed);
}

TEST_F(SmallRegionHoistingPassTest, NoInOutFeedHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule in_out_feed_while_loop, entry_computation_layout={(pred[])->(pred[])}

    body_fn (T.4: (pred[])) -> (pred[]) {
      T.4 = (pred[]) parameter(0)
      after-all.5 = token[] after-all()
      infeed.6 = ((f32[1,3]{1,0}, pred[], u32[]), token[]) infeed(token[] after-all.5)
      get-tuple-element.7 = token[] get-tuple-element(((f32[1,3]{1,0}, pred[], u32[]), token[]) infeed.6), index=1
      get-tuple-element.8 = (f32[1,3]{1,0}, pred[], u32[]) get-tuple-element(((f32[1,3]{1,0}, pred[], u32[]), token[]) infeed.6), index=0
      get-tuple-element.11 = f32[1,3]{1,0} get-tuple-element((f32[1,3]{1,0}, pred[], u32[]) get-tuple-element.8), index=0
      constant.12 = f32[] constant(1)
      broadcast.13 = f32[1,3]{1,0} broadcast(f32[] constant.12), dimensions={}
      multiply.14 = f32[1,3]{1,0} multiply(f32[1,3]{1,0} get-tuple-element.11, f32[1,3]{1,0} broadcast.13)
      concatenate.15 = f32[1,6]{1,0} concatenate(f32[1,3]{1,0} multiply.14, f32[1,3]{1,0} multiply.14), dimensions={1}
      get-tuple-element.10 = u32[] get-tuple-element((f32[1,3]{1,0}, pred[], u32[]) get-tuple-element.8), index=2
      tuple.16 = (f32[1,6]{1,0}, u32[]) tuple(f32[1,6]{1,0} concatenate.15, u32[] get-tuple-element.10)
      after-all.17 = token[] after-all()
      outfeed.18 = token[] outfeed((f32[1,6]{1,0}, u32[]) tuple.16, token[] after-all.17), outfeed_shape=(f32[1,6]{1,0}, u32[])
      tuple.19 = () tuple()
      get-tuple-element.9 = pred[] get-tuple-element((f32[1,3]{1,0}, pred[], u32[]) get-tuple-element.8), index=1
      ROOT tuple.20 = (pred[]) tuple(pred[] get-tuple-element.9)
    }

    condition_fn (T.22: (pred[])) -> pred[] {
      T.22 = (pred[]) parameter(0)
      ROOT get-tuple-element.23 = pred[] get-tuple-element((pred[]) T.22), index=0
    }

    ENTRY main (prev0.1: pred[]) -> (pred[]) {
      prev0.1 = pred[] parameter(0)
      tuple.2 = (pred[]) tuple(pred[] prev0.1)
      ROOT tuple.26 = (pred[]) while((pred[]) tuple.2), condition=condition_fn, body=body_fn
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get(), /*=*/1024));
  EXPECT_TRUE(changed);
  // The pass may collapse available runs inside the body, but infeed/outfeed
  // must never be pulled into a hoisted region.
  ExpectUnavailableOpNotHoisted(m.get(), HloOpcode::kInfeed);
  ExpectUnavailableOpNotHoisted(m.get(), HloOpcode::kOutfeed);
}

// ---------------------------------------------------------------------------
// jax #26145 (jaxley): a long while body with a scatter in the middle. The
// non-scatter runs on either side of the scatter must each collapse into a
// small_call INSIDE the body, while the scatter stays un-hoisted between them.
// This is the path-(b) case the pass must cover: partition regions inside
// while/conditional bodies, routing around scatters.
// ---------------------------------------------------------------------------

TEST_F(SmallRegionHoistingPassTest, ScatterSplitInsideWhileBody_jax26145) {
  constexpr absl::string_view hlo_string = R"(
    HloModule jaxley_scatter_body

    update_fn {
      lhs = f32[] parameter(0)
      ROOT rhs = f32[] parameter(1)
    }

    body {
      state = (s32[], f32[8]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      one = s32[] constant(1)
      inext = s32[] add(i, one)
      y = f32[8]{0} get-tuple-element(state), index=1
      c = f32[] constant(1.1)
      cb = f32[8]{0} broadcast(c), dimensions={}
      // Region A: >= 4 available ops feeding the scatter operand.
      a0 = f32[8]{0} multiply(y, cb)
      a1 = f32[8]{0} add(a0, cb)
      a2 = f32[8]{0} tanh(a1)
      a3 = f32[8]{0} subtract(a2, cb)
      // Boundary: scatter is unavailable on the legacy call emitter.
      idx = s32[1]{0} constant({3})
      upd = f32[1]{0} constant({0.5})
      sc = f32[8]{0} scatter(a3, idx, upd), update_window_dims={},
          inserted_window_dims={0}, scatter_dims_to_operand_dims={0},
          index_vector_dim=1, to_apply=update_fn
      // Region B: >= 4 available ops consuming the scatter result.
      b0 = f32[8]{0} multiply(sc, cb)
      b1 = f32[8]{0} add(b0, cb)
      b2 = f32[8]{0} tanh(b1)
      ynext = f32[8]{0} subtract(b2, cb)
      ROOT next = (s32[], f32[8]) tuple(inext, ynext)
    }

    cond {
      state = (s32[], f32[8]) parameter(0)
      i = s32[] get-tuple-element(state), index=0
      limit = s32[] constant(50)
      ROOT lt = pred[] compare(i, limit), direction=LT
    }

    ENTRY main (y0: f32[8]) -> f32[8] {
      zero = s32[] constant(0)
      y0 = f32[8]{0} parameter(0)
      init = (s32[], f32[8]) tuple(zero, y0)
      w = (s32[], f32[8]) while(init), condition=cond, body=body
      ROOT out = f32[8]{0} get-tuple-element(w), index=1
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  // Large smallness threshold so both body runs hoist; the scatter must not.
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get(), /*=*/1 << 20));
  EXPECT_TRUE(changed);

  // The while body itself must contain >= 2 small_call kCalls.
  const HloInstruction* while_instr =
      FindInstruction(m.get(), HloOpcode::kWhile);
  ASSERT_NE(while_instr, nullptr);
  const HloComputation* body = while_instr->while_body();
  int body_small_calls = 0;
  for (const HloInstruction* instr : body->instructions()) {
    if (instr->opcode() == HloOpcode::kCall &&
        instr->get_frontend_attribute("xla_cpu_small_call") == "true") {
      ++body_small_calls;
    }
  }
  EXPECT_GE(body_small_calls, 2);

  // The scatter must remain in the body, un-hoisted, between the two calls.
  // Its operand is produced by region A's call (directly, or via a
  // get-tuple-element if A has multiple live-out values); its result is
  // consumed by region B's call (likewise possibly through a tuple). In all
  // cases the scatter is sandwiched between two small_call kCalls and is not
  // itself hoisted.
  const HloInstruction* scatter = FindInstruction(m.get(), HloOpcode::kScatter);
  ASSERT_NE(scatter, nullptr);
  EXPECT_EQ(scatter->parent(), body);

  auto traces_to_small_call = [](const HloInstruction* instr) {
    if (instr->opcode() == HloOpcode::kGetTupleElement) {
      instr = instr->operand(0);
    }
    return instr->opcode() == HloOpcode::kCall &&
           instr->get_frontend_attribute("xla_cpu_small_call") == "true";
  };
  EXPECT_TRUE(traces_to_small_call(scatter->operand(0)));
  ASSERT_EQ(scatter->user_count(), 1);
  EXPECT_TRUE(traces_to_small_call(scatter->users()[0]));
}

TEST_F(SmallRegionHoistingPassTest, NoFftHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule fft_module

    %body_comp (arg_tuple.3: (s32[], c64[30])) -> (s32[], c64[30]) {
      %arg_tuple.3 = (s32[], c64[30]{0}) parameter(0)
      %get-tuple-element.4 = s32[] get-tuple-element(%arg_tuple.3), index=0
      %constant.6 = s32[] constant(1)
      %add.14 = s32[] add(%get-tuple-element.4, %constant.6)
      %get-tuple-element.5 = c64[30]{0} get-tuple-element(%arg_tuple.3), index=1
      %fft.10 = c64[30]{0} fft(%get-tuple-element.5), fft_type=FFT, fft_length={30}
      ROOT %tuple.15 = (s32[], c64[30]{0}) tuple(%add.14, %get-tuple-element.5)
    }

    %condition_comp (arg_tuple.17: (s32[], c64[30])) -> pred[] {
      %arg_tuple.17 = (s32[], c64[30]{0}) parameter(0)
      %get-tuple-element.18 = s32[] get-tuple-element(%arg_tuple.17), index=0
      %constant.20 = s32[] constant(10)
      ROOT %lt.21 = pred[] compare(%get-tuple-element.18, %constant.20), direction=LT
    }

    ENTRY %main.27 (args_0_.1: c64[30]) -> c64[30] {
      %constant.2 = s32[] constant(0)
      %args_0_.1 = c64[30]{0} parameter(0)
      %while.23 = (s32[], c64[30]{0}) tuple(%constant.2, %args_0_.1)
      %while.24 = (s32[], c64[30]{0}) while(%while.23), condition=%condition_comp, body=%body_comp
      %while.25 = s32[] get-tuple-element(%while.24), index=0
      ROOT %while.26 = c64[30]{0} get-tuple-element(%while.24), index=1
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get(), /*=*/1024));
  EXPECT_TRUE(changed);
  // The pass may now collapse available runs inside the body around the fft,
  // but the fft itself must never be pulled into a hoisted region.
  ExpectUnavailableOpNotHoisted(m.get(), HloOpcode::kFft);
}

// ---------------------------------------------------------------------------
// Hardening: token ordering and control dependencies.
// ---------------------------------------------------------------------------

// When the pass hoists an available run that sits between an infeed and an
// outfeed inside a while body, the side-effecting ops stay un-hoisted AND the
// token chain must remain structurally valid (the post-pass verifier checks
// token threading through the new small_call boundaries).
TEST_F(SmallRegionHoistingPassTest, TokenThreadingValidAfterBodyHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule tok_body

    body {
      st = (s32[], f32[8]) parameter(0)
      i = s32[] get-tuple-element(st), index=0
      one = s32[] constant(1)
      inext = s32[] add(i, one)
      v = f32[8] get-tuple-element(st), index=1
      tok0 = token[] after-all()
      infd = (f32[8], token[]) infeed(tok0)
      data = f32[8] get-tuple-element(infd), index=0
      itok = token[] get-tuple-element(infd), index=1
      a0 = f32[8] add(v, data)
      a1 = f32[8] multiply(a0, data)
      a2 = f32[8] tanh(a1)
      a3 = f32[8] subtract(a2, v)
      otok = token[] outfeed(a3, itok), outfeed_shape=f32[8]
      ROOT next = (s32[], f32[8]) tuple(inext, a3)
    }

    cond {
      st = (s32[], f32[8]) parameter(0)
      i = s32[] get-tuple-element(st), index=0
      lim = s32[] constant(5)
      ROOT lt = pred[] compare(i, lim), direction=LT
    }

    ENTRY main (v0: f32[8]) -> f32[8] {
      z = s32[] constant(0)
      v0 = f32[8] parameter(0)
      init = (s32[], f32[8]) tuple(z, v0)
      w = (s32[], f32[8]) while(init), condition=cond, body=body
      ROOT out = f32[8] get-tuple-element(w), index=1
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_TRUE(changed);  // the a0..a3 run between the feeds hoists
  // Side-effecting ops are never pulled into a kernel.
  ExpectUnavailableOpNotHoisted(m.get(), HloOpcode::kInfeed);
  ExpectUnavailableOpNotHoisted(m.get(), HloOpcode::kOutfeed);
  // Token threading must still be structurally valid after outlining.
  ASSERT_TRUE(verifier().Run(m.get()).status().ok());
}

// A control dependency crossing a region boundary must block hoisting (the
// pass skips the region rather than silently dropping the edge).
TEST_F(SmallRegionHoistingPassTest, CrossingControlDepBlocksHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule cdep

    ENTRY main (x: f32[16]) -> f32[16] {
      x = f32[16]{0} parameter(0)
      c = f32[] constant(2.0)
      bc = f32[16]{0} broadcast(c), dimensions={}
      a0 = f32[16]{0} multiply(x, bc)
      a1 = f32[16]{0} add(a0, bc)
      a2 = f32[16]{0} tanh(a1)
      a3 = f32[16]{0} subtract(a2, bc)
      cc = f32[16]{0} custom-call(a3), custom_call_target="Barrier"
      b0 = f32[16]{0} multiply(cc, cc)
      b1 = f32[16]{0} add(b0, cc)
      b2 = f32[16]{0} tanh(b1)
      ROOT b3 = f32[16]{0} subtract(b2, b0)
    }
    )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  // Add a control edge from region A (a0) to region B (b0), crossing both
  // regions and the custom-call boundary.
  HloInstruction* a0 = m->entry_computation()->GetInstructionWithName("a0");
  HloInstruction* b0 = m->entry_computation()->GetInstructionWithName("b0");
  ASSERT_NE(a0, nullptr);
  ASSERT_NE(b0, nullptr);
  ASSERT_TRUE(a0->AddControlDependencyTo(b0).ok());

  ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  // Both regions carry a boundary-crossing control dep, so neither hoists.
  EXPECT_FALSE(changed);
  EXPECT_EQ(SoleSmallCall(m.get()), nullptr);
}

TEST_F(SmallRegionHoistingPassTest, StraightLineSizeCapSplitsOversizedRegion) {
  std::string hlo_string = R"(
    HloModule long_chain
    ENTRY main (p: f32[16]) -> f32[16] {
      p = f32[16]{0} parameter(0)
  )";
  for (int i = 0; i < 50; ++i) {
    std::string operand = (i == 0) ? "p" : "v" + std::to_string(i - 1);
    hlo_string += "      v" + std::to_string(i) +
                  " = f32[16]{0} add(f32[16]{0} " + operand +
                  ", f32[16]{0} p)\n";
  }
  hlo_string +=
      "      ROOT root = f32[16]{0} add(f32[16]{0} v49, f32[16]{0} p)\n    }\n";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, RunPass(m.get()));
  EXPECT_TRUE(changed);

  int small_call_count = 0;
  for (const HloInstruction* instr : m->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kCall &&
        instr->get_frontend_attribute("xla_cpu_small_call") == "true") {
      small_call_count++;
    }
  }
  EXPECT_EQ(small_call_count, 2);
}

TEST_F(SmallRegionHoistingPassTest, ModelSizeGatePreventsHoisting) {
  constexpr absl::string_view hlo_string = R"(
    HloModule simple_while_loop

    body_comp {
      counter = s32[] parameter(0)
      increment = s32[] constant(1)
      ROOT incremented_counter = s32[] add(counter, increment)
    }

    condition_comp {
      counter = s32[] parameter(0)
      limit = s32[] constant(10)
      ROOT less_than = pred[] compare(counter, limit), direction=LT
    }

    ENTRY main {
      initial_counter = s32[] constant(0)
      ROOT while_loop = s32[] while(initial_counter), condition=condition_comp, body=body_comp
    }
  )";

  // Test 1: threshold = 3000. 2500 instructions inserted. Hoisting should
  // occur.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    HloComputation* entry = m->entry_computation();
    for (int i = 0; i < 2500; ++i) {
      entry->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(i)));
    }
    ASSERT_OK_AND_ASSIGN(bool changed, cpu::SmallRegionHoistingPass(
                                           /*small_buffer_access_size=*/1024,
                                           /*min_region_size=*/4,
                                           /*max_instruction_count=*/3000)
                                           .Run(m.get()));
    EXPECT_TRUE(changed);
  }

  // Test 2: threshold = 2000. 2500 instructions inserted. Hoisting should be
  // skipped.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    HloComputation* entry = m->entry_computation();
    for (int i = 0; i < 2500; ++i) {
      entry->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(i)));
    }
    ASSERT_OK_AND_ASSIGN(bool changed, cpu::SmallRegionHoistingPass(
                                           /*small_buffer_access_size=*/1024,
                                           /*min_region_size=*/4,
                                           /*max_instruction_count=*/2000)
                                           .Run(m.get()));
    EXPECT_FALSE(changed);
  }
}

TEST_F(SmallRegionHoistingPassTest, CandidateCountCapGatesHoisting) {
  // A module containing 3 candidate regions separated by custom-calls.
  constexpr absl::string_view hlo_string = R"(
    HloModule three_regions

    ENTRY main (x: f32[16], y: f32[16]) -> f32[16] {
      x = f32[16]{0} parameter(0)
      y = f32[16]{0} parameter(1)

      // Region A (4 available instructions)
      a0 = f32[16]{0} add(x, y)
      a1 = f32[16]{0} tanh(a0)
      a2 = f32[16]{0} add(a1, y)
      a3 = f32[16]{0} tanh(a2)

      // Boundary 1
      cc1 = f32[16]{0} custom-call(a3), custom_call_target="Barrier"

      // Region B (4 available instructions)
      b0 = f32[16]{0} add(cc1, y)
      b1 = f32[16]{0} tanh(b0)
      b2 = f32[16]{0} add(b1, y)
      b3 = f32[16]{0} tanh(b2)

      // Boundary 2
      cc2 = f32[16]{0} custom-call(b3), custom_call_target="Barrier"

      // Region C (4 available instructions)
      c0 = f32[16]{0} add(cc2, y)
      c1 = f32[16]{0} tanh(c0)
      c2 = f32[16]{0} add(c1, y)
      ROOT c3 = f32[16]{0} tanh(c2)
    }
  )";

  // Test 1: limit = 2.
  // The module has 3 candidate regions, which exceeds the limit, so hoisting is
  // disabled.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    ASSERT_OK_AND_ASSIGN(bool changed, cpu::SmallRegionHoistingPass(
                                           /*small_buffer_access_size=*/1 << 16,
                                           /*min_region_size=*/4,
                                           /*max_instruction_count=*/2000,
                                           /*max_regions_limit=*/2)
                                           .Run(m.get()));
    EXPECT_FALSE(changed);
  }

  // Test 2: limit = 3.
  // The candidate count is exactly at the limit, so hoisting should proceed.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    ASSERT_OK_AND_ASSIGN(bool changed, cpu::SmallRegionHoistingPass(
                                           /*small_buffer_access_size=*/1 << 16,
                                           /*min_region_size=*/4,
                                           /*max_instruction_count=*/2000,
                                           /*max_regions_limit=*/3)
                                           .Run(m.get()));
    EXPECT_TRUE(changed);
  }
}

}  // namespace
}  // namespace xla
