/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/analysis/hlo_reachability.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/random/random.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/service/device_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

class HloReachabilityTest : public HloHardwareIndependentTestBase {};

TEST_F(HloReachabilityTest, Reachability) {
  // Construct and test a reachability graph of the following form:
  /*
       a
      / \
     b   c
      \ / \
       d   e
  */
  auto builder = HloComputation::Builder(TestName());
  auto a = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto b = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto c = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto d = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto e = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0.0f)));
  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  HloReachabilityMap reachability({a, b, c, d, e});
  reachability.SetReachable(a, a);
  EXPECT_TRUE(reachability.SetReachabilityToUnion({a}, b));
  EXPECT_TRUE(reachability.SetReachabilityToUnion({a}, c));
  EXPECT_TRUE(reachability.SetReachabilityToUnion({b, c}, d));
  EXPECT_TRUE(reachability.SetReachabilityToUnion({c}, e));

  EXPECT_TRUE(reachability.IsReachable(a, a));
  EXPECT_TRUE(reachability.IsReachable(a, b));
  EXPECT_TRUE(reachability.IsReachable(a, c));
  EXPECT_TRUE(reachability.IsReachable(a, d));
  EXPECT_TRUE(reachability.IsReachable(a, e));

  EXPECT_FALSE(reachability.IsReachable(b, a));
  EXPECT_TRUE(reachability.IsReachable(b, b));
  EXPECT_FALSE(reachability.IsReachable(b, c));
  EXPECT_TRUE(reachability.IsReachable(b, d));
  EXPECT_FALSE(reachability.IsReachable(b, e));

  EXPECT_FALSE(reachability.IsReachable(e, a));
  EXPECT_FALSE(reachability.IsReachable(e, b));
  EXPECT_FALSE(reachability.IsReachable(e, c));
  EXPECT_FALSE(reachability.IsReachable(e, d));
  EXPECT_TRUE(reachability.IsReachable(e, e));

  // Recomputing the same reachability for a previously computed instruction
  // should return false (no change).
  EXPECT_FALSE(reachability.SetReachabilityToUnion({a}, b));
  EXPECT_FALSE(reachability.SetReachabilityToUnion({b, c}, d));
}

TEST_F(HloReachabilityTest, NonTrivialReachability) {
  // Test reachability of a non-trivial computation:
  //
  // const1    const2
  //    |         |
  //    | +-------+
  //    | |       |
  //    add ..   negate
  //     |   .     |
  //     |   .... exp
  //     |         |
  //     +---+   +-+---+
  //         |   |     |
  //       multiply   copy
  //
  // There is a control dependency from 'add' to 'exp'.
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder(TestName());
  auto constant1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto constant2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      r0f32, HloOpcode::kAdd, constant1, constant2));
  auto negate = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, constant2));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, negate));
  auto mul = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kMultiply, add, exp));
  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kCopy, exp));

  auto module = CreateNewVerifiedModule();
  auto computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/mul));

  CHECK_OK(add->AddControlDependencyTo(exp));
  auto reachability = HloReachabilityMap::Build(computation);

  EXPECT_TRUE(reachability->IsReachable(constant1, constant1));
  EXPECT_FALSE(reachability->IsReachable(constant1, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant1, add));
  EXPECT_FALSE(reachability->IsReachable(constant1, negate));
  EXPECT_TRUE(reachability->IsReachable(constant1, exp));
  EXPECT_TRUE(reachability->IsReachable(constant1, mul));
  EXPECT_TRUE(reachability->IsReachable(constant1, copy));

  EXPECT_FALSE(reachability->IsReachable(constant2, constant1));
  EXPECT_TRUE(reachability->IsReachable(constant2, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant2, add));
  EXPECT_TRUE(reachability->IsReachable(constant2, negate));
  EXPECT_TRUE(reachability->IsReachable(constant2, exp));
  EXPECT_TRUE(reachability->IsReachable(constant2, mul));
  EXPECT_TRUE(reachability->IsReachable(constant2, copy));

  EXPECT_FALSE(reachability->IsReachable(exp, constant1));
  EXPECT_FALSE(reachability->IsReachable(exp, constant2));
  EXPECT_FALSE(reachability->IsReachable(exp, add));
  EXPECT_FALSE(reachability->IsReachable(exp, negate));
  EXPECT_TRUE(reachability->IsReachable(exp, exp));
  EXPECT_TRUE(reachability->IsReachable(exp, mul));
  EXPECT_TRUE(reachability->IsReachable(exp, copy));

  EXPECT_FALSE(reachability->IsReachable(mul, constant1));
  EXPECT_FALSE(reachability->IsReachable(mul, constant2));
  EXPECT_FALSE(reachability->IsReachable(mul, add));
  EXPECT_FALSE(reachability->IsReachable(mul, negate));
  EXPECT_FALSE(reachability->IsReachable(mul, exp));
  EXPECT_TRUE(reachability->IsReachable(mul, mul));
  EXPECT_FALSE(reachability->IsReachable(mul, copy));

  EXPECT_TRUE(reachability->IsConnected(constant1, copy));
  EXPECT_TRUE(reachability->IsConnected(copy, constant1));
  EXPECT_FALSE(reachability->IsConnected(negate, add));
  EXPECT_FALSE(reachability->IsConnected(add, negate));

  // Remove the control dependency then update and verify the reachability map
  ASSERT_IS_OK(add->RemoveControlDependencyTo(exp));
  reachability->UpdateReachabilityThroughInstruction(exp);

  EXPECT_TRUE(reachability->IsReachable(constant1, constant1));
  EXPECT_FALSE(reachability->IsReachable(constant1, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant1, add));
  EXPECT_FALSE(reachability->IsReachable(constant1, negate));
  EXPECT_FALSE(reachability->IsReachable(constant1, exp));
  EXPECT_TRUE(reachability->IsReachable(constant1, mul));
  EXPECT_FALSE(reachability->IsReachable(constant1, copy));

  // Change a use within the graph then update and verify the reachability map
  ASSERT_IS_OK(constant2->ReplaceUseWith(negate, constant1));
  reachability->UpdateReachabilityThroughInstruction(negate);

  EXPECT_FALSE(reachability->IsReachable(constant2, constant1));
  EXPECT_TRUE(reachability->IsReachable(constant2, constant2));
  EXPECT_TRUE(reachability->IsReachable(constant2, add));
  EXPECT_FALSE(reachability->IsReachable(constant2, negate));
  EXPECT_FALSE(reachability->IsReachable(constant2, exp));
  EXPECT_TRUE(reachability->IsReachable(constant2, mul));
  EXPECT_FALSE(reachability->IsReachable(constant2, copy));
}

TEST_F(HloReachabilityTest, ChannelReachability) {
  const Shape shape = ShapeUtil::MakeShape(F32, {5, 7});
  HloComputation::Builder builder("ChannelReachability");
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto token0 = builder.AddInstruction(HloInstruction::CreateToken());
  auto send = builder.AddInstruction(HloInstruction::CreateSend(
      param, token0, /*channel_id=*/1, /*is_host_transfer=*/false));
  auto send_done = builder.AddInstruction(HloInstruction::CreateSendDone(
      send, send->channel_id(), /*is_host_transfer=*/false));
  auto token1 = builder.AddInstruction(HloInstruction::CreateToken());
  auto recv = builder.AddInstruction(HloInstruction::CreateRecv(
      shape, token1, /*channel_id=*/1, /*is_host_transfer=*/false));
  auto recv_done = builder.AddInstruction(HloInstruction::CreateRecvDone(
      recv, recv->channel_id(), /*is_host_transfer=*/false));

  auto module = CreateNewVerifiedModule();
  module->mutable_config().set_use_spmd_partitioning(false);
  module->mutable_config().set_static_device_assignment(DeviceAssignment(1, 2));
  auto computation = module->AddEntryComputation(builder.Build(recv_done));
  auto reachability = HloReachabilityMap::Build(computation);
  EXPECT_FALSE(reachability->IsReachable(param, recv_done));
  EXPECT_FALSE(reachability->IsReachable(send, recv));
  EXPECT_FALSE(reachability->IsReachable(send_done, recv));
}

TEST_F(HloReachabilityTest, ReplaceInstructions) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY entry {
      p0 = f32[28,28]{1,0} parameter(0)
      ROOT add = f32[28,28]{1,0} add(p0, p0)
    })")
                    .value();
  auto computation = module->entry_computation();
  auto reachability = HloReachabilityMap::Build(computation);
  auto* add = module->entry_computation()->root_instruction();
  auto* p0 = add->operand(0);
  EXPECT_TRUE(reachability->IsReachable(p0, add));

  // Replacing an instruction with itself is a noop.
  reachability->Replace(add, add);
  EXPECT_TRUE(reachability->IsReachable(p0, add));

  // Introduce a fusion instruction taking the place of `add`.
  auto* fusion = computation->AddInstruction(HloInstruction::CreateFusion(
      add->shape(), HloInstruction::FusionKind::kLoop, add));
  EXPECT_FALSE(reachability->IsPresent(fusion));
  EXPECT_TRUE(reachability->IsReachable(p0, add));

  // Replace `add` with `fusion` in the readability map.
  reachability->Replace(add, fusion);
  EXPECT_FALSE(reachability->IsPresent(add));
  EXPECT_TRUE(reachability->IsReachable(p0, fusion));
}

TEST_F(HloReachabilityTest, UpdateMultipleInstructions) {
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  auto builder = HloComputation::Builder(TestName());
  auto a = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto b = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  auto c = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kNegate, a));
  auto d = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, b));
  auto e = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kCopy, c));
  auto f = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kCopy, d));

  auto module = CreateNewVerifiedModule();
  auto computation =
      module->AddEntryComputation(builder.Build(/*root_instruction=*/f));

  auto reachability = HloReachabilityMap::Build(computation);

  EXPECT_TRUE(reachability->IsReachable(a, c));
  EXPECT_TRUE(reachability->IsReachable(c, e));
  EXPECT_TRUE(reachability->IsReachable(a, e));

  EXPECT_FALSE(reachability->IsReachable(b, c));
  EXPECT_FALSE(reachability->IsReachable(b, e));
  EXPECT_FALSE(reachability->IsReachable(d, e));
  EXPECT_FALSE(reachability->IsReachable(a, d));
  EXPECT_FALSE(reachability->IsReachable(a, f));

  // Add a control dependency from b to c, and d to e.
  ASSERT_IS_OK(b->AddControlDependencyTo(c));
  ASSERT_IS_OK(d->AddControlDependencyTo(e));

  absl::flat_hash_map<const HloInstruction*,
                      absl::flat_hash_set<const HloInstruction*>>
      to_update;
  to_update[c].insert(b);
  to_update[e].insert(d);

  reachability->UpdateMultipleInstructions(to_update);

  // Now b should be reachable to c, e
  EXPECT_TRUE(reachability->IsReachable(b, c));
  EXPECT_TRUE(reachability->IsReachable(b, e));

  // d should be reachable to e
  EXPECT_TRUE(reachability->IsReachable(d, e));

  // a is still reachable to c, e
  EXPECT_TRUE(reachability->IsReachable(a, c));
  EXPECT_TRUE(reachability->IsReachable(a, e));

  // a is still not reachable to d, f
  EXPECT_FALSE(reachability->IsReachable(a, d));
  EXPECT_FALSE(reachability->IsReachable(a, f));
}

// Expects `reachability` to answer every query among `instructions` like a
// map built from scratch for the current graph of `computation`.
void ExpectMatchesRebuiltMap(const HloReachabilityMap& reachability,
                             const HloComputation* computation,
                             absl::Span<HloInstruction* const> instructions) {
  std::unique_ptr<HloReachabilityMap> rebuilt =
      HloReachabilityMap::Build(computation);
  for (const HloInstruction* a : instructions) {
    for (const HloInstruction* b : instructions) {
      EXPECT_EQ(reachability.IsReachable(a, b), rebuilt->IsReachable(a, b))
          << a->name() << " -> " << b->name();
    }
  }
}

TEST_F(HloReachabilityTest, UpdateMultipleInstructionsMatchesRebuiltMap) {
  // Two subgraphs joined only by the control edge c2 -> g, the first one a
  // lattice of diamonds:
  //
  //   p -> a1, a2, a3;  a1, a2 -> b1;  a2, a3 -> b2;  b1 -> c1;  b2 -> c2;
  //   c1, c2 -> d -> e;  q -> f1 -> f2;  q -> g;  c2 -> g (control)
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY entry {
      p = f32[] parameter(0)
      q = f32[] parameter(1)
      a1 = f32[] negate(p)
      a2 = f32[] exponential(p)
      a3 = f32[] abs(p)
      b1 = f32[] add(a1, a2)
      b2 = f32[] multiply(a2, a3)
      c1 = f32[] negate(b1)
      c2 = f32[] exponential(b2)
      d = f32[] add(c1, c2)
      e = f32[] negate(d)
      f1 = f32[] negate(q)
      f2 = f32[] exponential(f1)
      g = f32[] abs(q), control-predecessors={c2}
      ROOT t = (f32[], f32[], f32[]) tuple(e, f2, g)
    })"));
  HloComputation* computation = module->entry_computation();
  const std::vector<HloInstruction*> instructions =
      computation->MakeInstructionPostOrder();
  auto instruction = [&](absl::string_view name) {
    return FindInstruction(module.get(), name);
  };
  auto reachability = HloReachabilityMap::Build(computation);

  // Both branches into the join d change, so d has two changed predecessors,
  // and g gains f1 only through the control edge from c2.
  ASSERT_IS_OK(instruction("f1")->AddControlDependencyTo(instruction("a1")));
  ASSERT_IS_OK(instruction("f1")->AddControlDependencyTo(instruction("a2")));
  EXPECT_FALSE(reachability->IsReachable(instruction("f1"), instruction("e")));
  reachability->UpdateMultipleInstructions(
      {{instruction("a1"), {instruction("f1")}},
       {instruction("a2"), {instruction("f1")}}});
  EXPECT_TRUE(reachability->IsReachable(instruction("q"), instruction("e")));
  EXPECT_TRUE(reachability->IsReachable(instruction("f1"), instruction("g")));
  EXPECT_FALSE(reachability->IsReachable(instruction("f1"), instruction("a3")));
  ExpectMatchesRebuiltMap(*reachability, computation, instructions);

  // The updated f1 is upstream of the updated f2, and a3 reaches f2 only by
  // way of f1: the row of f2's new predecessor c1 does not contain a3.
  ASSERT_IS_OK(instruction("a3")->AddControlDependencyTo(instruction("f1")));
  ASSERT_IS_OK(instruction("c1")->AddControlDependencyTo(instruction("f2")));
  EXPECT_FALSE(reachability->IsReachable(instruction("a3"), instruction("c1")));
  reachability->UpdateMultipleInstructions(
      {{instruction("f1"), {instruction("a3")}},
       {instruction("f2"), {instruction("c1")}}});
  EXPECT_TRUE(reachability->IsReachable(instruction("a3"), instruction("f2")));
  EXPECT_TRUE(reachability->IsReachable(instruction("b1"), instruction("f2")));
  EXPECT_FALSE(reachability->IsReachable(instruction("d"), instruction("f2")));
  ExpectMatchesRebuiltMap(*reachability, computation, instructions);
}

TEST_F(HloReachabilityTest,
       UpdateMultipleInstructionsLooksThroughAbsentInstructions) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test

    ENTRY entry {
      p = f32[] parameter(0)
      a = f32[] negate(p)
      a0 = f32[] abs(p)
      x = f32[] add(a, a0)
      b = f32[] negate(x)
      y = f32[] exponential(b)
      c = f32[] negate(y)
      c0 = f32[] abs(y)
      d = f32[] abs(p)
      e = f32[] negate(p)
      z = f32[] negate(e)
      w = f32[] abs(z)
      ROOT t = (f32[], f32[], f32[], f32[]) tuple(c, c0, d, w)
    })"));
  HloComputation* computation = module->entry_computation();
  auto instruction = [&](absl::string_view name) {
    return FindInstruction(module.get(), name);
  };
  // A map over the graph without x, y and z, closed through them.
  const std::vector<HloInstruction*> present = {
      instruction("p"), instruction("a"), instruction("a0"),
      instruction("b"), instruction("c"), instruction("c0"),
      instruction("d"), instruction("e"), instruction("w")};
  HloReachabilityMap reachability(present);
  reachability.SetReachabilityToUnion({instruction("p")}, instruction("a"));
  reachability.SetReachabilityToUnion({instruction("p")}, instruction("a0"));
  reachability.SetReachabilityToUnion({instruction("a"), instruction("a0")},
                                      instruction("b"));
  reachability.SetReachabilityToUnion({instruction("b")}, instruction("c"));
  reachability.SetReachabilityToUnion({instruction("b")}, instruction("c0"));
  reachability.SetReachabilityToUnion({instruction("p")}, instruction("d"));
  reachability.SetReachabilityToUnion({instruction("p")}, instruction("e"));
  reachability.SetReachabilityToUnion({instruction("e")}, instruction("w"));
  EXPECT_FALSE(reachability.IsPresent(instruction("x")));
  EXPECT_TRUE(reachability.IsReachable(instruction("a0"), instruction("c0")));

  // The updated instruction y (two present users) and the new predecessor x
  // (two present predecessors) are absent: d reaches c and c0 through y, a and
  // a0 reach e through x, and from there w through the absent z.
  ASSERT_IS_OK(instruction("d")->AddControlDependencyTo(instruction("y")));
  ASSERT_IS_OK(instruction("x")->AddControlDependencyTo(instruction("e")));
  reachability.UpdateMultipleInstructions(
      {{instruction("y"), {instruction("d")}},
       {instruction("e"), {instruction("x")}}});
  EXPECT_TRUE(reachability.IsReachable(instruction("d"), instruction("c")));
  EXPECT_TRUE(reachability.IsReachable(instruction("d"), instruction("c0")));
  EXPECT_TRUE(reachability.IsReachable(instruction("a"), instruction("e")));
  EXPECT_TRUE(reachability.IsReachable(instruction("a0"), instruction("e")));
  EXPECT_TRUE(reachability.IsReachable(instruction("a"), instruction("w")));
  EXPECT_FALSE(reachability.IsReachable(instruction("b"), instruction("e")));
  EXPECT_FALSE(reachability.IsReachable(instruction("d"), instruction("w")));
  ExpectMatchesRebuiltMap(reachability, computation, present);
}

}  // namespace

class HloReachabilityMapBitSetBenchmark {
 public:
  explicit HloReachabilityMapBitSetBenchmark(int size) {
    size_t nwords = (size + 63) / 64;
    space_.resize(2 * nwords);
    a_ = HloReachabilityMap::BitSet(&space_[0], nwords);
    b_ = HloReachabilityMap::BitSet(&space_[nwords], nwords);
    // Initialize the bit sets to random inputs. Done out of caution -- note
    // that a sufficiently smart optimizer might realize that the bit sets
    // are otherwise initialized to 0.
    absl::BitGen gen;
    for (int i = 0; i < size; ++i) {
      if (absl::Bernoulli(gen, 0.5)) a_.Set(i);
      if (absl::Bernoulli(gen, 0.5)) b_.Set(i);
    }
  }
  void Union() { a_ |= b_; }

  void OrUpdatePartial(
      const std::vector<std::pair<size_t, HloReachabilityMap::BitSet::Word>>&
          diff) {
    a_.OrUpdatePartial(diff);
  }

  std::vector<std::pair<size_t, HloReachabilityMap::BitSet::Word>> GenerateDiff(
      int num_elements) {
    std::vector<std::pair<size_t, HloReachabilityMap::BitSet::Word>> diff;
    size_t nwords = a_.NumWords();
    if (nwords == 0) {
      return diff;
    }
    absl::BitGen gen;
    if (num_elements >= nwords) {
      for (size_t i = 0; i < nwords; ++i) {
        diff.push_back(
            {i, absl::Uniform<HloReachabilityMap::BitSet::Word>(gen)});
      }
    } else {
      absl::flat_hash_set<uint64_t> indices;
      while (indices.size() < num_elements) {
        indices.insert(absl::Uniform<size_t>(gen, 0, nwords));
      }
      std::vector<uint64_t> sorted_indices(indices.begin(), indices.end());
      std::sort(sorted_indices.begin(), sorted_indices.end());
      for (uint64_t idx : sorted_indices) {
        diff.push_back(
            {idx, absl::Uniform<HloReachabilityMap::BitSet::Word>(gen)});
      }
    }
    return diff;
  }

 private:
  std::vector<uint64_t> space_;
  HloReachabilityMap::BitSet a_;
  HloReachabilityMap::BitSet b_;
};

namespace {

void BM_HloReachabilityBitSetUnion(benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  for (auto s : state) {
    bm.Union();
  }
}
#define BM_ARGS Arg(1)->Arg(64)->Arg(128)->Arg(256)->Range(512, 256 * 1024)
BENCHMARK(BM_HloReachabilityBitSetUnion)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartialRandom2Diff(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  auto diff = bm.GenerateDiff(2);
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartialRandom2Diff)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartialRandom10Diff(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  auto diff = bm.GenerateDiff(10);
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartialRandom10Diff)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial1Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords / 100));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial1Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial10Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords / 10));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial10Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial25Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords / 4));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial25Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial50Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords / 2));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial50Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartial75Percent(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  int nwords = (state.range(0) + 63) / 64;
  auto diff = bm.GenerateDiff(std::max<int>(1, nwords * 3 / 4));
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartial75Percent)->BM_ARGS;

void BM_HloReachabilityBitSetUnion_OrUpdatePartialDenseRandom(
    benchmark::State& state) {
  HloReachabilityMapBitSetBenchmark bm(state.range(0));
  auto diff = bm.GenerateDiff((state.range(0) + 63) / 64);
  for (auto s : state) {
    bm.OrUpdatePartial(diff);
  }
}
BENCHMARK(BM_HloReachabilityBitSetUnion_OrUpdatePartialDenseRandom)->BM_ARGS;

class HloReachabilityBenchmark {
 public:
  HloReachabilityBenchmark(int size, absl::string_view name) : name_(name) {
    Shape r0f32 = ShapeUtil::MakeShape(F32, {});
    auto builder = HloComputation::Builder(name);

    // Build a graph of chained Exponentials, i.e. Exp(...(Exp(Input))...).
    HloInstruction* constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
    HloInstruction* prev = constant;
    for (int i = 1; i < size; ++i) {
      prev = builder.AddInstruction(
          HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, prev));
    }

    HloModuleConfig hlo_config;
    module_ = std::make_unique<HloModule>(name_, hlo_config);
    computation_ =
        module_->AddEntryComputation(builder.Build(/*root_instruction=*/prev));
  }
  std::unique_ptr<HloReachabilityMap> Build() {
    return HloReachabilityMap::Build(computation_);
  }

 private:
  std::unique_ptr<HloModule> module_;
  HloComputation* computation_;
  const std::string name_;
};

void BM_HloReachabilityBuild(benchmark::State& state) {
  HloReachabilityBenchmark bm(state.range(0), state.name());
  for (auto s : state) {
    benchmark::DoNotOptimize(bm.Build());
  }
}
BENCHMARK(BM_HloReachabilityBuild)->BM_ARGS;

}  // namespace

}  // namespace xla
