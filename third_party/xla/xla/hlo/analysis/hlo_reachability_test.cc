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

#include <memory>
#include <set>
#include <string>

#include "absl/random/random.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/test_benchmark.h"

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

  TF_CHECK_OK(add->AddControlDependencyTo(exp));
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

}  // namespace

class HloReachabilityMapBitSetBenchmark {
 public:
  explicit HloReachabilityMapBitSetBenchmark(int size) : a_(size), b_(size) {
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

 private:
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
