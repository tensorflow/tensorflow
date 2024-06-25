/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_dfs_reachability.h"

#include <cstddef>
#include <set>
#include <string_view>

#include "absl/random/random.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test_benchmark.h"

namespace xla {

namespace {

class HloDfsReachabilityTest : public HloTestBase {};

TEST_F(HloDfsReachabilityTest, NonTrivialReachability) {
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
  auto reachability = HloDfsReachability::Build(computation);

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
}

TEST_F(HloDfsReachabilityTest, ChannelReachability) {
  const Shape shape = ShapeUtil::MakeShape(F32, {5, 7});
  HloComputation::Builder builder("ChannelReachability");
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  auto token0 = builder.AddInstruction(HloInstruction::CreateToken());
  auto send =
      builder.AddInstruction(HloInstruction::CreateSend(param, token0, 1));
  auto send_done = builder.AddInstruction(HloInstruction::CreateSendDone(send));
  auto token1 = builder.AddInstruction(HloInstruction::CreateToken());
  auto recv =
      builder.AddInstruction(HloInstruction::CreateRecv(shape, token1, 1));
  auto recv_done = builder.AddInstruction(HloInstruction::CreateRecvDone(recv));

  auto module = CreateNewVerifiedModule();
  module->mutable_config().set_use_spmd_partitioning(false);
  module->mutable_config().set_static_device_assignment(DeviceAssignment(1, 2));
  auto computation = module->AddEntryComputation(builder.Build(recv_done));
  auto reachability = HloDfsReachability::Build(computation);
  EXPECT_FALSE(reachability->IsReachable(param, recv_done));
  EXPECT_FALSE(reachability->IsReachable(send, recv));
  EXPECT_FALSE(reachability->IsReachable(send_done, recv));
}

class HloDfsReachabilityBenchmark {
 public:
  HloDfsReachabilityBenchmark(int size, std::string_view name) : name_(name) {
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

  std::unique_ptr<HloDfsReachability> Build() {
    return HloDfsReachability::Build(computation_);
  }

  const HloComputation* computation() { return computation_; }

 private:
  std::unique_ptr<HloModule> module_;
  HloComputation* computation_;
  const std::string name_;
};

void BM_HloDfsReachabilityBuild(benchmark::State& state) {
  int num_nodes = state.range(0);
  HloDfsReachabilityBenchmark bm(num_nodes, state.name());
  while (state.KeepRunningBatch(num_nodes)) {
    benchmark::DoNotOptimize(bm.Build());
  }
}

void BM_HloDfsReachabilityCheck(benchmark::State& state) {
  size_t size = state.range(0);

  HloDfsReachabilityBenchmark bm(size, state.name());
  auto reachability = bm.Build();
  auto instrs = bm.computation()->MakeInstructionPostOrder();

  size_t i = 0;
  for (auto s : state) {
    size_t from = i % size;
    size_t to = (++i + size / 2) % size;
    reachability->IsReachable(instrs[from], instrs[to]);
  }
}

#define BM_ARGS Arg(1)->Arg(64)->Arg(128)->Arg(256)->Range(512, 256 * 1024)
BENCHMARK(BM_HloDfsReachabilityBuild)->BM_ARGS;
BENCHMARK(BM_HloDfsReachabilityCheck)->BM_ARGS;

}  // namespace

}  // namespace xla
