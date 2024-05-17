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

#include "xla/service/hlo_rematerialization.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/layout.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_memory_scheduler.h"
#include "xla/service/hlo_rematerialization_test_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using ::testing::_;

class AsyncRematerializationTest : public RematerializationTestBase {
 protected:
  absl::StatusOr<bool> RunHloRematerialization(
      int64_t memory_limit_bytes, HloModule* module,
      const absl::flat_hash_map<HloComputation*, int64_t>&
          async_computation_parallelism,
      int64_t min_remat_size = 0) {
    TF_EXPECT_OK(verifier().Run(module).status());
    if (!module->has_schedule()) {
      HloMemoryScheduler scheduler(
          [](const BufferValue& buffer) { return ByteSizeOf(buffer.shape()); },
          ComputationSchedulerToModuleScheduler(DefaultMemoryScheduler));
      TF_EXPECT_OK(scheduler.Run(module).status());
    }
    HloRematerialization::RematerializationModeConfig config(
        /*recompute=*/true, /*compress=*/true, /*host_offload=*/false);
    auto shape_size_func = [](const Shape& shape) { return ByteSizeOf(shape); };
    HloCostAnalysis cost_analysis(shape_size_func);
    HloRematerialization::Options options(
        cost_analysis, config, memory_limit_bytes,
        /*block_size_limit=*/1, /*block_rematerialization_factor=*/1,
        min_remat_size, /*compact_shape_function=*/nullptr,
        /*host_memory_offload_config=*/std::nullopt,
        /*async_computation_parallelism=*/async_computation_parallelism);
    HloRematerialization::RematerializationSizes sizes;
    HloRematerialization remat(options, sizes);
    return remat.Run(module, {HloInstruction::kMainExecutionThread});
  }

  static constexpr int64_t kNumParallelThreads = 16;
};

TEST_F(AsyncRematerializationTest, AsyncComputation) {
  constexpr std::string_view hlo = R"(
HloModule async, is_scheduled=true

%offload_computation {
  %param = f32[1]{0} parameter(0)
  %reshape = f32[] reshape(f32[1]{0} %param)
  %broadcast = f32[1024]{0} broadcast(f32[] %reshape), dimensions={}
  %negate = f32[1024]{0} negate(f32[1024]{0} %broadcast)
  %concatenate = f32[2048]{0} concatenate(f32[1024]{0} %negate, f32[1024]{0} %negate), dimensions={0}
  %slice = f32[1]{0} slice(f32[2048]{0} %concatenate), slice={[0:1]}
  %concatenate.1 = f32[1025]{0} concatenate(f32[1024]{0} %broadcast, f32[1]{0} %slice), dimensions={0}
  ROOT %slice.1 = f32[1]{0} slice(f32[1025]{0} %concatenate.1), slice={[0:1]}
}

%main_computation {
  %param = f32[1]{0} parameter(0)
  %reshape = f32[] reshape(f32[1]{0} %param)
  %broadcast = f32[1024]{0} broadcast(f32[] %reshape), dimensions={}
  %negate = f32[1024]{0} negate(f32[1024]{0} %broadcast)
  %concatenate = f32[2048]{0} concatenate(f32[1024]{0} %negate, f32[1024]{0} %negate), dimensions={0}
  %slice = f32[1]{0} slice(f32[2048]{0} %concatenate), slice={[0:1]}
  %concatenate.1 = f32[1025]{0} concatenate(f32[1024]{0} %broadcast, f32[1]{0} %slice), dimensions={0}
  ROOT %slice.1 = f32[1]{0} slice(f32[1025]{0} %concatenate.1), slice={[0:1]}
}

ENTRY %main {
  %param = f32[1]{0} parameter(0)
  %call-start = ((f32[1]{0}), f32[1]{0}, s32[]) call-start(f32[1]{0} %param), to_apply=%offload_computation, async_execution_thread="offload"
  %call-done = f32[1]{0} call-done(((f32[1]{0}), f32[1]{0}, s32[]) %call-start)
  ROOT %call = f32[1]{0} call(f32[1]{0} %call-done), to_apply=%main_computation
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  HloInstruction* call_start = FindInstruction(module.get(), "call-start");
  // Computation requires 16KB without rematerialization, but uses only 12KB
  // with rematerialization so pick a memory limit between these values (14KB).
  // Asynchronous computation will run on 16 devices and we do not rematerialize
  // it, so it will reserve 16 * 16Kb from the memory limit.
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloRematerialization(
          /*memory_limit_bytes=*/kNumParallelThreads * 16 * 1024 + 14 * 1024,
          module.get(),
          {{call_start->async_wrapped_computation(), kNumParallelThreads}}));
  EXPECT_TRUE(changed);
}

// Inherits methods to create rematerializable computations. See
// RematerializationTestBase for more.
class RecomputeAndCompressHloRematerializationTest
    : public RematerializationTestBase {
 protected:
  absl::StatusOr<bool> RunHloRematerialization(int64_t memory_limit_bytes,
                                               HloModule* module,
                                               int64_t min_remat_size = 0) {
    TF_EXPECT_OK(verifier().Run(module).status());
    if (!module->has_schedule()) {
      HloMemoryScheduler scheduler(
          [](const BufferValue& buffer) { return ByteSizeOf(buffer.shape()); },
          ComputationSchedulerToModuleScheduler(DefaultMemoryScheduler));
      TF_EXPECT_OK(scheduler.Run(module).status());
    }

    // First, get a set of instruction names before running remat.
    for (const HloComputation* computation : module->computations()) {
      before_computation_names_.insert(computation->name());
      for (const HloInstruction* instruction : computation->instructions()) {
        before_instruction_names_.insert(instruction->name());
      }
    }

    // Run remat.
    HloRematerialization::RematerializationModeConfig config(
        /*recompute=*/true, /*compress=*/true, /*host_offload=*/false);
    auto shape_size_func = [](const Shape& shape) { return ByteSizeOf(shape); };
    HloCostAnalysis cost_analysis(shape_size_func);
    HloRematerialization::Options options(
        cost_analysis, config, memory_limit_bytes,
        /*block_size_limit=*/1, /*block_rematerialization_factor=*/1,
        min_remat_size, /*compact_shape_function=*/nullptr,
        /*host_memory_offload_config=*/std::nullopt,
        /*async_threads=*/{});
    HloRematerialization::RematerializationSizes sizes;
    HloRematerialization remat(options, sizes);
    absl::StatusOr<bool> result = remat.Run(module);

    // Finally, get a set of instruction names after running remat.
    for (const HloComputation* computation : module->computations()) {
      if (!before_computation_names_.contains(computation->name())) {
        // This computation was cloned by remat. Skip.
        continue;
      }
      for (const HloInstruction* instruction : computation->instructions()) {
        after_instruction_names_.insert(instruction->name());
      }
    }

    return result;
  }

  void CheckForRematInInstructionNames(absl::string_view test_case_name) {
    constexpr const absl::string_view kRematInstructionNameMustContain =
        ".remat";
    for (const auto& instruction_name : after_instruction_names_) {
      if (!before_instruction_names_.contains(instruction_name)) {
        // This is a newly inserted instruction by remat, check that it contains
        // the target name.
        EXPECT_TRUE(absl::StrContains(instruction_name,
                                      kRematInstructionNameMustContain))
            << "[" << test_case_name << "] Instruction \"" << instruction_name
            << "\" must contain \"" << kRematInstructionNameMustContain << "\"";
      }
    }
  }

 private:
  absl::flat_hash_set<absl::string_view> before_computation_names_;
  absl::flat_hash_set<absl::string_view> before_instruction_names_;
  absl::flat_hash_set<absl::string_view> after_instruction_names_;
};

// Test rematerialization of a single computation produced by
// MakeRematerializableComputation.
TEST_F(RecomputeAndCompressHloRematerializationTest, SingleComputation) {
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(MakeRematerializableComputation());

  // Find and save the original broadcast instruction which should be
  // rematerialized.
  const HloInstruction* slice = computation->root_instruction();
  ASSERT_THAT(slice, op::Slice(op::Concatenate(op::Broadcast(_), _)));
  const HloInstruction* concat = slice->operand(0);
  const HloInstruction* bcast = concat->operand(0);

  // Computation requires 16KB without rematerialization, but uses only 12KB
  // with rematerialization so pick a memory limit between these values (14KB).
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/14 * 1024, module.get()));
  EXPECT_TRUE(changed);

  // Root should not have changed.
  EXPECT_EQ(computation->root_instruction(), slice);

  // The broadcast should have been rematerialized.
  const HloInstruction* remat_bcast = concat->operand(0);
  EXPECT_THAT(remat_bcast, op::Broadcast(::testing::Ne(bcast)));

  // The rematerialized broadcast should be immediate before the concat in the
  // sequence.
  EXPECT_EQ(module->schedule()
                .sequence(computation)
                .instructions()[computation->instruction_count() - 2],
            concat);
  EXPECT_EQ(module->schedule()
                .sequence(computation)
                .instructions()[computation->instruction_count() - 3],
            remat_bcast);
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

// Test rematerialization of a single computation that contains nodes that
// doesn't contain node worth using remat.
TEST_F(RecomputeAndCompressHloRematerializationTest,
       SingleComputationNoWorthRemat) {
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(MakeRematerializableComputation());

  // Find and save the original broadcast instruction which should be
  // rematerialized.
  const HloInstruction* slice = computation->root_instruction();
  ASSERT_THAT(slice, op::Slice(op::Concatenate(op::Broadcast(_), _)));

  // Set the minimum remat size to 14KiB, meaning no nodes should be remat.
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/14 * 1024, module.get(),
                              /*min_remat_size=*/14 * 1024));
  EXPECT_FALSE(changed);
}

// Test rematerialization of a single computation produced by
// MakeRematerializableComputation but with a sufficiently high memory limit
// such that no instructions are rematerialized.
TEST_F(RecomputeAndCompressHloRematerializationTest,
       SingleComputationNoRematerialization) {
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(MakeRematerializableComputation());

  EXPECT_EQ(computation->instruction_count(), 8);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/20 * 1024, module.get()));

  // No instructions should have been materialized.
  EXPECT_FALSE(changed);
  EXPECT_EQ(computation->instruction_count(), 8);
}

// Test rematerialization of a computation which calls another computation via a
// while. Both the entry computation and while body computation can have memory
// usage reduced via rematerialization however the memory limit is set such that
// only one computation needs to have an instruction rematerialized. The entry
// computation should be the one chosen because rematerialization in the while
// will presumably be more expensive.
TEST_F(RecomputeAndCompressHloRematerializationTest, RematerializeAroundWhile) {
  auto module = CreateNewVerifiedModule();

  auto cond_builder = HloComputation::Builder(TestName() + ".cond");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec1_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloComputation* while_cond =
      module->AddEmbeddedComputation(cond_builder.Build());

  HloComputation* body_computation = module->AddEmbeddedComputation(
      MakeRematerializableComputation(/*suffix=*/".body"));
  HloComputation* entry_computation =
      module->AddEntryComputation(MakeRematerializableWhileComputation(
          while_cond, /*while_body=*/body_computation));

  EXPECT_EQ(entry_computation->instruction_count(), 7);
  EXPECT_EQ(body_computation->instruction_count(), 8);

  // The body computation uses 16KB and the entry computation uses 2KB at the
  // while so the peak memory use of the module is 18KB. Set the memory limit a
  // bit lower (17KB) to force rematerialization of the entry computation.
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/17 * 1024, module.get()));
  EXPECT_TRUE(changed);

  // Only the entry computation should have a rematerialized instruction added.
  EXPECT_EQ(entry_computation->instruction_count(), 8);
  EXPECT_EQ(body_computation->instruction_count(), 8);
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

// Test rematerialization of a computation which calls another computation via a
// while. Both the entry computation and while body computation should have
// computations rematerialized.
TEST_F(RecomputeAndCompressHloRematerializationTest,
       RematerializeEntryAndWhileBody) {
  auto module = CreateNewVerifiedModule();

  auto cond_builder = HloComputation::Builder(TestName() + ".cond");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec1_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloComputation* while_cond =
      module->AddEmbeddedComputation(cond_builder.Build());

  HloComputation* body_computation = module->AddEmbeddedComputation(
      MakeRematerializableComputation(/*suffix=*/".body"));
  HloComputation* entry_computation =
      module->AddEntryComputation(MakeRematerializableWhileComputation(
          while_cond, /*while_body=*/body_computation));

  EXPECT_EQ(entry_computation->instruction_count(), 7);
  EXPECT_EQ(body_computation->instruction_count(), 8);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/15 * 1024, module.get()));
  EXPECT_TRUE(changed);

  // Both computations should have rematerialized instructions added.
  EXPECT_EQ(entry_computation->instruction_count(), 9);
  EXPECT_EQ(body_computation->instruction_count(), 9);
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

// Test rematerialization of a doubly nested computation. All computations
// should have an instruction rematerialized.
TEST_F(RecomputeAndCompressHloRematerializationTest,
       RematerializeNestedComputations) {
  auto module = CreateNewVerifiedModule();

  auto cond_builder = HloComputation::Builder(TestName() + ".cond");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec1_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloComputation* while_cond =
      module->AddEmbeddedComputation(cond_builder.Build());
  HloComputation* while_cond_copy =
      module->AddEmbeddedComputation(while_cond->Clone());

  HloComputation* inner_computation = module->AddEmbeddedComputation(
      MakeRematerializableComputation(/*suffix=*/".inner"));
  HloComputation* middle_computation =
      module->AddEmbeddedComputation(MakeRematerializableWhileComputation(
          while_cond, /*while_body=*/inner_computation,
          /*suffix=*/".middle"));
  HloComputation* entry_computation =
      module->AddEntryComputation(MakeRematerializableWhileComputation(
          while_cond_copy, /*while_body=*/middle_computation));

  EXPECT_EQ(entry_computation->instruction_count(), 7);
  EXPECT_EQ(middle_computation->instruction_count(), 7);
  EXPECT_EQ(inner_computation->instruction_count(), 8);

  // If all computations are maximally rematerialized then peak memory usage is
  // ~12K so pick something slightly larger.
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/13 * 1024, module.get()));
  EXPECT_TRUE(changed);

  // All computations should have rematerialized instructions added.
  EXPECT_EQ(entry_computation->instruction_count(), 9);
  EXPECT_EQ(middle_computation->instruction_count(), 9);
  EXPECT_EQ(inner_computation->instruction_count(), 9);
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

TEST_F(RecomputeAndCompressHloRematerializationTest, RngNotRematerialized) {
  // Test that a single rng is not rematerialized:
  //
  // Entry computation:
  //   F32[] %param = {...}
  //   F32[1024] rng = rng(param)
  //   F32[1024] tanh = tanh(rng)
  //   F32[1024] exp = exp(rng)
  //   F32[1024] add_0 = add(rng, tanh)              // LIVE: add_0 + rng +
  //                                                 //       tanh + exp
  //
  //   F32[1024] add_1 = add(rng, add(exp, add_0))   // LIVE: add_1 + add_0 +
  //                                                 //       rng + tanh + exp
  //
  //   F32[1024] add_2 = add(rng, add(tanh, add_1))  // LIVE: add_2 + add_1 +
  //                                                 //       rng + tanh + exp
  auto module = CreateNewVerifiedModule();

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  auto rng = builder.AddInstruction(HloInstruction::CreateRng(
      vec1024_shape_, RandomDistribution::RNG_UNIFORM, {param, param}));
  auto tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kTanh, rng));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kExp, rng));
  auto add_0 = builder.AddInstruction(
      HloInstruction::CreateBinary(vec1024_shape_, HloOpcode::kAdd, rng, tanh));
  auto add_1 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, rng,
      builder.AddInstruction(HloInstruction::CreateBinary(
          vec1024_shape_, HloOpcode::kAdd, exp, add_0))));
  builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, rng,
      builder.AddInstruction(HloInstruction::CreateBinary(
          vec1024_shape_, HloOpcode::kAdd, tanh, add_1))));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  auto count_rngs = [](const HloComputation* computation) {
    int64_t rng_count = 0;
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kRng) {
        ++rng_count;
      }
    }
    return rng_count;
  };
  // Before rematerialization there should be a single broadcast rng in
  // the graph.
  ASSERT_EQ(count_rngs(entry_computation), 1);
  const int64_t original_instruction_count =
      entry_computation->instruction_count();
  // Pick a memory limit some where between 24KB (initial peak memory including
  // parameter and output) and 20KB (peak memory possible with
  // rematerialization).
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloRematerialization(
          /*memory_limit_bytes=*/4 * ByteSizeOf(vec1024_shape_), module.get()));
  EXPECT_TRUE(changed);
  // The rng should not have been rematerialized.
  EXPECT_EQ(count_rngs(entry_computation), 1);
  // There should have been rematerialization.
  EXPECT_GT(entry_computation->instruction_count(), original_instruction_count);
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

TEST_F(RecomputeAndCompressHloRematerializationTest,
       InstructionRematerializedMultipleTimes) {
  // Test that a single instruction is rematerialized several times. Module:
  //
  // Entry computation:
  //   F32[] %param = {...}
  //   F32[1024] %bcast = broadcast(%param)
  //   F32[1024] %add_1 = add(%bcast, bcast)
  //   F32[1024] %call_1 = call(Subcomputation, {%add_1})
  //   F32[1024] %add_2 = add(%bcast, call_1)
  //   F32[1024] %call_2 = call(SubComputation, {%add_2})
  //   F32[1024] %add_3 = add(%bcast, call_2)
  //   F32[1024] %call_3 = call(Subcomputation, {%add_3})
  //   F32[1024] %add_4 = add(%bcast, call_3)
  //
  // Subcomputation:
  //   F32[1024] %param = {...}
  //   F32[2048] %concat = concat({%param, %param})
  //   F32[1024] %slice = slice(%concat)
  //
  // The value %bcast is live across each call of Subcomputation (which requires
  // 8KB) though the value is not used in the calls. Rematerializing %bcast
  // across these calls reduces peak memory use from ~20KB down to ~16KB.
  auto module = CreateNewVerifiedModule();

  HloComputation* subcomputation = nullptr;
  {
    auto builder = HloComputation::Builder(TestName() + ".subcomputation");
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1024_shape_, "param"));
    auto concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {2048}), {param, param},
        /*dimension=*/0));
    builder.AddInstruction(HloInstruction::CreateSlice(
        vec1024_shape_, concat, /*start_indices=*/{0},
        /*limit_indices=*/{1024}, /*strides=*/{1}));
    subcomputation = module->AddEmbeddedComputation(builder.Build());
  }

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  auto bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(vec1024_shape_, param, {}));
  auto add_1 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, bcast, bcast));
  auto call_1 = builder.AddInstruction(
      HloInstruction::CreateCall(vec1024_shape_, {add_1}, subcomputation));
  auto add_2 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, bcast, call_1));
  auto call_2 = builder.AddInstruction(
      HloInstruction::CreateCall(vec1024_shape_, {add_2}, subcomputation));
  auto add_3 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, bcast, call_2));
  auto call_3 = builder.AddInstruction(
      HloInstruction::CreateCall(vec1024_shape_, {add_3}, subcomputation));
  auto add_4 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, bcast, call_3));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  auto count_broadcasts = [](const HloComputation* computation) {
    int64_t bcast_count = 0;
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kBroadcast) {
        bcast_count++;
      }
    }
    return bcast_count;
  };

  // Before rematerialization there should be a single broadcast instruction in
  // the graph.
  EXPECT_EQ(count_broadcasts(entry_computation), 1);
  EXPECT_EQ(entry_computation->instruction_count(), 9);

  EXPECT_EQ(add_2->operand(0), bcast);
  EXPECT_EQ(add_3->operand(0), bcast);
  EXPECT_EQ(add_4->operand(0), bcast);

  // Pick a memory limit some where between 24KB (initial peak memory including
  // parameter and output) and 20KB (peak memory possible with
  // rematerialization).
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/22 * 1024, module.get()));
  EXPECT_TRUE(changed);

  // The broadcast should have been rematerialized 3 times.
  EXPECT_EQ(count_broadcasts(entry_computation), 4);
  EXPECT_EQ(entry_computation->instruction_count(), 12);

  // The operands of add_2, add_3, and add_4 should all be rematerialized
  // broadcasts.
  EXPECT_NE(add_2->operand(0), bcast);
  EXPECT_THAT(add_2->operand(0), op::Broadcast(param));
  EXPECT_NE(add_3->operand(0), bcast);
  EXPECT_THAT(add_3->operand(0), op::Broadcast(param));
  EXPECT_NE(add_4->operand(0), bcast);
  EXPECT_THAT(add_4->operand(0), op::Broadcast(param));
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

TEST_F(RecomputeAndCompressHloRematerializationTest, CopyNotRematerialized) {
  // Test that copies are not rematerialized.
  auto module = CreateNewVerifiedModule();

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec1024_shape_, "param"));

  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kCopy, param));

  auto negate_a_1 = builder.AddInstruction(
      HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kNegate, copy));

  auto negate_a_2 = builder.AddInstruction(HloInstruction::CreateUnary(
      vec1024_shape_, HloOpcode::kNegate, negate_a_1));

  auto negate_b_1 = builder.AddInstruction(
      HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kNegate, copy));

  auto negate_b_2 = builder.AddInstruction(HloInstruction::CreateUnary(
      vec1024_shape_, HloOpcode::kNegate, negate_b_1));

  builder.AddInstruction(HloInstruction::CreateTuple({negate_a_2, negate_b_2}));

  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/1 * 1024, module.get()));

  auto count_copies = [](const HloComputation* computation) {
    int64_t copy_count = 0;
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCopy) {
        copy_count++;
      }
    }
    return copy_count;
  };
  EXPECT_TRUE(changed);

  EXPECT_EQ(count_copies(entry_computation), 1);
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

// Test rematerialization of values through bitcasts
// Its expected that the broadcast gets rematerialized
TEST_F(RecomputeAndCompressHloRematerializationTest, ThroughBitcastRemat) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

ENTRY %mycomp (param: f32[1]) -> f32[1] {
  %param = f32[1]{0} parameter(0)
  %reshape = f32[] reshape(f32[1]{0} %param)
  %broadcast = f32[1024,1]{1,0} broadcast(f32[] %reshape), dimensions={}
  %bitcast = f32[1024]{0} bitcast(f32[1024,1]{1,0} %broadcast)
  %negate = f32[1024,1]{1,0} negate(f32[1024,1]{1,0} %broadcast)
  %concatenate = f32[2048,1]{1,0} concatenate(f32[1024,1]{1,0} %negate, f32[1024,1]{1,0} %negate), dimensions={0}
  %slice = f32[1,1]{1,0} slice(f32[2048,1]{1,0} %concatenate), slice={[0:1], [0:1]}
  %bitcast.1 = f32[1]{0} bitcast(f32[1,1]{1,0} %slice)
  %concatenate.1 = f32[1025]{0} concatenate(f32[1024]{0} %bitcast, f32[1]{0} %bitcast.1), dimensions={0}
  ROOT %slice.1 = f32[1]{0} slice(f32[1025]{0} %concatenate.1), slice={[0:1]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto* computation = module->entry_computation();
  // Find and save the original broadcast instruction which should be
  // rematerialized.
  const HloInstruction* slice = computation->root_instruction();
  ASSERT_THAT(slice,
              op::Slice(op::Concatenate(op::Bitcast(op::Broadcast(_)), _)));
  const HloInstruction* concat = slice->operand(0);
  const HloInstruction* bcast = concat->operand(0)->operand(0);

  // Computation requires 16KB without rematerialization, but uses only 12KB
  // with rematerialization so pick a memory limit between these values (14KB).
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/14 * 1024, module.get()));
  EXPECT_TRUE(changed);

  // Root should not have changed.
  EXPECT_EQ(computation->root_instruction(), slice);

  // The bitcast for the rematerialized broadcast
  const HloInstruction* remat_bitcast = concat->operand(0);
  // The broadcast should have been rematerialized.
  const HloInstruction* remat_broadcast = remat_bitcast->operand(0);

  EXPECT_THAT(remat_broadcast, op::Broadcast(::testing::Ne(bcast)));

  // The rematerialized broadcast should be immediately before its bitcast
  // and the bitcast before the concatenate in the sequence.
  EXPECT_EQ(module->schedule()
                .sequence(computation)
                .instructions()[computation->instruction_count() - 2],
            concat);
  EXPECT_EQ(module->schedule()
                .sequence(computation)
                .instructions()[computation->instruction_count() - 3],
            remat_bitcast);
  EXPECT_EQ(module->schedule()
                .sequence(computation)
                .instructions()[computation->instruction_count() - 4],
            remat_broadcast);
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

// Test that the "deny list for move remats" engages when we rematerialize
// through bitcasts.
TEST_F(RecomputeAndCompressHloRematerializationTest,
       ThroughBitcastRematInfiniteLoop) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

ENTRY %mycomp (param: f32[1]) -> f32[1024] {
  %param = f32[1]{0} parameter(0)
  %reshape = f32[] reshape(f32[1]{0} %param)
  %broadcast = f32[1024,1]{1,0} broadcast(f32[] %reshape), dimensions={}
  %bitcast = f32[1024]{0} bitcast(f32[1024,1]{1,0} %broadcast)
  %broadcast2 = f32[1024,1]{1,0} broadcast(f32[] %reshape), dimensions={}
  %bitcast2 = f32[1024]{0} bitcast(f32[1024,1]{1,0} %broadcast2)
  ROOT %add = f32[1024]{0} add(f32[1024]{0} %bitcast, f32[1024]{0} %bitcast2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto* computation = module->entry_computation();
  // Find and save the original broadcasts instruction which should be
  // rematerialized.
  const HloInstruction* add = computation->root_instruction();
  // Run with a low rematerialization limit that cannot be satisfied to make
  // sure that we don't get stuck in a loop trying to lower it.
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/1024, module.get()));
  ASSERT_THAT(add, op::Add(op::Bitcast(op::Broadcast(_)),
                           op::Bitcast(op::Broadcast(_))));
  EXPECT_TRUE(changed);
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

TEST_F(RecomputeAndCompressHloRematerializationTest, RematTupleShape) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

%add_mul_comp {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  %x = f32[1024]{0} broadcast(f32[] %p0), dimensions={}
  %y = f32[1024]{0} broadcast(f32[] %p1), dimensions={}
  %add = f32[1024] add(%x, %y)
  %mul = f32[1024] multiply(%x, %y)
  ROOT %out = (f32[1024], f32[1024]) tuple(%add, %mul)
}

ENTRY %entry {
  %param.0 = f32[] parameter(0)
  %param.1 = f32[] parameter(1)
  %fus = (f32[1024]{0}, f32[1024]{0}) fusion(%param.0, %param.1), kind=kLoop,
    calls=%add_mul_comp
  %gte.1 = f32[1024]{0} get-tuple-element(%fus), index=0
  %add = f32[1024]{0} add(f32[1024]{0} %gte.1, f32[1024]{0} %gte.1)
  %broadcast.1 = f32[1024]{0} broadcast(f32[] %param.0), dimensions={}
  %mul = f32[1024]{0} multiply(f32[1024]{0} %add, f32[1024]{0} %broadcast.1)
  %gte.2 = f32[1024]{0} get-tuple-element(%fus), index=1
  ROOT %add.2 = f32[1024]{0} add(f32[1024]{0} %mul, f32[1024]{0} %gte.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloComputation* computation = module->entry_computation();
  const HloInstruction* add = computation->root_instruction();
  ASSERT_THAT(add, op::Add(op::Multiply(), op::GetTupleElement(op::Fusion())));
  const HloInstruction* fusion = add->operand(0)->operand(0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/11 * 1024, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_THAT(
      add, op::Add(op::Multiply(), AllOf(op::Fusion(), ::testing::Ne(fusion))));
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

TEST_F(RecomputeAndCompressHloRematerializationTest, RematTupleShapeDoubleUse) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

%add_mul_comp {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  %x = f32[1024]{0} broadcast(f32[] %p0), dimensions={}
  %y = f32[1024]{0} broadcast(f32[] %p1), dimensions={}
  %add = f32[1024] add(%x, %y)
  %mul = f32[1024] multiply(%x, %y)
  ROOT %out = (f32[1024], f32[1024]) tuple(%add, %mul)
}

ENTRY %entry {
  %param.0 = f32[] parameter(0)
  %param.1 = f32[] parameter(1)
  %fus = (f32[1024]{0}, f32[1024]{0}) fusion(%param.0, %param.1), kind=kLoop,
    calls=%add_mul_comp
  %gte.1 = f32[1024]{0} get-tuple-element(%fus), index=0
  %add = f32[1024]{0} add(f32[1024]{0} %gte.1, f32[1024]{0} %gte.1)
  %broadcast.1 = f32[1024]{0} broadcast(f32[] %param.0), dimensions={}
  %mul = f32[1024]{0} multiply(f32[1024]{0} %add, f32[1024]{0} %broadcast.1)
  %gte.2 = f32[1024]{0} get-tuple-element(%fus), index=1
  %gte.3 = f32[1024]{0} get-tuple-element(%fus), index=0
  %add.2 = f32[1024]{0} add(f32[1024]{0} %mul, f32[1024]{0} %gte.2)
  ROOT %mul.2 = f32[1024]{0} multiply(f32[1024]{0} %add.2, f32[1024]{0} %gte.3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloComputation* computation = module->entry_computation();
  const HloInstruction* add = computation->root_instruction();
  ASSERT_THAT(add, op::Multiply(op::Add(op::Multiply(),
                                        op::GetTupleElement(op::Fusion())),
                                op::GetTupleElement(op::Fusion())));
  const HloInstruction* fusion = add->operand(0)->operand(0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/11 * 1024, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_THAT(
      add,
      op::Multiply(
          op::Add(op::Multiply(), op::GetTupleElement(AllOf(
                                      op::Fusion(), ::testing::Ne(fusion)))),
          op::GetTupleElement(AllOf(op::Fusion(), ::testing::Ne(fusion)))));
  // Check that the rematerialized fusion is the same for both ops.
  EXPECT_EQ(add->operand(0)->operand(1)->operand(0),
            add->operand(1)->operand(0));
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

TEST_F(RecomputeAndCompressHloRematerializationTest,
       RematTupleShapeThroughBitcasts) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

%add_mul_comp {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  %x = f32[1024]{0} broadcast(f32[] %p0), dimensions={}
  %y = f32[1024]{0} broadcast(f32[] %p1), dimensions={}
  %add = f32[1024] add(%x, %y)
  %mul = f32[1024] multiply(%x, %y)
  ROOT %out = (f32[1024], f32[1024]) tuple(%add, %mul)
}

ENTRY %entry {
  %param.0 = f32[] parameter(0)
  %param.1 = f32[] parameter(1)
  %fus = (f32[1024]{0}, f32[1024]{0}) fusion(%param.0, %param.1), kind=kLoop,
    calls=%add_mul_comp
  %gte.1 = f32[1024]{0} get-tuple-element(%fus), index=0
  %add = f32[1024]{0} add(f32[1024]{0} %gte.1, f32[1024]{0} %gte.1)
  %broadcast.1 = f32[1024]{0} broadcast(f32[] %param.0), dimensions={}
  %mul = f32[1024]{0} multiply(f32[1024]{0} %add, f32[1024]{0} %broadcast.1)
  %gte.2 = f32[1024]{0} get-tuple-element(%fus), index=1
  %bc.1 = f32[1024,1]{0,1} bitcast(%mul)
  %bc.2 = f32[1024,1]{0,1} bitcast(%gte.2)
  ROOT %add.2 = f32[1024,1]{0,1} add(f32[1024,1]{0,1} %bc.1,
    f32[1024,1]{0,1} %bc.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloComputation* computation = module->entry_computation();
  const HloInstruction* add = computation->root_instruction();
  ASSERT_THAT(add, op::Add(op::Bitcast(op::Multiply()),
                           op::Bitcast(op::GetTupleElement(op::Fusion()))));
  const HloInstruction* fusion = add->operand(0)->operand(0)->operand(0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/11 * 1024, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_THAT(add,
              op::Add(op::Bitcast(op::Multiply()),
                      op::Bitcast(AllOf(op::Fusion(), ::testing::Ne(fusion)))));
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

TEST_F(RecomputeAndCompressHloRematerializationTest, RematThroughTuple) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

%add_mul_comp {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  %x = f32[1024]{0} broadcast(f32[] %p0), dimensions={}
  %y = f32[1024]{0} broadcast(f32[] %p1), dimensions={}
  %add = f32[1024] add(%x, %y)
  %mul = f32[1024] multiply(%x, %y)
  ROOT %out = (f32[1024], f32[1024]) tuple(%add, %mul)
}

ENTRY %entry {
  %param.0 = f32[] parameter(0)
  %param.1 = f32[] parameter(1)
  %fus = (f32[1024]{0}, f32[1024]{0}) fusion(%param.0, %param.1), kind=kLoop,
    calls=%add_mul_comp
  %gte.1 = f32[1024]{0} get-tuple-element(%fus), index=0
  %gte.3 = f32[1024]{0} get-tuple-element(%fus), index=1
  %add = f32[1024]{0} add(f32[1024]{0} %gte.1, f32[1024]{0} %gte.3)
  %broadcast.1 = f32[1024]{0} broadcast(f32[] %param.0), dimensions={}
  %mul = f32[1024]{0} multiply(f32[1024]{0} %add, f32[1024]{0} %broadcast.1)
  %tpl = (f32[1024]{0}, f32[1024]{0}) tuple(%gte.1, %add)
  %bc.1 = f32[1024,1]{0,1} bitcast(%mul)
  %gte.2 = f32[1024]{0} get-tuple-element(%tpl), index=0
  ROOT %add.2 = f32[1024]{0} add(f32[1024]{0} %gte.2, f32[1024]{0} %add)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  const HloComputation* computation = module->entry_computation();
  const HloInstruction* add = computation->root_instruction();
  ASSERT_THAT(add, op::Add(op::GetTupleElement(
                               op::Tuple(op::GetTupleElement(op::Fusion()), _)),
                           op::Add()));
  const HloInstruction* tuple = add->operand(0)->operand(0);
  const HloInstruction* fusion = tuple->operand(0)->operand(0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/11 * 1024, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_THAT(add, op::Add(AllOf(op::Fusion(), ::testing::Ne(tuple),
                                 ::testing::Ne(fusion)),
                           op::Add()));
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

// Make sure when rematerializing all-gathers we increment channel_ids properly.
TEST_F(RecomputeAndCompressHloRematerializationTest, AllGatherChannelId) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

ENTRY %mycomp (param: f32[1]) -> f32[1] {
  %param = f32[1]{0} parameter(0)
  %reshape = f32[] reshape(f32[1]{0} %param)
  %broadcast = f32[256,1]{1,0} broadcast(f32[] %reshape), dimensions={}
  %ag = f32[1024,1]{1,0} all-gather(f32[256,1]{1,0} %broadcast), dimensions={0},
    channel_id=1, replica_groups={{0,1,2,3}}, use_global_device_ids=true
  %bitcast = f32[1024]{0} bitcast(f32[1024,1]{1,0} %ag)
  %negate = f32[1024,1]{1,0} negate(f32[1024,1]{1,0} %ag)
  %concatenate = f32[2048,1]{1,0} concatenate(f32[1024,1]{1,0} %negate,
    f32[1024,1]{1,0} %negate), dimensions={0}
  %slice = f32[1,1]{1,0} slice(f32[2048,1]{1,0} %concatenate),
    slice={[0:1], [0:1]}
  %bitcast.1 = f32[1]{0} bitcast(f32[1,1]{1,0} %slice)
  %concatenate.1 = f32[1025]{0} concatenate(f32[1024]{0} %bitcast,
    f32[1]{0} %bitcast.1), dimensions={0}
  ROOT %slice.1 = f32[1]{0} slice(f32[1025]{0} %concatenate.1), slice={[0:1]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto* computation = module->entry_computation();
  // Find and save the original broadcast instruction which should be
  // rematerialized.
  const HloInstruction* slice = computation->root_instruction();
  ASSERT_THAT(slice, op::Slice(op::Concatenate(
                         op::Bitcast(op::AllGather(op::Broadcast(_))), _)));

  // Computation requires 16KB without rematerialization, but uses only 12KB
  // with rematerialization so pick a memory limit between these values (14KB).
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/14 * 1024, module.get()));
  EXPECT_TRUE(changed);

  // Root should not have changed.
  EXPECT_EQ(computation->root_instruction(), slice);

  // Original all-gather.
  const HloInstruction* original_ag = FindInstruction(module.get(), "ag");
  // The all-gather should have been rematerialized
  const HloInstruction* remat_ag = FindInstruction(module.get(), "ag.remat");

  EXPECT_NE(remat_ag, nullptr);
  EXPECT_TRUE(original_ag->channel_id().has_value());
  EXPECT_TRUE(remat_ag->channel_id().has_value());
  EXPECT_EQ(*remat_ag->channel_id(), *original_ag->channel_id() + 1);
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

TEST_F(RecomputeAndCompressHloRematerializationTest, RematTupleArgFusion) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

%add_mul_comp {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  %x = f32[1024]{0} broadcast(f32[] %p0), dimensions={}
  %y = f32[1024]{0} broadcast(f32[] %p1), dimensions={}
  %add = f32[1024] add(%x, %y)
  %mul = f32[1024] multiply(%x, %y)
  ROOT %out = (f32[1024], f32[1024]) tuple(%add, %mul)
}

%add_comp {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %add = add(%p0, %p1)
}

%add_tuple_comp {
  %p = (f32[1024]{0}, f32[1024]{0}) parameter(0)
  %p0 = get-tuple-element(%p), index=0
  %p1 = get-tuple-element(%p), index=1
  ROOT %add = add(%p0, %p1)
}

ENTRY %entry {
  %param.0 = f32[] parameter(0)
  %param.1 = f32[] parameter(1)
  %fus = (f32[1024]{0}, f32[1024]{0}) fusion(%param.0, %param.1), kind=kLoop,
    calls=%add_mul_comp
  %gte.1 = f32[1024]{0} get-tuple-element(%fus), index=0
  %gte.3 = f32[1024]{0} get-tuple-element(%fus), index=1
  %add.0 = f32[1024]{0} add(f32[1024]{0} %gte.1, f32[1024]{0} %gte.3)
  %broadcast.1 = f32[1024]{0} broadcast(f32[] %param.0), dimensions={}
  %add.1 = f32[1024]{0} add(f32[1024]{0} %add.0, f32[1024]{0} %broadcast.1)
  %c = f32[] constant(0)
  %reduce = f32[] reduce(%add.1, %c), dimensions={0}, to_apply=add_comp
  %fus.1 = f32[1024]{0} fusion(%fus), kind=kLoop, calls=%add_tuple_comp
  ROOT %tuple = tuple(%reduce, %fus.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  const HloComputation* computation = module->entry_computation();
  const HloInstruction* root = computation->root_instruction();
  ASSERT_THAT(root, op::Tuple(op::Reduce(), op::Fusion(op::Fusion())));
  const HloInstruction* fusion1 = root->operand(1);
  const HloInstruction* fusion0 = fusion1->operand(0);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/11 * 1024, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_THAT(
      root, op::Tuple(op::Reduce(),
                      op::Fusion(AllOf(op::Fusion(), ::testing::Ne(fusion0)))));
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

TEST_F(RecomputeAndCompressHloRematerializationTest,
       RematFusionUpdateSchedule) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

%custom_call_comp {
  %p = f32[1024]{0} parameter(0)
  ROOT %n = f32[1024]{0} negate(p)
}

%add_mul_comp {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  %x = f32[1024]{0} broadcast(f32[] %p0), dimensions={}
  %y = f32[1024]{0} broadcast(f32[] %p1), dimensions={}
  %add = f32[1024] add(%x, %y)
  %mul = f32[1024] multiply(%x, %y)
  %c = f32[1024] custom-call(%mul), custom_call_target="SomeCall", called_computations={custom_call_comp}
  ROOT %out = (f32[1024], f32[1024]) tuple(%add, %c)
}

ENTRY %entry {
  %param.0 = f32[] parameter(0)
  %param.1 = f32[] parameter(1)
  %fus = (f32[1024]{0}, f32[1024]{0}) fusion(%param.0, %param.1), kind=kLoop,
    calls=%add_mul_comp
  %gte.1 = f32[1024]{0} get-tuple-element(%fus), index=0
  %add = f32[1024]{0} add(f32[1024]{0} %gte.1, f32[1024]{0} %gte.1)
  %broadcast.1 = f32[1024]{0} broadcast(f32[] %param.0), dimensions={}
  %mul = f32[1024]{0} multiply(f32[1024]{0} %add, f32[1024]{0} %broadcast.1)
  %gte.2 = f32[1024]{0} get-tuple-element(%fus), index=1
  %gte.3 = f32[1024]{0} get-tuple-element(%fus), index=0
  %add.2 = f32[1024]{0} add(f32[1024]{0} %mul, f32[1024]{0} %gte.2)
  ROOT %mul.2 = f32[1024]{0} multiply(f32[1024]{0} %add.2, f32[1024]{0} %gte.3)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloComputation* computation = module->entry_computation();
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/11 * 1024, module.get()));
  EXPECT_TRUE(changed);
  XLA_VLOG_LINES(1, module->ToString());
  const HloInstruction* add = computation->root_instruction();
  const HloInstruction* fusion = add->operand(0)->operand(0);
  ASSERT_THAT(
      add,
      op::Multiply(
          op::Add(op::Multiply(), op::GetTupleElement(AllOf(
                                      op::Fusion(), ::testing::Ne(fusion)))),
          op::GetTupleElement(AllOf(op::Fusion(), ::testing::Ne(fusion)))));
  // Check that the rematerialized fusion is the same for both ops.
  const HloInstruction* fusion0 = add->operand(0)->operand(1)->operand(0);
  const HloInstruction* fusion1 = add->operand(1)->operand(0);
  auto it = std::find_if(fusion0->fused_instructions().begin(),
                         fusion0->fused_instructions().end(),
                         [](const HloInstruction* instr) {
                           return instr->opcode() == HloOpcode::kCustomCall;
                         });
  ASSERT_NE(it, fusion0->fused_instructions().end());
  auto it2 = std::find_if(fusion1->fused_instructions().begin(),
                          fusion1->fused_instructions().end(),
                          [](const HloInstruction* instr) {
                            return instr->opcode() == HloOpcode::kCustomCall;
                          });
  ASSERT_NE(it2, fusion1->fused_instructions().end());
  EXPECT_TRUE(module->schedule().is_computation_scheduled(
      (*it)->called_computations()[0]));
  EXPECT_TRUE(module->schedule().is_computation_scheduled(
      (*it2)->called_computations()[0]));
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

class CompressingRematerializationTest : public RematerializationTestBase {
 protected:
  // A special shape size function, which pads the most minor dimension to 64.
  static int64_t ShapeSizePadMinorTo64(const Shape& shape) {
    if (shape.IsTuple()) {
      // Size of a tuple is 4 bytes.
      return 4;
    }
    Shape descending_shape =
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(shape);
    int64_t size =
        ShapeUtil::ByteSizeOfPrimitiveType(descending_shape.element_type());
    for (int64_t i = 0; i < descending_shape.rank(); ++i) {
      int64_t dim = descending_shape.dimensions(i);
      if (i == descending_shape.rank() - 1) {
        dim = RoundUpTo<int64_t>(dim, 64);
      }
      size *= dim;
    }
    return size;
  }

  // Swap the layout of the two most-minor dimensions if the second-minor
  // dimension is bigger than the most-minor dimension.
  static absl::StatusOr<Shape> ChooseCompactLayoutForShape(const Shape& shape) {
    if (shape.rank() != 2) {
      return shape;
    }
    Shape result = shape;
    Layout layout = result.layout();
    int64_t most_minor_index = layout.minor_to_major()[0];
    int64_t second_minor_index = layout.minor_to_major()[1];
    int64_t most_minor = result.dimensions(most_minor_index);
    int64_t second_minor = result.dimensions(second_minor_index);
    if (most_minor < second_minor) {
      Layout new_layout = layout;
      new_layout.set_minor_to_major(0, second_minor_index);
      new_layout.set_minor_to_major(1, most_minor_index);
      *result.mutable_layout() = new_layout;
    }
    return result;
  }

  absl::StatusOr<bool> RunHloRematerialization(int64_t memory_limit_bytes,
                                               HloModule* module,
                                               int64_t min_remat_size = 0) {
    TF_EXPECT_OK(verifier().Run(module).status());
    HloRematerialization::RematerializationModeConfig config(
        /*recompute=*/false, /*compress=*/true, /*host_offload=*/false);
    auto shape_size_func = [](const Shape& shape) {
      return ShapeSizePadMinorTo64(shape);
    };
    HloCostAnalysis cost_analysis(shape_size_func);
    HloRematerialization::Options options(
        cost_analysis, config, memory_limit_bytes,
        /*block_size_limit=*/1, /*block_rematerialization_factor=*/1,
        min_remat_size, ChooseCompactLayoutForShape,
        /*host_memory_offload_config=*/std::nullopt,
        /*async_threads=*/{});
    HloRematerialization::RematerializationSizes sizes;
    HloRematerialization remat(options, sizes);
    return remat.Run(module);
  }
};

// Test rematerialization only remats big buffer that pass certain limits.
TEST_F(CompressingRematerializationTest, OnlyRematBigBuffer) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

%add_float {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

ENTRY %entry {
  %param.0 = f32[] parameter(0)
  %constant = f32[] constant(0)
  %broadcast.0 = f32[64,2]{1,0} broadcast(f32[] %param.0), dimensions={}
  %broadcast.1 = f32[10,2]{1,0} broadcast(f32[] %param.0), dimensions={}
  %negate = f32[64,2]{1,0} negate(f32[64,2]{1,0} broadcast.0)
  %reduce.0 = f32[] reduce(f32[64,2]{1,0} %negate, f32[] %constant), dimensions={1, 0}, to_apply=%add_float
  %reduce.1 = f32[] reduce(f32[64,2]{1,0} %broadcast.0, f32[] %constant), dimensions={1, 0}, to_apply=%add_float
  %reduce.2 = f32[] reduce(f32[10,2]{1,0} %broadcast.1, f32[] %constant), dimensions={1, 0}, to_apply=%add_float
  %add = f32[] add(f32[] %reduce.0, f32[] %reduce.1)
  ROOT %add.2 = f32[] add(f32[] %add, f32[] %reduce.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Only rematerialize buffers which have shaep f32[64, 2]. Buffers with shape
  // f32[10, 2] are ignored.
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloRematerialization(
                                            /*memory_limit_bytes=*/30 * 1024,
                                            module.get(), 10 * 1024));
  EXPECT_TRUE(changed);
  HloInstruction* broadcast =
      module->entry_computation()->GetInstructionWithName("broadcast.0");
  HloInstruction* broadcast_2 =
      module->entry_computation()->GetInstructionWithName("broadcast.1");
  HloInstruction* reduce =
      module->entry_computation()->GetInstructionWithName("reduce.1");
  HloInstruction* reduce_2 =
      module->entry_computation()->GetInstructionWithName("reduce.2");
  EXPECT_THAT(reduce,
              op::Reduce(op::Copy(op::Copy(broadcast)), op::Constant()));
  EXPECT_THAT(reduce_2, op::Reduce(broadcast_2, op::Constant()));
}

// Test rematerialization of a single instruction.
TEST_F(CompressingRematerializationTest, SingleRemat) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

%add_float {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

ENTRY %entry {
  %param.0 = f32[] parameter(0)
  %constant = f32[] constant(0)
  %broadcast.0 = f32[64,2]{1,0} broadcast(f32[] %param.0), dimensions={}
  %negate = f32[64,2]{1,0} negate(f32[64,2]{1,0} broadcast.0)
  %reduce.0 = f32[] reduce(f32[64,2]{1,0} %negate, f32[] %constant), dimensions={1, 0}, to_apply=%add_float
  %reduce.1 = f32[] reduce(f32[64,2]{1,0} %broadcast.0, f32[] %constant), dimensions={1, 0}, to_apply=%add_float
  %add = f32[] add(f32[] %reduce.0, f32[] %reduce.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/30 * 1024, module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* broadcast =
      module->entry_computation()->GetInstructionWithName("broadcast.0");
  HloInstruction* reduce =
      module->entry_computation()->GetInstructionWithName("reduce.1");
  EXPECT_THAT(reduce,
              op::Reduce(op::Copy(op::Copy(broadcast)), op::Constant()));
}

// Test a pathological case where the peak memory is largely due to a single
// tensor (broadcast.0) and compressing it would actually increase the peak
// memory.
TEST_F(CompressingRematerializationTest, AvoidPathologicalCompress) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

%add_float {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

ENTRY %entry {
  %param.0 = f32[] parameter(0)
  %constant = f32[] constant(0)
  %broadcast.0 = f32[63,60]{1,0} broadcast(f32[] %param.0), dimensions={}
  %broadcast.1 = f32[16,64]{1,0} broadcast(f32[] %param.0), dimensions={}
  %reduce.0 = f32[] reduce(%broadcast.1, f32[] %constant), dimensions={1, 0}, to_apply=%add_float
  %reduce.1 = f32[] reduce(%broadcast.0, f32[] %constant), dimensions={1, 0}, to_apply=%add_float
  %add = f32[] add(f32[] %reduce.0, f32[] %reduce.1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/16 * 1024, module.get()));
  EXPECT_FALSE(changed);
  HloInstruction* broadcast =
      module->entry_computation()->GetInstructionWithName("broadcast.0");
  HloInstruction* reduce =
      module->entry_computation()->GetInstructionWithName("reduce.1");
  EXPECT_THAT(reduce, op::Reduce(broadcast, op::Constant()));
}

TEST_F(CompressingRematerializationTest, AllUsersUseSameCopy) {
  const std::string& hlo_string = R"(
HloModule fusion, is_scheduled=true

%add_float {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %x, f32[] %y)
}

ENTRY %entry {
  %param.0 = f32[] parameter(0)
  %constant = f32[] constant(0)
  %broadcast.0 = f32[64,2]{1,0} broadcast(f32[] %param.0), dimensions={}
  %negate = f32[64,2]{1,0} negate(f32[64,2]{1,0} broadcast.0)
  %reduce.0 = f32[] reduce(f32[64,2]{1,0} %negate, f32[] %constant), dimensions={1, 0}, to_apply=%add_float
  %reduce.1 = f32[] reduce(f32[64,2]{1,0} %negate, f32[] %constant), dimensions={1, 0}, to_apply=%add_float
  %reduce.2 = f32[] reduce(f32[64,2]{1,0} %broadcast.0, f32[] %constant), dimensions={1, 0}, to_apply=%add_float
  %add = f32[] add(f32[] %reduce.0, f32[] %reduce.1)
  %reduce.3 = f32[] reduce(f32[64,2]{1,0} %broadcast.0, f32[] %constant), dimensions={1, 0}, to_apply=%add_float
  %add.2 = f32[] add(f32[] %reduce.2, f32[] %reduce.3)
  ROOT %tuple = (f32[], f32[]) tuple (f32[] add, f32[] add.2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/30 * 1024, module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* broadcast =
      module->entry_computation()->GetInstructionWithName("broadcast.0");

  // Both reduces reuse the same copy instruction.
  HloInstruction* reduce_2 =
      module->entry_computation()->GetInstructionWithName("reduce.2");

  HloInstruction* reduce_3 =
      module->entry_computation()->GetInstructionWithName("reduce.3");

  EXPECT_THAT(reduce_2,
              op::Reduce(op::Copy(op::Copy(broadcast)), op::Constant()));

  EXPECT_THAT(reduce_3,
              op::Reduce(op::Copy(op::Copy(broadcast)), op::Constant()));
}

class OffloadingRematerializationTest : public RematerializationTestBase {
 protected:
  absl::StatusOr<bool> RunHloRematerialization(int64_t memory_limit_bytes,
                                               HloModule* module,
                                               int64_t min_remat_size = 0) {
    TF_EXPECT_OK(verifier().Run(module).status());
    if (!module->has_schedule()) {
      HloMemoryScheduler scheduler(
          [](const BufferValue& buffer) { return ByteSizeOf(buffer.shape()); },
          ComputationSchedulerToModuleScheduler(DefaultMemoryScheduler));
      TF_EXPECT_OK(scheduler.Run(module).status());
    }
    // Create a configuration where any compute is much much slower than any
    // number of number of copies.
    HloCostAnalysis::Options hlo_cost_analysis_options;
    hlo_cost_analysis_options.shape_size = [](const Shape& shape) {
      return ByteSizeOf(shape);
    };
    hlo_cost_analysis_options.set_flops_per_second(flops_per_second_);
    hlo_cost_analysis_options.set_transcendentals_per_second(
        transcendentals_per_second_);
    HloCostAnalysis cost_analysis(hlo_cost_analysis_options);
    HloRematerialization::RematerializationModeConfig config(
        /*recompute=*/false, /*compress=*/false, /*host_offload=*/true);
    HloRematerialization::HostMemoryOffloadConfig host_memory_offload_config(
        kHostMemorySpaceColor, copy_to_host_speed_, copy_from_host_speed_);
    HloRematerialization::Options options(
        cost_analysis, config, memory_limit_bytes,
        /*block_size_limit=*/1, /*block_rematerialization_factor=*/1,
        min_remat_size, /*compact_shape_function=*/nullptr,
        host_memory_offload_config,
        /*async_threads=*/{});
    HloRematerialization::RematerializationSizes sizes;
    HloRematerialization remat(options, sizes);
    return remat.Run(module);
  }
  void SetCopyToHostSpeed(float val) { copy_to_host_speed_ = val; }
  void SetCopyFromHostSpeed(float val) { copy_from_host_speed_ = val; }
  void SetFlopsPerSecond(float val) { flops_per_second_ = val; }
  void SetTranscendentalsPerSecond(float val) {
    transcendentals_per_second_ = val;
  }

  static constexpr const int64_t kHostMemorySpaceColor{5};

 private:
  float copy_to_host_speed_{1.0f};
  float copy_from_host_speed_{1.0f};
  float flops_per_second_{1.0f};
  float transcendentals_per_second_{1.0f};
};

TEST_F(OffloadingRematerializationTest, BasicSuccessfulHostOffload) {
  const std::string& hlo_string = R"(
HloModule MyModule, is_scheduled=true, entry_computation_layout={(f32[1024]{0}, f32[1024]{0})->f32[1024]{0}}

ENTRY MyModule {
  param_0 = f32[1024]{0} parameter(0)
  param_1 = f32[1024]{0} parameter(1)
  res_3 = f32[1024]{0} add(param_0, param_1)
  res_4 = f32[1024]{0} tanh(res_3)
  res_5 = f32[1024]{0} tanh(res_4)
  res_6 = f32[1024]{0} tanh(res_5)
  res_7 = f32[1024]{0} add(res_6, res_6)
  res_8 = f32[1024]{0} add(res_7, res_5)
  res_9 = f32[1024]{0} add(res_8, res_4)
  res_10 = f32[1024]{0} add(res_9, res_3)
  ROOT res_11 = f32[1024]{0} tanh(res_10)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Set some "hardware" constants so that we can test that instructions are
  // placed in the places we expect.
  SetCopyToHostSpeed(4.0 * 1024);
  SetCopyFromHostSpeed(4.0 * 1024);
  SetFlopsPerSecond(2 * 1024);
  SetTranscendentalsPerSecond(2 * 1024);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/10 * 1024, module.get()));
  ASSERT_TRUE(changed);

  // The module should still have a schedule.
  ASSERT_TRUE(module->has_schedule());

  // Verify that exactly two instructions are rematerialized.
  auto res_3_matcher = op::Add(op::Parameter(), op::Parameter());
  auto res_3_rematted_matcher = op::AsyncCopy(
      xla::Layout::kDefaultMemorySpace, kHostMemorySpaceColor,
      op::AsyncCopy(kHostMemorySpaceColor, xla::Layout::kDefaultMemorySpace,
                    res_3_matcher));
  auto res_4_matcher = op::Tanh(res_3_matcher);
  auto res_4_rematted_matcher = op::AsyncCopy(
      xla::Layout::kDefaultMemorySpace, kHostMemorySpaceColor,
      op::AsyncCopy(kHostMemorySpaceColor, xla::Layout::kDefaultMemorySpace,
                    res_4_matcher));
  auto res_5_matcher = op::Tanh(res_4_matcher);
  auto res_6_matcher = op::Tanh(res_5_matcher);
  auto res_7_matcher = op::Add(res_6_matcher, res_6_matcher);
  auto res_8_matcher = op::Add(res_7_matcher, res_5_matcher);
  auto res_9_matcher = op::Add(res_8_matcher, res_4_rematted_matcher);
  auto res_10_matcher = op::Add(res_9_matcher, res_3_rematted_matcher);

  const auto instruction_sequence =
      module->schedule().sequence(module->entry_computation());
  ASSERT_THAT(instruction_sequence.instructions().back(),
              op::Tanh(res_10_matcher));
}

TEST_F(OffloadingRematerializationTest, SkipOffloadWhenBitcastIsInvolved) {
  const std::string& hlo_string = R"(
HloModule MyModule, is_scheduled=true, entry_computation_layout={(f32[1024]{0}, f32[1024]{0})->f32[1024]{0}}

ENTRY MyModule {
  param_0 = f32[1024]{0} parameter(0)
  param_1 = f32[1024]{0} parameter(1)
  res_3 = f32[1024]{0} add(param_0, param_1)
  bitcast = f32[1024]{0} bitcast(res_3)
  res_4 = f32[1024]{0} tanh(res_3)
  res_5 = f32[1024]{0} tanh(res_4)
  res_6 = f32[1024]{0} tanh(res_5)
  res_7 = f32[1024]{0} add(res_6, res_6)
  res_8 = f32[1024]{0} add(res_7, res_5)
  res_9 = f32[1024]{0} add(res_8, res_4)
  res_10 = f32[1024]{0} add(res_9, bitcast)
  ROOT res_11 = f32[1024]{0} tanh(res_10)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Set some "hardware" constants so that we can test that instructions are
  // placed in the places we expect.
  SetCopyToHostSpeed(4.0 * 1024);
  SetCopyFromHostSpeed(4.0 * 1024);
  SetFlopsPerSecond(2 * 1024);
  SetTranscendentalsPerSecond(2 * 1024);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/10 * 1024, module.get()));
  ASSERT_TRUE(changed);

  // The module should still have a schedule.
  ASSERT_TRUE(module->has_schedule());

  // Verify that exactly one instruction is rematerialized. Once we handle
  // bitcasts, res_3 can be rematted, but not currently.
  auto res_3_matcher = op::Add(op::Parameter(), op::Parameter());
  auto res_4_matcher = op::Tanh(res_3_matcher);
  auto res_4_rematted_matcher = op::AsyncCopy(
      xla::Layout::kDefaultMemorySpace, kHostMemorySpaceColor,
      op::AsyncCopy(kHostMemorySpaceColor, xla::Layout::kDefaultMemorySpace,
                    res_4_matcher));
  auto res_5_matcher = op::Tanh(res_4_matcher);
  auto res_6_matcher = op::Tanh(res_5_matcher);
  auto res_7_matcher = op::Add(res_6_matcher, res_6_matcher);
  auto res_8_matcher = op::Add(res_7_matcher, res_5_matcher);
  auto res_9_matcher = op::Add(res_8_matcher, res_4_rematted_matcher);
  auto res_10_matcher = op::Add(res_9_matcher, op::Bitcast(res_3_matcher));

  const auto instruction_sequence =
      module->schedule().sequence(module->entry_computation());
  ASSERT_THAT(instruction_sequence.instructions().back(),
              op::Tanh(res_10_matcher));
}

class IndirectUseTest : public RecomputeAndCompressHloRematerializationTest,
                        public ::testing::WithParamInterface<bool> {};

TEST_P(IndirectUseTest, IndirectUseRematerialized) {
  // Test that an rematerializable instruction is rematerialized if it has
  // indirect use
  // Module:
  //
  // Entry computation:
  //   F32[] %param = {...}
  //   F32[1024] %bcast = broadcast(%param)
  //   F32[1024] %add_1 = add(%bcast, bcast)
  //   F32[1024] %call = call(Subcomputation, {%add_1})
  //   F32[1024] %add_2 = add(%bcast, call)
  //   {F32[1024], F32[1024]} %tuple = tuple(%bcast, %add_2)
  //   F32[1024] %gte = GetTupleElement(%tuple, 0)
  //   F32[1024] %negate = negate(%gte)
  //
  // Subcomputation:
  //   F32[1024] %param = {...}
  //   F32[2048] %concat = concat({%param, %param})
  //   F32[1024] %slice = slice(%concat)
  //
  // The value %bcast is live across the call and rematerialization of %bcast
  // across that point would reduce peak memory use by 4KB.
  //
  // This test is parameterized on whether the broadcast has an indirect use
  // or not. The indirect use is controlled by the index of the GetTupleElement
  // instruction. If the element is 0, then the %negate operand aliases %bcast
  // (ie %bcast is used indirectly by %negate), otherwise the %negate operand
  // aliases %add_2.
  const bool indirectly_used = GetParam();
  auto module = CreateNewVerifiedModule();

  HloComputation* subcomputation = nullptr;
  {
    auto builder = HloComputation::Builder(TestName() + ".subcomputation");
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1024_shape_, "param"));
    auto concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {2048}), {param, param},
        /*dimension=*/0));
    builder.AddInstruction(HloInstruction::CreateSlice(
        vec1024_shape_, concat, /*start_indices=*/{0},
        /*limit_indices=*/{1024}, /*strides=*/{1}));
    subcomputation = module->AddEmbeddedComputation(builder.Build());
  }

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  auto bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(vec1024_shape_, param, {}));
  auto add_1 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, bcast, bcast));
  auto call_1 = builder.AddInstruction(
      HloInstruction::CreateCall(vec1024_shape_, {add_1}, subcomputation));
  auto add_2 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, bcast, call_1));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({bcast, add_2}));
  auto gte = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      vec1024_shape_, tuple, indirectly_used ? 0 : 1));
  builder.AddInstruction(
      HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kNegate, gte));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  EXPECT_EQ(entry_computation->instruction_count(), 8);

  // Pick a memory limit some where between 24KB (initial peak memory
  // including parameter and output) and 20KB (peak memory possible with
  // rematerialization).
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/22 * 1024, module.get()));
  // Rematerialization should only occur if the rematerializable instruction
  // has no indirect uses.
  if (indirectly_used) {
    EXPECT_TRUE(changed);
    EXPECT_EQ(entry_computation->instruction_count(), 3);
  } else {
    EXPECT_TRUE(changed);
    EXPECT_EQ(entry_computation->instruction_count(), 9);
  }
  CheckForRematInInstructionNames(
      ::testing::UnitTest::GetInstance()->current_test_info()->name());
}

INSTANTIATE_TEST_SUITE_P(IndirectUseTestInstantiation, IndirectUseTest,
                         ::testing::Values(true, false));

}  // namespace

}  // namespace xla
