/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/double_buffer_loop_unrolling.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

int64_t CountInstructions(HloComputation& computation, HloOpcode opcode) {
  int64_t count = 0;
  hlo_query::ForEachInstructionWithOpcode(
      computation, opcode, [&count](HloInstruction* instr) { count++; });
  return count;
}

int64_t CountInstructions(HloModule& module, HloOpcode opcode) {
  int64_t count = 0;
  hlo_query::ForEachInstructionWithOpcode(
      module, opcode, [&count](HloInstruction* instr) { count++; });
  return count;
}

using GpuLoopDoubleBufferTransformerTest = HloHardwareIndependentTestBase;

TEST_F(GpuLoopDoubleBufferTransformerTest,
       AutoUnrollLoopWhenCollectivesArePresent) {
  absl::string_view kModuleString = R"(
HloModule m
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}

body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  all-reduce-start = f32[] all-reduce-start(param_0), channel_id=8, replica_groups={{0}}, to_apply=ar_add, backend_config={"collective_backend_config": {"is_sync": false}}
  one = s32[] constant(1)
  all-reduce-done = f32[] all-reduce-done(all-reduce-start)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(all-reduce-done, cond_plus_1)
}

ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body,
      backend_config={"known_trip_count":{"n":"10"},
                      "known_induction_variable":{"tuple_index":"1"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  HloPassPipeline pipeline("double-buffering-pipeline");
  DoubleBufferLoopUnrolling unroller(
      DoubleBufferLoopUnrolling::UnrollStrategy::kAuto);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, unroller.Run(module.get()));

  EXPECT_TRUE(changed);

  HloInstruction* while_instruction = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_trip_count().n(), 5);
  EXPECT_EQ(config.known_induction_variable().tuple_index(), 1);
  EXPECT_EQ(CountInstructions((*while_instruction->while_body()),
                              HloOpcode::kAllReduceStart),
            2);
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       DoNotAutoUnrollLoopWhenCollectivesAreNotPresent) {
  absl::string_view kModuleString = R"(
HloModule m
condition {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (s32[]) tuple(cond_plus_1)
}

ENTRY main {
  param_0 = s32[] constant(0)
  tuple = (s32[]) tuple(param_0)
  ROOT while = (s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling unroller(
      DoubleBufferLoopUnrolling::UnrollStrategy::kAuto);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, unroller.Run(module.get()));

  EXPECT_FALSE(changed);

  HloInstruction* while_instruction = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_trip_count().n(), 10);
}

TEST_F(GpuLoopDoubleBufferTransformerTest, FullUnrollOddTripCountTest) {
  const char* const kModuleString = R"(
HloModule all_gather_overlapping
condition {
  input_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=3
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
 input_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) parameter(0)
 param_0 = f32[1,128] get-tuple-element(input_tuple), index=0
 param_1 = f32[2,128] get-tuple-element(input_tuple), index=2
 cond = s32[] get-tuple-element(input_tuple), index=3
 c0 = f32[] constant(0)
 splat_c0 = f32[1,128] broadcast(c0), dimensions={}
 add = f32[1,128] add(splat_c0, param_0)
 all-gather-start = (f32[1,128], f32[2,128]) all-gather-start(add), channel_id=1337, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true
 c1_s32 = s32[] constant(1)
 c0_s32 = s32[] constant(0)
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 dynamic-slice = f32[1,128] dynamic-slice(param_1, c1_s32, c0_s32), dynamic_slice_sizes={1,128}
 all-gather-done = f32[2,128] all-gather-done(all-gather-start)
 ROOT output_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) tuple(param_0, dynamic-slice, all-gather-done, cond_plus_1)
}

ENTRY main {
 param_0 = f32[1,128] parameter(0)
 param_1 = f32[2,128] parameter(1)
 param_2 = s32[] constant(0)
 tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) tuple(param_0, param_0, param_1, param_2)
 ROOT while = (f32[1,128], f32[1,128], f32[2,128], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"11"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kFullUnroll);
  TupleSimplifier tuple_simp;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, double_buffer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(changed, tuple_simp.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* while_instruction = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  int64_t exact_trip_count = config.known_trip_count().n();
  EXPECT_EQ(exact_trip_count, 1);
  EXPECT_EQ(CountInstructions((*while_instruction->while_body()),
                              HloOpcode::kAllGatherStart),
            11);
  EXPECT_EQ(CountInstructions((*module), HloOpcode::kAllGatherStart), 11);
}

TEST_F(GpuLoopDoubleBufferTransformerTest, FullUnrollEvenTripCountTest) {
  const char* const kModuleString = R"(
HloModule all_gather_overlapping
condition {
  input_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=3
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
 input_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) parameter(0)
 param_0 = f32[1,128] get-tuple-element(input_tuple), index=0
 param_1 = f32[2,128] get-tuple-element(input_tuple), index=2
 cond = s32[] get-tuple-element(input_tuple), index=3
 c0 = f32[] constant(0)
 splat_c0 = f32[1,128] broadcast(c0), dimensions={}
 add = f32[1,128] add(splat_c0, param_0)
 // Start all-gather communication
 all-gather-start = (f32[1,128], f32[2,128]) all-gather-start(add), channel_id=1337, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true
 c1_s32 = s32[] constant(1)
 c0_s32 = s32[] constant(0)
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 dynamic-slice = f32[1,128] dynamic-slice(param_1, c1_s32, c0_s32), dynamic_slice_sizes={1,128}
 all-gather-done = f32[2,128] all-gather-done(all-gather-start)
 ROOT output_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) tuple(param_0, dynamic-slice, all-gather-done, cond_plus_1)
}

ENTRY main {
 param_0 = f32[1,128] parameter(0)
 param_1 = f32[2,128] parameter(1)
 param_2 = s32[] constant(0)
 tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) tuple(param_0, param_0, param_1, param_2)
 ROOT while = (f32[1,128], f32[1,128], f32[2,128], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kFullUnroll);
  TupleSimplifier tuple_simp;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, double_buffer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(changed, tuple_simp.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* while_instruction;
  for (auto instr : module->entry_computation()->instructions()) {
    if (HloPredicateIsOp<HloOpcode::kWhile>(instr)) {
      while_instruction = instr;
    }
  }
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  int64_t exact_trip_count = config.known_trip_count().n();
  EXPECT_EQ(exact_trip_count, 1);
  EXPECT_EQ(CountInstructions((*while_instruction->while_body()),
                              HloOpcode::kAllGatherStart),
            10);
  EXPECT_EQ(CountInstructions((*module), HloOpcode::kAllGatherStart), 10);
}

TEST_F(GpuLoopDoubleBufferTransformerTest, UnrolledLoopEvenTripCount) {
  const char* const kModuleString = R"(
HloModule all_gather_overlapping
condition {
  input_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=3
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
 input_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) parameter(0)
 param_0 = f32[1,128] get-tuple-element(input_tuple), index=0
 param_1 = f32[2,128] get-tuple-element(input_tuple), index=2
 cond = s32[] get-tuple-element(input_tuple), index=3
 c0 = f32[] constant(0)
 splat_c0 = f32[1,128] broadcast(c0), dimensions={}
 add = f32[1,128] add(splat_c0, param_0)
 // Start all-gather communication
 all-gather-start = (f32[1,128], f32[2,128]) all-gather-start(add), channel_id=1337, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true
 // Intertwined with the all-gather communication, an operation happens which
 // depends on param_1, but crucially has a different output shape (which
 // excludes reusing param_1's buffer for its output).
 c1_s32 = s32[] constant(1)
 c0_s32 = s32[] constant(0)
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 dynamic-slice = f32[1,128] dynamic-slice(param_1, c1_s32, c0_s32), dynamic_slice_sizes={1,128}
 // The all-gather communication finishes
 all-gather-done = f32[2,128] all-gather-done(all-gather-start)
 ROOT output_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) tuple(param_0, dynamic-slice, all-gather-done, cond_plus_1)
}

ENTRY main {
 param_0 = f32[1,128] parameter(0)
 param_1 = f32[2,128] parameter(1)
 param_2 = s32[] constant(0)
 tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) tuple(param_0, param_0, param_1, param_2)
 ROOT while = (f32[1,128], f32[1,128], f32[2,128], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer;
  TupleSimplifier tuple_simp;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, double_buffer.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(changed, tuple_simp.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* while_instruction = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  int64_t exact_trip_count = config.known_trip_count().n();
  // We expect that after unrolling, the total trip count is half of original
  // count.
  EXPECT_EQ(exact_trip_count, 5);
  // We expect that after unrolling, there should be 2 allgather starts,
  // both in while body.
  EXPECT_EQ(CountInstructions((*while_instruction->while_body()),
                              HloOpcode::kAllGatherStart),
            2);
  EXPECT_EQ(CountInstructions((*module), HloOpcode::kAllGatherStart), 2);
}

TEST_F(GpuLoopDoubleBufferTransformerTest, UnrolledLoopOddTripCount) {
  const char* const kModuleString = R"(
HloModule all_gather_overlapping
condition {
  input_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=3
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
 input_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) parameter(0)
 param_0 = f32[1,128] get-tuple-element(input_tuple), index=0
 param_1 = f32[2,128] get-tuple-element(input_tuple), index=2
 cond = s32[] get-tuple-element(input_tuple), index=3
 c0 = f32[] constant(0)
 splat_c0 = f32[1,128] broadcast(c0), dimensions={}
 add = f32[1,128] add(splat_c0, param_0)
 // Start all-gather communication
 all-gather-start = (f32[1,128], f32[2,128]) all-gather-start(add), channel_id=1337, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true
 // Intertwined with the all-gather communication, an operation happens which
 // depends on param_1, but crucially has a different output shape (which
 // excludes reusing param_1's buffer for its output).
 c1_s32 = s32[] constant(1)
 c0_s32 = s32[] constant(0)
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 dynamic-slice = f32[1,128] dynamic-slice(param_1, c1_s32, c0_s32), dynamic_slice_sizes={1,128}
 // The all-gather communication finishes
 all-gather-done = f32[2,128] all-gather-done(all-gather-start)
 ROOT output_tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) tuple(param_0, dynamic-slice, all-gather-done, cond_plus_1)
}

ENTRY main {
 param_0 = f32[1,128] parameter(0)
 param_1 = f32[2,128] parameter(1)
 param_2 = s32[] constant(0)
 tuple = (f32[1,128], f32[1,128], f32[2,128], s32[]) tuple(param_0, param_0, param_1, param_2)
 ROOT while = (f32[1,128], f32[1,128], f32[2,128], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"11"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer;
  TupleSimplifier tuple_simp;
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(tuple_simp.Run(module.get()), absl_testing::IsOkAndHolds(true));

  // We expect that for the while loop, no further copy needs to be added to the
  // module.
  HloInstruction* while_instruction = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  int64_t exact_trip_count = config.known_trip_count().n();
  // We expect that after unrolling, the total trip count is half of original
  // count.
  EXPECT_EQ(exact_trip_count, 5);

  // We expect that after unrolling, there should be 3 allgather starts,
  // 1 in parent computation, 2 in while body.
  EXPECT_EQ(CountInstructions((*while_instruction->while_body()),
                              HloOpcode::kAllGatherStart),
            2);
  EXPECT_EQ(CountInstructions((*module), HloOpcode::kAllGatherStart), 3);

  // We expect that after unrolling, the third operand of the input tuple should
  // be the peeled allgather done.
  EXPECT_EQ(while_instruction->operand(0)->operand(2)->opcode(),
            HloOpcode::kAllGatherDone);
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       UnrolledLoopNoControlDepsForConstantAdd) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
 input_tuple = (f32[], s32[]) parameter(0)
 param_0 = f32[] get-tuple-element(input_tuple), index=0
 cond = s32[] get-tuple-element(input_tuple), index=1
 c2 = f32[] constant(2)
 add = f32[] add(c2, param_0)
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output_tuple = (f32[], s32[]) tuple(add, cond_plus_1)
}

ENTRY main {
 param_0 = f32[] parameter(0)
 param_2 = s32[] constant(0)
 tuple = (f32[], s32[]) tuple(param_0, param_2)
 ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"11"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer;
  TupleSimplifier tuple_simp;
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(tuple_simp.Run(module.get()), absl_testing::IsOkAndHolds(true));

  HloInstruction* while_instruction = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  int64_t exact_trip_count = config.known_trip_count().n();
  // We expect that after unrolling, the total trip count is half of original
  // count.
  EXPECT_EQ(exact_trip_count, 5);

  // We expect that after unrolling, there should be 4 adds
  EXPECT_EQ(
      CountInstructions((*while_instruction->while_body()), HloOpcode::kAdd),
      4);

  // We expect that after unrolling, the first operand of the output tuple
  // should not have any control dependency since it's a elementwise add with a
  // constant operand.
  EXPECT_EQ(while_instruction->while_body()
                ->root_instruction()
                ->operand(0)
                ->control_predecessors()
                .size(),
            0);
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       UnrolledLoopNoControlDepsForCollective) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}

body {
 input_tuple = (f32[], s32[]) parameter(0)
 param_0 = f32[] get-tuple-element(input_tuple), index=0
 cond = s32[] get-tuple-element(input_tuple), index=1
 all-reduce-start = f32[] all-reduce-start(param_0), channel_id=8, replica_groups={{0}}, to_apply=ar_add, backend_config={"collective_backend_config": {"is_sync": false}}
 one = s32[] constant(1)
 all-reduce-done = f32[] all-reduce-done(all-reduce-start)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output_tuple = (f32[], s32[]) tuple(all-reduce-done, cond_plus_1)
}

ENTRY main {
 param_0 = f32[] parameter(0)
 param_2 = s32[] constant(0)
 tuple = (f32[], s32[]) tuple(param_0, param_2)
 ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer;
  TupleSimplifier tuple_simp;
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(tuple_simp.Run(module.get()), absl_testing::IsOkAndHolds(true));

  HloInstruction* while_instruction = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  int64_t exact_trip_count = config.known_trip_count().n();
  // We expect that after unrolling, the total trip count is half of original
  // count.
  EXPECT_EQ(exact_trip_count, 5);

  // We expect that after unrolling, there should be 2 all-reduce-starts
  EXPECT_EQ(CountInstructions((*while_instruction->while_body()),
                              HloOpcode::kAllReduceStart),
            2);
  absl::flat_hash_set<int64_t> channel_ids;
  hlo_query::ForEachInstructionWithOpcode(
      *while_instruction->while_body(), HloOpcode::kAllReduceStart,
      [&channel_ids](HloInstruction* ar) {
        // We expect that after unrolling, all-reduces should not have any
        // control deps.
        EXPECT_EQ(ar->control_predecessors().size(), 0);
        channel_ids.insert(*(ar->channel_id()));
      });
  // we expect that all 2 all-reduces will have different channel ids.
  EXPECT_EQ(channel_ids.size(), 2);
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       FullyUnrolledLoopNoControlDepsForCollective) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}

body {
 input_tuple = (f32[], s32[]) parameter(0)
 param_0 = f32[] get-tuple-element(input_tuple), index=0
 cond = s32[] get-tuple-element(input_tuple), index=1
 all-reduce-start = f32[] all-reduce-start(param_0), channel_id=8, replica_groups={{0}}, to_apply=ar_add, backend_config={"collective_backend_config": {"is_sync": false}}
 one = s32[] constant(1)
 all-reduce-done = f32[] all-reduce-done(all-reduce-start)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output_tuple = (f32[], s32[]) tuple(all-reduce-done, cond_plus_1)
}

ENTRY main {
 param_0 = f32[] parameter(0)
 param_2 = s32[] constant(0)
 tuple = (f32[], s32[]) tuple(param_0, param_2)
 ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kFullUnroll);
  TupleSimplifier tuple_simp;
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(tuple_simp.Run(module.get()), absl_testing::IsOkAndHolds(true));

  HloInstruction* while_instruction = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  int64_t exact_trip_count = config.known_trip_count().n();
  EXPECT_EQ(exact_trip_count, 1);

  // We expect that after unrolling, there should be 10 all-reduce-starts
  EXPECT_EQ(CountInstructions((*while_instruction->while_body()),
                              HloOpcode::kAllReduceStart),
            10);
  absl::flat_hash_set<int64_t> channel_ids;
  hlo_query::ForEachInstructionWithOpcode(
      *while_instruction->while_body(), HloOpcode::kAllReduceStart,
      [&channel_ids](HloInstruction* ar) {
        // We expect that after unrolling, all-reduces should not have any
        // control deps.
        EXPECT_EQ(ar->control_predecessors().size(), 0);
        channel_ids.insert(*(ar->channel_id()));
      });
  // we expect that all 10 all-reduces will have different channel ids.
  EXPECT_EQ(channel_ids.size(), 10);
}

TEST_F(GpuLoopDoubleBufferTransformerTest, ControlDepsCopiedWhenUnrolled) {
  // Test control dependencies are correctly copied when unrolling.
  const char* const kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
 input_tuple = (f32[], s32[]) parameter(0)
 param_0 = f32[] get-tuple-element(input_tuple), index=0
 cond = s32[] get-tuple-element(input_tuple), index=1
 c2 = f32[] constant(2)
 multiply = f32[] multiply(c2, param_0), control-predecessors={cond}
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output_tuple = (f32[], s32[]) tuple(multiply, cond_plus_1)
}

ENTRY main {
 param_0 = f32[] parameter(0)
 param_2 = s32[] constant(0)
 tuple = (f32[], s32[]) tuple(param_0, param_2)
 ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"11"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  HloInstruction* while_instruction = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());

  // After unrolling, there should be 2 multiplies, each with a GTE control
  // predecessor.
  EXPECT_EQ(CountInstructions((*while_instruction->while_body()),
                              HloOpcode::kMultiply),
            2);
  for (HloInstruction* instr :
       while_instruction->while_body()->MakeInstructionPostOrder()) {
    if (instr->opcode() != HloOpcode::kMultiply) {
      continue;
    }
    EXPECT_EQ(instr->control_predecessors().size(), 1);
    EXPECT_EQ(instr->control_predecessors()[0]->opcode(),
              HloOpcode::kGetTupleElement);
    EXPECT_EQ(instr->control_predecessors()[0]->parent(), instr->parent());
  }

  // After unrolling, there should be 1 multiply in the parent computation.
  EXPECT_EQ(
      CountInstructions((*module->entry_computation()), HloOpcode::kMultiply),
      1);
  HloInstruction* multiply_instruction =
      hlo_query::GetFirstInstructionWithOpcode(*module->entry_computation(),
                                               HloOpcode::kMultiply);
  EXPECT_EQ(multiply_instruction->control_predecessors().size(), 1);
  EXPECT_EQ(multiply_instruction->control_predecessors()[0]->opcode(),
            HloOpcode::kGetTupleElement);
  EXPECT_EQ(multiply_instruction->control_predecessors()[0]->parent(),
            multiply_instruction->parent());
}

// The following 2 tests also address the regression described here:
// https://github.com/openxla/xla/issues/6353
TEST_F(GpuLoopDoubleBufferTransformerTest, NestedWhileLoopRemainsFlattened) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_nested_while_loop_remains_flattened

condition_nested {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body_nested {
 input_tuple = (s32[]) parameter(0)
 cond = s32[] get-tuple-element(input_tuple), index=0
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output = (s32[]) tuple(cond_plus_1)
}

condition {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (s32[]) parameter(0)
  ROOT output = (s32[]) while(input_tuple), condition=condition_nested, body=body_nested
}

ENTRY main {
 param_0 = (s32[]) parameter(0)
 ROOT while = (s32[]) while(param_0), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  absl::flat_hash_set<const HloComputation*> while_loops_callees;

  hlo_query::ForEachInstructionWithOpcode(
      *module, HloOpcode::kWhile,
      [&while_loops_callees](HloInstruction* instr) {
        EXPECT_TRUE(
            while_loops_callees.insert(instr->while_condition()).second);
        EXPECT_TRUE(while_loops_callees.insert(instr->while_body()).second);
      });

  // We expect that the nested while loop has been duplicated, along with its
  // associated computations.
  EXPECT_EQ(while_loops_callees.size(), 6);
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       NestedWhileLoopRemainsFlattenedOddTripCount) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_nested_while_loop_remains_flattened

condition_nested {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body_nested {
 input_tuple = (s32[]) parameter(0)
 cond = s32[] get-tuple-element(input_tuple), index=0
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output = (s32[]) tuple(cond_plus_1)
}

condition {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (s32[]) parameter(0)
  ROOT output = (s32[]) while(input_tuple), condition=condition_nested, body=body_nested
}

ENTRY main {
 param_0 = (s32[]) parameter(0)
 ROOT while = (s32[]) while(param_0), condition=condition, body=body, backend_config={"known_trip_count":{"n":"11"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  absl::flat_hash_set<const HloComputation*> while_loops_callees;

  hlo_query::ForEachInstructionWithOpcode(
      *module, HloOpcode::kWhile,
      [&while_loops_callees](HloInstruction* instr) {
        EXPECT_TRUE(
            while_loops_callees.insert(instr->while_condition()).second);
        EXPECT_TRUE(while_loops_callees.insert(instr->while_body()).second);
      });

  // We expect that the nested while loop has been duplicated, along with its
  // associated computations.
  EXPECT_EQ(while_loops_callees.size(), 8);
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       NestedWhileLoopRemainsFlattenedWhenFullyUnrolled) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_nested_while_loop_remains_flattened

condition_nested {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body_nested {
 input_tuple = (s32[]) parameter(0)
 cond = s32[] get-tuple-element(input_tuple), index=0
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output = (s32[]) tuple(cond_plus_1)
}

condition {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (s32[]) parameter(0)
  ROOT output = (s32[]) while(input_tuple), condition=condition_nested, body=body_nested
}

ENTRY main {
 param_0 = (s32[]) parameter(0)
 ROOT while = (s32[]) while(param_0), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kFullUnroll);
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  absl::flat_hash_set<const HloComputation*> while_loops_callees;

  hlo_query::ForEachInstructionWithOpcode(
      *module, HloOpcode::kWhile,
      [&while_loops_callees](HloInstruction* instr) {
        EXPECT_TRUE(
            while_loops_callees.insert(instr->while_condition()).second);
        EXPECT_TRUE(while_loops_callees.insert(instr->while_body()).second);
      });

  hlo_query::ForEachInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile,
      [](HloInstruction* instr) {
        TF_ASSERT_OK_AND_ASSIGN(
            WhileLoopBackendConfig config,
            instr->backend_config<WhileLoopBackendConfig>());
        int64_t exact_trip_count = config.known_trip_count().n();
        EXPECT_EQ(exact_trip_count, 1);
      });

  // We expect that the nested while loop has been fully duplicated 10
  // times. The one outer while loop still remains so that's 11 while
  // instructions. We check whether there are 22 distinct computations for
  // each while loop body and condition.
  EXPECT_EQ(while_loops_callees.size(), 22);
}

TEST_F(GpuLoopDoubleBufferTransformerTest, NestedWhileLoopAreUnrolled) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_nested_are_unrolled
condition_nested {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
body_nested {
 input_tuple = (s32[]) parameter(0)
 cond = s32[] get-tuple-element(input_tuple), index=0
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output = (s32[]) tuple(cond_plus_1)
}
condition {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
body {
  input_tuple = (s32[]) parameter(0)
  ROOT output = (s32[]) while(input_tuple), condition=condition_nested, body=body_nested, backend_config={"known_trip_count":{"n":"11"}}
}
ENTRY main {
 param_0 = (s32[]) parameter(0)
 ROOT while = (s32[]) while(param_0), condition=condition, body=body, backend_config={"known_trip_count":{"n":"11"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer;
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  int64_t num_whiles = 0;
  hlo_query::ForEachInstructionWithOpcode(
      *module, HloOpcode::kWhile, [&num_whiles](HloInstruction* instr) {
        EXPECT_EQ(instr->backend_config<WhileLoopBackendConfig>()
                      ->known_trip_count()
                      .n(),
                  5);
        ++num_whiles;
      });
  // We expect the number of while loops to be 4 in total after unrolling.
  EXPECT_EQ(num_whiles, 4);
}

TEST_F(GpuLoopDoubleBufferTransformerTest, NestedWhileLoopAreFullyUnrolled) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_nested_are_unrolled
condition_nested {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
body_nested {
 input_tuple = (s32[]) parameter(0)
 cond = s32[] get-tuple-element(input_tuple), index=0
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output = (s32[]) tuple(cond_plus_1)
}
condition {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
body {
  input_tuple = (s32[]) parameter(0)
  ROOT output = (s32[]) while(input_tuple), condition=condition_nested, body=body_nested, backend_config={"known_trip_count":{"n":"11"}}
}
ENTRY main {
 param_0 = (s32[]) parameter(0)
 ROOT while = (s32[]) while(param_0), condition=condition, body=body, backend_config={"known_trip_count":{"n":"11"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kFullUnroll);
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  int64_t num_whiles = 0;
  hlo_query::ForEachInstructionWithOpcode(
      *module, HloOpcode::kWhile, [&num_whiles](HloInstruction* instr) {
        EXPECT_EQ(instr->backend_config<WhileLoopBackendConfig>()
                      ->known_trip_count()
                      .n(),
                  1);
        ++num_whiles;
      });
  EXPECT_EQ(num_whiles, 12);
}

TEST_F(GpuLoopDoubleBufferTransformerTest, WhileLoopWithCollectivePermute) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}
body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  collective-permute = f32[] collective-permute(param_0), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(collective-permute, cond_plus_1)
}
ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer);
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  VLOG(1) << module->ToString();
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: %body {{.+}} {
    // CHECK:   %[[cp1:.+]] = {{.+}} collective-permute({{.+}}), {{.+}}
    // CHECK:   %[[out1:.+]] = {{.+}} tuple({{.*}}%[[cp1]], {{.*}})
    // CHECK:   %[[param2:.+]] = {{.+}} get-tuple-element({{.*}}%[[out1]]), index=0
    // CHECK:   %[[cp2:.+]] = {{.+}} collective-permute({{.*}}%[[param2]]), {{.+}}
    // CHECK:   ROOT {{.+}} = {{.+}} tuple({{.*}}%[[cp2]], {{.*}})
    // CHECK: }
    // CHECK: ENTRY %main {{.+}} {
    // CHECK-NOT: collective-permute
    // CHECK: }
  )"));
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       WhileLoopWithCollectivePermutePeeled) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(15)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}
body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  collective-permute = f32[] collective-permute(param_0), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,0}}
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(collective-permute, cond_plus_1)
}
ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"15"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer);
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));
  VLOG(1) << module->ToString();
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: %body
    // CHECK:   %[[cp1:.+]] = {{.+}} collective-permute({{.*}}), {{.+}}
    // CHECK:   %[[out1:.+]] = {{.+}} tuple({{.*}}%[[cp1]], {{.*}})
    // CHECK:   %[[param2:.+]] = {{.+}} get-tuple-element({{.*}}%[[out1]])
    // CHECK:   %[[cp2:.+]] = {{.+}} collective-permute({{.*}}), {{.+}}
    // CHECK:   ROOT {{.+}} = {{.+}} tuple({{.*}}%[[cp2]], {{.*}})
    // CHECK: ENTRY %main {{.+}} {
    // CHECK:   %[[cp_peeled:.+]] = {{.+}} collective-permute({{.*}}), {{.+}}
    // CHECK:   %[[out_peeled:.+]] = {{.+}} tuple({{.*}}%[[cp_peeled]], {{.*}})
    // CHECK:   %[[while:.+]] = {{.+}} while({{.*}}%[[out_peeled]])
    // CHECK: }
    )"));
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       WhileLoopWithCollectivePermuteBackwardCycle) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(14)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}
body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  collective-permute = f32[] collective-permute(param_0), channel_id=1, source_target_pairs={{0,7},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6}}
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(collective-permute, cond_plus_1)
}
ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"14"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer);
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: %body
    // CHECK:   %[[cp1:.+]] = f32[] collective-permute(%param_0), {{.+}}
    // CHECK:   %[[out1:.+]] = {{.+}} tuple({{.*}}%[[cp1]], {{.*}})
    // CHECK:   %[[param2:.+]] = {{.+}} get-tuple-element({{.*}}%[[out1]]), index=0
    // CHECK:   %[[cp2:.+]] = {{.+}} collective-permute({{.*}}%[[param2]]), {{.+}}
    // CHECK:   ROOT {{.+}} = {{.+}} tuple({{.*}}%[[cp2]], {{.*}})
    // CHECK: ENTRY %main
    // CHECK-NOT: collective-permute
    // CHECK: }
  )"));
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       WhileLoopWithCollectivePermuteBackwardCyclePeeled) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(15)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}
body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  collective-permute = f32[] collective-permute(param_0), channel_id=1, source_target_pairs={{0,7},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6}}
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(collective-permute, cond_plus_1)
}
ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"15"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer);
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: %body
    // CHECK:   %[[cp1:.+]] = {{.+}} collective-permute({{.+}}), {{.+}}
    // CHECK:   %[[out1:.+]] = {{.+}} tuple({{.*}}%[[cp1]], {{.*}})
    // CHECK:   %[[param2:.+]] = {{.+}} get-tuple-element({{.*}}%[[out1]]), index=0
    // CHECK:   %[[cp2:.+]] = {{.+}} collective-permute({{.*}}%[[param2]]), {{.+}}
    // CHECK:   ROOT {{.+}} = {{.+}} tuple({{.*}}%[[cp2]], {{.*}})
    // CHECK: }
    // CHECK: ENTRY %main
    // CHECK:   %[[cp_peeled:.+]] = {{.+}} collective-permute({{.+}}), {{.+}}
    // CHECK:   %[[out_peeled:.+]] = {{.+}} tuple({{.*}}%[[cp_peeled]], {{.*}})
    // CHECK:   ROOT {{.+}} = {{.+}} while({{.*}}%[[out_peeled]])
    // CHECK: }
  )"));
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       WhileLoopWithCollectivePermuteStartDone) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}
body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  collective-permute-start = (f32[], f32[], u32[], u32[]) collective-permute-start(param_0), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
  collective-permute = f32[] collective-permute-done(collective-permute-start)
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(collective-permute, cond_plus_1)
}
ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer);
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: %body
    // CHECK:   %[[cp_start1:.+]] = {{.+}} collective-permute-start({{.+}}), {{.+}}
    // CHECK:   %[[cp1:.+]] = {{.+}} collective-permute-done({{.*}}%[[cp_start1]])
    // CHECK:   %[[out1:.+]] = {{.+}} tuple({{.*}}%[[cp1]], {{.*}})
    // CHECK:   %[[param2:.+]] = {{.+}} get-tuple-element({{.*}}%[[out1]]), index=0
    // CHECK:   %[[cp_start2:.+]] = {{.+}} collective-permute-start({{.*}}), {{.+}}
    // CHECK:   %[[cp2:.+]] = {{.+}} collective-permute-done({{.*}}%[[cp_start2]])
    // CHECK:   ROOT {{.+}} = {{.+}} tuple({{.*}}%[[cp2]], {{.*}})
    // CHECK: }
    // CHECK: ENTRY %main
    // CHECK-NOT: collective-permute
    // CHECK: }
  )"));
}

TEST_F(GpuLoopDoubleBufferTransformerTest, WhileLoopWithRecvDone) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}
body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  after-all.0 = token[] after-all()
  recv.0 = (f32[], u32[], token[]) recv(after-all.0), channel_id=1,
        frontend_attributes={
          _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3},{3,0}}",
          _xla_send_recv_pipeline="0"
        }
  recv-done.0 = (f32[], token[]) recv-done(recv.0), channel_id=1,
        frontend_attributes={
          _xla_send_recv_pipeline="0"
        }
  recv-data = f32[] get-tuple-element(recv-done.0), index=0
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(recv-data, cond_plus_1)
}
ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer);
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: %body
    // CHECK:   %[[recv1:.+]] = {{.+}} recv({{.+}}), {{.+}}
    // CHECK:   %[[recv2:.+]] = {{.+}} recv({{.+}}), {{.+}}
    // CHECK: ENTRY %main
    // CHECK-NOT: recv
    // CHECK: }
  )"));
}

TEST_F(GpuLoopDoubleBufferTransformerTest, WhileLoopWithSendDone) {
  const char* kModuleString = R"(
HloModule loop_unrolling_no_deps
condition {
  input_tuple = (f32[], s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=1
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
ar_add {
  Arg_1 = f32[] parameter(1)
  Arg_0 = f32[] parameter(0)
  ROOT add_ar = f32[] add(Arg_1, Arg_0)
}
body {
  input_tuple = (f32[], s32[]) parameter(0)
  param_0 = f32[] get-tuple-element(input_tuple), index=0
  cond = s32[] get-tuple-element(input_tuple), index=1
  after-all.0 = token[] after-all()
  send.0 = (f32[], u32[], token[]) send(param_0, after-all.0), channel_id=1,
        frontend_attributes={
          _xla_send_recv_source_target_pairs="{{0,1},{1,2},{2,3},{3,0}}",
          _xla_send_recv_pipeline="0"
        }
  send-done.0 = token[] send-done(send.0), channel_id=1,
        frontend_attributes={
          _xla_send_recv_pipeline="0"
        }
  one = s32[] constant(1)
  cond_plus_1 = s32[] add(cond, one)
  ROOT output_tuple = (f32[], s32[]) tuple(param_0, cond_plus_1)
}
ENTRY main {
  param_0 = f32[] parameter(0)
  param_2 = s32[] constant(0)
  tuple = (f32[], s32[]) tuple(param_0, param_2)
  ROOT while = (f32[], s32[]) while(tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"}}
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer);
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(true));

  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    // CHECK: %body
    // CHECK:   %[[send1:.+]] = {{.+}} send({{.+}}), {{.+}}
    // CHECK:   %[[send2:.+]] = {{.+}} send({{.+}}), {{.+}}
    // CHECK: ENTRY %main
    // CHECK-NOT: send
    // CHECK: }
  )"));
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       WhileLoopWithTripCount1ShouldBeSkipped) {
  const char* const kModuleString = R"(
HloModule loop_unrolling_skipped
condition_nested {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(0)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
body_nested {
 input_tuple = (s32[]) parameter(0)
 cond = s32[] get-tuple-element(input_tuple), index=0
 one = s32[] constant(1)
 cond_plus_1 = s32[] add(cond, one)
 ROOT output = (s32[]) tuple(cond_plus_1)
}
condition {
  input_tuple = (s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(0)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}
body {
  input_tuple = (s32[]) parameter(0)
  ROOT output = (s32[]) while(input_tuple), condition=condition_nested, body=body_nested, backend_config={"known_trip_count":{"n":"1"}}
}
ENTRY main {
 param_0 = (s32[]) parameter(0)
 ROOT while = (s32[]) while(param_0), condition=condition, body=body, backend_config={"known_trip_count":{"n":"1"}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kFullUnroll);
  // The processing of the loop should be completely skipped.
  EXPECT_THAT(double_buffer.Run(module.get()),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(GpuLoopDoubleBufferTransformerTest, UpdateInitStepOddTripCount) {
  absl::string_view kModuleString = R"(
    HloModule m
      condition {
      input_tuple = (s32[]) parameter(0)
      iter = s32[] get-tuple-element(input_tuple), index=0
      c12 = s32[] constant(12)
      ROOT continue = pred[] compare(iter, c12), direction=LT
    }

    body {
      input_tuple = (s32[]) parameter(0)
      iter = s32[] get-tuple-element(input_tuple), index=0
      c2 = s32[] constant(2)
      next_iter = s32[] add(iter, c2)
      ROOT output_tuple = (s32[]) tuple(next_iter)
    }

    ENTRY main {
      c3 = s32[] constant(3)
      tuple = (s32[]) tuple(c3)
      // Values: 3, 5, 7, 9, 11
      ROOT while = (s32[]) while(tuple), condition=condition, body=body,
        backend_config={"known_trip_count":{"n":"5"},
                        "known_init_step":{"init":"3","step":"2"}}
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling unroller(
      DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, unroller.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* while_instruction = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_trip_count().n(), 2);
  EXPECT_EQ(config.known_init_step().init(), 5);
  EXPECT_EQ(config.known_init_step().step(), 4);
}

TEST_F(GpuLoopDoubleBufferTransformerTest, UpdateInitStepEvenTripCount) {
  absl::string_view kModuleString = R"(
    HloModule m
      condition {
      input_tuple = (s32[]) parameter(0)
      iter = s32[] get-tuple-element(input_tuple), index=0
      c14 = s32[] constant(14)
      ROOT continue = pred[] compare(iter, c14), direction=LT
    }

    body {
      input_tuple = (s32[]) parameter(0)
      iter = s32[] get-tuple-element(input_tuple), index=0
      c2 = s32[] constant(2)
      next_iter = s32[] add(iter, c2)
      ROOT output_tuple = (s32[]) tuple(next_iter)
    }

    ENTRY main {
      c3 = s32[] constant(3)
      tuple = (s32[]) tuple(c3)
      // Values: 3, 5, 7, 9, 11, 13
      ROOT while = (s32[]) while(tuple), condition=condition, body=body,
        backend_config={"known_trip_count":{"n":"6"},
                        "known_init_step":{"init":"3","step":"2"}}
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));
  DoubleBufferLoopUnrolling unroller(
      DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, unroller.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* while_instruction = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  TF_ASSERT_OK_AND_ASSIGN(
      WhileLoopBackendConfig config,
      while_instruction->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_trip_count().n(), 3);
  EXPECT_EQ(config.known_init_step().init(), 3);
  EXPECT_EQ(config.known_init_step().step(), 4);
}

TEST_F(GpuLoopDoubleBufferTransformerTest,
       PreserveDynamicVariableIndicesAfterDoubleBuffering) {
  absl::string_view kModuleString = R"(
HloModule test

condition {
  input_tuple = (s32[], f32[2,8]{1,0:S(5)}, f32[1,8]{1,0}, s32[]) parameter(0)
  cond = s32[] get-tuple-element(input_tuple), index=0
  trip_count = s32[] constant(10)
  ROOT done = pred[] compare(cond, trip_count), direction=LT
}

body {
  input_tuple = (s32[], f32[2,8]{1,0:S(5)}, f32[1,8]{1,0}, s32[]) parameter(0)
  idx = s32[] get-tuple-element(input_tuple), index=0
  buffer = f32[2,8]{1,0:S(5)} get-tuple-element(input_tuple), index=1
  update = f32[1,8]{1,0} get-tuple-element(input_tuple), index=2
  counter = s32[] get-tuple-element(input_tuple), index=3

  c0 = s32[] constant(0)
  dus = f32[2,8]{1,0:S(5)} dynamic-update-slice(buffer, update, idx, c0)

  c1 = s32[] constant(1)
  idx_plus_1 = s32[] add(idx, c1)
  counter_plus_1 = s32[] add(counter, c1)
  ROOT output = (s32[], f32[2,8]{1,0:S(5)}, f32[1,8]{1,0}, s32[]) tuple(idx_plus_1, dus, update, counter_plus_1)
}

ENTRY main {
  c0 = s32[] constant(0)
  buffer_init = f32[2,8]{1,0:S(5)} parameter(0)
  update_init = f32[1,8]{1,0} parameter(1)
  input_tuple = (s32[], f32[2,8]{1,0:S(5)}, f32[1,8]{1,0}, s32[]) tuple(c0, buffer_init, update_init, c0)
  ROOT while = (s32[], f32[2,8]{1,0:S(5)}, f32[1,8]{1,0}, s32[]) while(input_tuple), condition=condition, body=body, backend_config={"known_trip_count":{"n":"10"},"known_induction_variable":{"tuple_index":"0"},"dynamic_variable_tuple_indices":["3","0"]}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleString));

  DoubleBufferLoopUnrolling double_buffer(
      DoubleBufferLoopUnrolling::UnrollStrategy::kDoubleBuffer);
  TupleSimplifier tuple_simplifier;

  TF_ASSERT_OK_AND_ASSIGN(bool changed, double_buffer.Run(module.get()));
  ASSERT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(changed, tuple_simplifier.Run(module.get()));

  std::vector<HloInstruction*> while_loops;
  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kWhile) {
        while_loops.push_back(instr);
      }
    }
  }

  ASSERT_FALSE(while_loops.empty())
      << "Expected at least one while loop after double buffering";

  for (HloInstruction* while_loop : while_loops) {
    TF_ASSERT_OK_AND_ASSIGN(
        WhileLoopBackendConfig config,
        while_loop->backend_config<WhileLoopBackendConfig>());

    std::set<int64_t> dynamic_indices(
        config.dynamic_variable_tuple_indices().begin(),
        config.dynamic_variable_tuple_indices().end());

    EXPECT_FALSE(dynamic_indices.empty())
        << "Expected dynamic_variable_tuple_indices to be preserved for while "
           "loop: "
        << while_loop->name()
        << ". Double buffering should not erase indices set by "
           "CollectivePipeliner.";

    EXPECT_NE(dynamic_indices.find(0), dynamic_indices.end())
        << "Expected tuple index 0 (induction variable) to be preserved as "
           "dynamic for while loop: "
        << while_loop->name();

    EXPECT_NE(dynamic_indices.find(3), dynamic_indices.end())
        << "Expected tuple index 3 (additional counter) to be preserved as "
           "dynamic for while loop: "
        << while_loop->name();
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
