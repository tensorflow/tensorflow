/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/combine_instructions.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_fuser.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inter_ipu_copy_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/look_ahead_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/sync_list_scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include <algorithm>

namespace xla {
namespace poplarplugin {
namespace {

using CombineInstructionsTest = HloTestBase;

TEST_F(CombineInstructionsTest, TestSyncScheduler) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %arg1 = f16[4] parameter(1)
  %arg2 = f16[4] parameter(2)
  %a1 = f16[4] all-reduce(arg0), to_apply=add
  %a2 = f16[4] all-reduce(arg1), to_apply=add
  %a3 = f16[4] all-reduce(arg2), to_apply=add
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(f16[4] %a1, f16[4] %a2, f16[4] %a3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      IpuToMemorySchedulerAlgorithm(CreateSyncListMemoryScheduler(64 * 1024)));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  // Check the inplace instructions are all GTEs
  auto inplace_instructions = GetInplaceInstructions(module);
  EXPECT_EQ(inplace_instructions.size(), 3);
  for (auto inplace_inst : inplace_instructions) {
    EXPECT_EQ(inplace_inst->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_TRUE(inplace_inst->tuple_index() < 3);
  }

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 8);

  auto pred = [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kAllReduce;
  };
  ASSERT_EQ(absl::c_count_if(seq, pred), 1);
}

TEST_F(CombineInstructionsTest, TestLookAheadScheduler) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %arg1 = f16[4] parameter(1)
  %arg2 = f16[4] parameter(2)
  %a1 = f16[4] all-reduce(arg0), to_apply=add
  %a2 = f16[4] all-reduce(arg1), to_apply=add
  %a3 = f16[4] all-reduce(arg2), to_apply=add
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(f16[4] %a1, f16[4] %a2, f16[4] %a3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      IpuToMemorySchedulerAlgorithm(
          CreateLookAheadMemoryScheduler({64 * 1024, 64 * 1024})));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  // Check the inplace instructions are all GTEs
  auto inplace_instructions = GetInplaceInstructions(module);
  EXPECT_EQ(inplace_instructions.size(), 3);
  for (auto inplace_inst : inplace_instructions) {
    EXPECT_EQ(inplace_inst->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_TRUE(inplace_inst->tuple_index() < 3);
  }

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 8);

  auto pred = [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kAllReduce;
  };
  ASSERT_EQ(absl::c_count_if(seq, pred), 1);
}

TEST_F(CombineInstructionsTest, TestMergeInterIpuCopiesLookAheadScheduler) {
  std::string hlo_string = R"(
HloModule top

loop_body (arg_tuple.0: (s32[], f32[2], s32[])) -> (s32[], f32[2], s32[]) {
  after-all.1 = token[] after-all(), sharding={maximal device=0}
  infeed = ((f32[2]), token[]) infeed(after-all.1), infeed_config="\010\002\022\005feed0", sharding={{maximal device=0}, {maximal device=0}}
  get-tuple-element.5 = (f32[2]) get-tuple-element(infeed), index=0, sharding={{maximal device=0}}, backend_config="{\"isInplace\":true}"
  get-tuple-element.6 = f32[2] get-tuple-element(get-tuple-element.5), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  multiply = f32[2] multiply(get-tuple-element.6, get-tuple-element.6), sharding={maximal device=0}
  constant.7 = s32[] constant(2), sharding={maximal device=0}
  arg_tuple.0 = (s32[], f32[2], s32[]) parameter(0), sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}
  get-tuple-element.4 = f32[2] get-tuple-element(arg_tuple.0), index=1, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  add.1 = f32[2] add(get-tuple-element.4, get-tuple-element.6), sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  add.2 = f32[2] add(add.1, multiply), sharding={maximal device=1}, backend_config="{\"isInplace\":true}"
  get-tuple-element.3 = s32[] get-tuple-element(arg_tuple.0), index=0, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
  ROOT tuple.1 = (s32[], f32[2], s32[]) tuple(get-tuple-element.3, add.2, constant.7), sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}, backend_config="{\"isInplace\":true}"
}

_pop_op_wide_const () -> f32[2] {
  constant.1 = f32[] constant(0)
  ROOT broadcast.2 = f32[2] broadcast(constant.1), dimensions={}
}

ENTRY entry () -> f32[2] {
  fusion = f32[2] fusion(), kind=kCustom, calls=_pop_op_wide_const, sharding={maximal device=0}, backend_config="{}"
  constant.6 = s32[] constant(2), sharding={maximal device=0}
  tuple.7 = (s32[], f32[2], s32[]) tuple(constant.6, fusion, constant.6), sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}, backend_config="{\"isInplace\":true}"
  call = (s32[], f32[2], s32[]) call(tuple.7), to_apply=loop_body, sharding={{maximal device=0}, {maximal device=0}, {maximal device=0}}, backend_config="{\"repeatConfig\":{\"isRepeatLoop\":true,\"repeatCount\":\"2\"},\"isInplace\":true}"
  ROOT get-tuple-element.52 = f32[2] get-tuple-element(call), index=1, sharding={maximal device=0}, backend_config="{\"isInplace\":true}"
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  auto* comp = module->entry_computation();
  auto* repeat = comp->GetInstructionWithName("call");
  auto* body = repeat->to_apply();

  EXPECT_EQ(body->instruction_count(), 12);
  InterIpuCopyInserter inserterPass;
  EXPECT_TRUE(inserterPass.Run(module).ValueOrDie());

  auto pred = [](const HloInstruction* inst) { return IsInterIpuCopy(inst); };
  // Expect three inter IPU copies to have been inserted.
  EXPECT_EQ(body->instruction_count(), 15);
  ASSERT_EQ(absl::c_count_if(body->instructions(), pred), 3);

  // Schedule and combine.
  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      IpuToMemorySchedulerAlgorithm(
          CreateLookAheadMemoryScheduler({64 * 1024, 64 * 1024})));

  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());
  // Two IPU copies have been merged.
  EXPECT_EQ(absl::c_count_if(body->instructions(), pred), 2);
  EXPECT_EQ(body->instruction_count(), 16);
}

TEST_F(CombineInstructionsTest, TestLookAheadSchedulerGradientAccumulation) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %ga0 = f16[4] custom-call(arg0), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  %a0 = f16[4] all-reduce(ga0), to_apply=add
  %norm0 = f16[4] custom-call(a0), custom_call_target="Poputil::ReplicationNormalise", backend_config="{}\n"
  %arg1 = f16[4] parameter(1)
  %ga1 = f16[4] custom-call(arg1), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  %a1 = f16[4] all-reduce(ga1), to_apply=add
  %norm1 = f16[4] custom-call(a1), custom_call_target="Poputil::ReplicationNormalise", backend_config="{}\n"
  %arg2 = f16[4] parameter(2)
  %ga2 = f16[4] custom-call(arg2), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  %a2 = f16[4] all-reduce(ga2), to_apply=add
  %norm2 = f16[4] custom-call(a2), custom_call_target="Poputil::ReplicationNormalise", backend_config="{}\n"
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(norm0, norm1, norm2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  CompilerAnnotations annotations(module);
  auto* entry = module->entry_computation();

  // Replace and fuse the gradient accumulations.
  EXPECT_EQ(entry->instruction_count(), 13);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 13);
  GradientAccumulationFuser fuser(annotations);
  EXPECT_TRUE(fuser.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 10);

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      IpuToMemorySchedulerAlgorithm(
          CreateLookAheadMemoryScheduler({64 * 1024, 64 * 1024})));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  // Check the inplace instructions are all GTEs
  auto inplace_instructions = GetInplaceInstructions(module);
  EXPECT_EQ(inplace_instructions.size(), 3);
  for (auto inplace_inst : inplace_instructions) {
    EXPECT_EQ(inplace_inst->opcode(), HloOpcode::kGetTupleElement);
    EXPECT_TRUE(inplace_inst->tuple_index() < 3);
  }

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 11);
  auto pred = [](const HloInstruction* inst) {
    return IsInstructionType<HloStatefulGradientAccumulateAndAllReduce>(inst);
  };
  ASSERT_EQ(absl::c_count_if(seq, pred), 1);
}

TEST_F(CombineInstructionsTest,
       TestLookAheadSchedulerGradientAccumulationDifferentMiniBatches) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %ga0 = f16[4] custom-call(arg0), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  %a0 = f16[4] all-reduce(ga0), to_apply=add
  %norm0 = f16[4] custom-call(a0), custom_call_target="Poputil::ReplicationNormalise", backend_config="{}\n"
  %arg1 = f16[4] parameter(1)
  %ga1 = f16[4] custom-call(arg1), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":5}\n"
  %a1 = f16[4] all-reduce(ga1), to_apply=add
  %norm1 = f16[4] custom-call(a1), custom_call_target="Poputil::ReplicationNormalise", backend_config="{}\n"
  %arg2 = f16[4] parameter(2)
  %ga2 = f16[4] custom-call(arg2), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":6}\n"
  %a2 = f16[4] all-reduce(ga2), to_apply=add
  %norm2 = f16[4] custom-call(a2), custom_call_target="Poputil::ReplicationNormalise", backend_config="{}\n"
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(norm0, norm1, norm2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  CompilerAnnotations annotations(module);
  auto* entry = module->entry_computation();

  // Replace and fuse the gradient accumulations.
  EXPECT_EQ(entry->instruction_count(), 13);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 13);
  GradientAccumulationFuser fuser(annotations);
  EXPECT_TRUE(fuser.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 10);

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      IpuToMemorySchedulerAlgorithm(
          CreateLookAheadMemoryScheduler({64 * 1024, 64 * 1024})));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_FALSE(combine_instructions.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 10);
}

TEST_F(CombineInstructionsTest, TestInplace) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %ga0 = f16[4] custom-call(arg0), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  %a0 = f16[4] all-reduce(ga0), to_apply=add
  %norm0 = f16[4] custom-call(a0), custom_call_target="Poputil::ReplicationNormalise", backend_config="{}\n"
  %arg1 = f16[4] parameter(1)
  %ga1 = f16[4] custom-call(arg1), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  %a1 = f16[4] all-reduce(ga1), to_apply=add
  %norm1 = f16[4] custom-call(a1), custom_call_target="Poputil::ReplicationNormalise", backend_config="{}\n"
  %arg2 = f16[4] parameter(2)
  %ga2 = f16[4] custom-call(arg2), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  %a2 = f16[4] all-reduce(ga2), to_apply=add
  %norm2 = f16[4] custom-call(a2), custom_call_target="Poputil::ReplicationNormalise", backend_config="{}\n"
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(norm0, norm1, norm2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();
  CompilerAnnotations annotations(module);
  auto* entry = module->entry_computation();

  // Replace and fuse the gradient accumulations.
  EXPECT_EQ(entry->instruction_count(), 13);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 13);
  GradientAccumulationFuser fuser(annotations);
  EXPECT_TRUE(fuser.Run(module).ValueOrDie());
  EXPECT_EQ(entry->instruction_count(), 10);

  // Run the inplacer.
  InplaceFinder inplace_finder;
  EXPECT_TRUE(inplace_finder.Run(module).ValueOrDie());

  auto pred = [](const HloInstruction* inst) {
    return IsInstructionType<HloStatefulGradientAccumulateAndAllReduce>(inst);
  };

  // Expect the gradient accumulations to be inplace.
  auto inplace_instructions = GetInplaceInstructions(module);
  ASSERT_EQ(absl::c_count_if(inplace_instructions, pred), 3);

  // Make one of the gradient accumulations not inplace.
  auto root = entry->root_instruction();
  auto norm1 = root->mutable_operand(1);
  auto ga_and_ar = norm1->mutable_operand(0);
  MakeUsedNotInplace(ga_and_ar);

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      IpuToMemorySchedulerAlgorithm(
          CreateLookAheadMemoryScheduler({64 * 1024, 64 * 1024})));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());
  CombineInstructions combine_instructions;
  EXPECT_TRUE(combine_instructions.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 11);
  // Expect two gradient accumulation instructions.
  ASSERT_EQ(absl::c_count_if(seq, pred), 2);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
