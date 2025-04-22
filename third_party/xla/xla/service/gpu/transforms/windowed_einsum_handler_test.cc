/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/transforms/windowed_einsum_handler.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/pattern_matcher.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

using WindowedEinsumHandlerTest = HloHardwareIndependentTestBase;

HloInstruction* FindInstructionByName(HloComputation* comp, std::string name) {
  for (auto inst : comp->instructions()) {
    if (inst->name() == name) {
      return inst;
    }
  }
  return nullptr;
}

TEST_F(WindowedEinsumHandlerTest, AgLoopsHaveStreamIds) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[1,512,24576]{2,1,0}, bf16[24576,24576]{1,0})->bf16[2048,24576]{1,0}}, num_partitions=4

windowed_dot_general_body_ag.1 {
  param = (bf16[512,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[2048,24576]{1,0}, bf16[2048,24576]{1,0}, u32[]) parameter(0)
  get-tuple-element = bf16[512,24576]{1,0} get-tuple-element(param), index=0
  collective-permute.send_first_lhs_shard = bf16[512,24576]{1,0} collective-permute(get-tuple-element), channel_id=2, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
  get-tuple-element.lhs = bf16[24576,24576]{1,0} get-tuple-element(param), index=1
  get-tuple-element.rhs = bf16[2048,24576]{1,0} get-tuple-element(param), index=2
  dot.2 = bf16[512,24576]{1,0} dot(get-tuple-element, get-tuple-element.lhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
  constant.1 = s32[4]{0} constant({0, 512, 1024, 1536})
  get-tuple-element.4 = u32[] get-tuple-element(param), index=4
  partition-id = u32[] partition-id()
  add = u32[] add(get-tuple-element.4, partition-id)
  constant = u32[] constant(4)
  remainder = u32[] remainder(add, constant)
  dynamic-slice = s32[1]{0} dynamic-slice(constant.1, remainder), dynamic_slice_sizes={1}
  reshape.4 = s32[] reshape(dynamic-slice)
  constant.2 = s32[] constant(0)
  dynamic-update-slice = bf16[2048,24576]{1,0} dynamic-update-slice(get-tuple-element.rhs, dot.2, reshape.4, constant.2), backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
  dot.3 = bf16[512,24576]{1,0} dot(collective-permute.send_first_lhs_shard, get-tuple-element.lhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  constant.3 = u32[] constant(1)
  add.1 = u32[] add(get-tuple-element.4, constant.3)
  add.2 = u32[] add(add.1, partition-id)
  remainder.1 = u32[] remainder(add.2, constant)
  dynamic-slice.1 = s32[1]{0} dynamic-slice(constant.1, remainder.1), dynamic_slice_sizes={1}
  reshape.5 = s32[] reshape(dynamic-slice.1)
  dynamic-update-slice.1 = bf16[2048,24576]{1,0} dynamic-update-slice(dynamic-update-slice, dot.3, reshape.5, constant.2)
  get-tuple-element.3 = bf16[2048,24576]{1,0} get-tuple-element(param), index=3
  add.3 = u32[] add(add.1, constant.3)
  ROOT tuple = (bf16[512,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[2048,24576]{1,0}, bf16[2048,24576]{1,0}, u32[]) tuple(collective-permute.send_first_lhs_shard, get-tuple-element.lhs, dynamic-update-slice.1, get-tuple-element.3, add.3)
} // windowed_dot_general_body_ag.1

windowed_dot_general_cond_ag {
  param.1 = (bf16[512,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[2048,24576]{1,0}, bf16[2048,24576]{1,0}, u32[]) parameter(0)
  get-tuple-element.5 = u32[] get-tuple-element(param.1), index=4
  constant.8 = u32[] constant(4)
  ROOT compare = pred[] compare(get-tuple-element.5, constant.8), direction=LT
}

ENTRY test_main {
  param.4 = bf16[1,512,24576]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
  reshape.8 = bf16[512,24576]{1,0} reshape(param.4)
  param.5 = bf16[24576,24576]{1,0} parameter(1), sharding={devices=[1,4]<=[4]}
  constant.18 = bf16[] constant(0)
  broadcast = bf16[2048,24576]{1,0} broadcast(constant.18), dimensions={}
  constant.20 = u32[] constant(0)
  tuple.2 = (bf16[512,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[2048,24576]{1,0}, bf16[2048,24576]{1,0}, u32[]) tuple(reshape.8, param.5, broadcast, broadcast, constant.20)
  while = (bf16[512,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[2048,24576]{1,0}, bf16[2048,24576]{1,0}, u32[]) while(tuple.2), condition=windowed_dot_general_cond_ag, body=windowed_dot_general_body_ag.1
  ROOT get-tuple-element.13 = bf16[2048,24576]{1,0} get-tuple-element(while), index=2
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  WindowedEinsumHandler gpu_handler;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* ag_loop =
      module->entry_computation()->root_instruction()->mutable_operand(0);
  HloComputation* ag_loop_body = ag_loop->while_body();
  int64_t dot_count = 0;
  for (HloInstruction* inst : ag_loop_body->MakeInstructionPostOrder()) {
    if (HloPredicateIsOp<HloOpcode::kDot>(inst)) {
      dot_count++;
      EXPECT_GT(inst->backend_config<GpuBackendConfig>()->operation_queue_id(),
                0);
    }
  }
  EXPECT_EQ(dot_count, 4);

  HloInstruction* cp1 = FindInstructionByName(
      ag_loop_body, "collective-permute.send_first_lhs_shard.3");
  EXPECT_TRUE(
      cp1->backend_config<GpuBackendConfig>()->force_earliest_schedule());
}

TEST_F(WindowedEinsumHandlerTest, RsLoopsHaveStreamIds) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[24576,24576]{1,0}, bf16[512,24576]{1,0}, bf16[2048,24576]{1,0})->bf16[512,24576]{1,0}}, num_partitions=4

windowed_dot_general_body_rs_clone.1 {
  param.2 = (bf16[2048,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[512,24576]{1,0}, bf16[512,24576]{1,0}, u32[]) parameter(0)
  get-tuple-element.6 = bf16[2048,24576]{1,0} get-tuple-element(param.2), index=0
  get-tuple-element.7 = bf16[24576,24576]{1,0} get-tuple-element(param.2), index=1
  get-tuple-element.9 = bf16[512,24576]{1,0} get-tuple-element(param.2), index=2
  collective-permute.send_second_lhs_shard = bf16[512,24576]{1,0} collective-permute(get-tuple-element.9), channel_id=4, source_target_pairs={{0,2},{1,3},{2,0},{3,1}}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
  constant.10 = s32[4]{0} constant({0, 512, 1024, 1536})
  get-tuple-element.11 = u32[] get-tuple-element(param.2), index=4
  constant.12 = u32[] constant(2)
  add.8 = u32[] add(get-tuple-element.11, constant.12)
  constant.13 = u32[] constant(1)
  add.9 = u32[] add(add.8, constant.13)
  partition-id.3 = u32[] partition-id()
  add.10 = u32[] add(add.9, partition-id.3)
  constant.9 = u32[] constant(4)
  remainder.3 = u32[] remainder(add.10, constant.9)
  dynamic-slice.4 = s32[1]{0} dynamic-slice(constant.10, remainder.3), dynamic_slice_sizes={1}
  reshape.7 = s32[] reshape(dynamic-slice.4)
  constant.11 = s32[] constant(0)
  dynamic-slice.5 = bf16[512,24576]{1,0} dynamic-slice(get-tuple-element.6, reshape.7, constant.11), dynamic_slice_sizes={512,24576}
  dot.7 = bf16[512,24576]{1,0} dot(dynamic-slice.5, get-tuple-element.7), lhs_contracting_dims={1}, rhs_contracting_dims={0}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
  add.11 = bf16[512,24576]{1,0} add(collective-permute.send_second_lhs_shard, dot.7), backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
  get-tuple-element.10 = bf16[512,24576]{1,0} get-tuple-element(param.2), index=3
  add.6 = u32[] add(get-tuple-element.11, partition-id.3)
  remainder.2 = u32[] remainder(add.6, constant.9)
  dynamic-slice.2 = s32[1]{0} dynamic-slice(constant.10, remainder.2), dynamic_slice_sizes={1}
  reshape.6 = s32[] reshape(dynamic-slice.2)
  dynamic-slice.3 = bf16[512,24576]{1,0} dynamic-slice(get-tuple-element.6, reshape.6, constant.11), dynamic_slice_sizes={512,24576}
  dot.5 = bf16[512,24576]{1,0} dot(dynamic-slice.3, get-tuple-element.7), lhs_contracting_dims={1}, rhs_contracting_dims={0}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
  add.7 = bf16[512,24576]{1,0} add(get-tuple-element.10, dot.5), backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
  collective-permute.2 = bf16[512,24576]{1,0} collective-permute(add.7), channel_id=5, source_target_pairs={{0,2},{1,3},{2,0},{3,1}}
  ROOT tuple.1 = (bf16[2048,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[512,24576]{1,0}, bf16[512,24576]{1,0}, u32[]) tuple(get-tuple-element.6, get-tuple-element.7, add.11, collective-permute.2, add.8)
}

windowed_dot_general_cond_rs {
  param.3 = (bf16[2048,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[512,24576]{1,0}, bf16[512,24576]{1,0}, u32[]) parameter(0)
  get-tuple-element.12 = u32[] get-tuple-element(param.3), index=4
  constant.17 = u32[] constant(4)
  ROOT compare.1 = pred[] compare(get-tuple-element.12, constant.17), direction=LT
}

ENTRY main.9_spmd {
  param.6 = bf16[24576,24576]{1,0} parameter(0), sharding={devices=[4,1]<=[4]}
  param.7 = bf16[512,24576]{1,0} parameter(1)
  param.8 = bf16[2048,24576]{1,0} parameter(2)
  constant.20 = u32[] constant(0)
  tuple.3 = (bf16[2048,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[512,24576]{1,0}, bf16[512,24576]{1,0}, u32[]) tuple(param.8, param.6, param.7, param.7, constant.20)
  while.1 = (bf16[2048,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[512,24576]{1,0}, bf16[512,24576]{1,0}, u32[]) while(tuple.3), condition=windowed_dot_general_cond_rs, body=windowed_dot_general_body_rs_clone.1
  ROOT get-tuple-element.14 = bf16[512,24576]{1,0} get-tuple-element(while.1), index=2
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  WindowedEinsumHandler gpu_handler;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* rs_loop =
      module->entry_computation()->root_instruction()->mutable_operand(0);
  HloComputation* rs_loop_body = rs_loop->while_body();
  int64_t dot_count = 0;
  for (HloInstruction* inst : rs_loop_body->MakeInstructionPostOrder()) {
    if (HloPredicateIsOp<HloOpcode::kDot>(inst)) {
      dot_count++;
      EXPECT_GT(inst->backend_config<GpuBackendConfig>()->operation_queue_id(),
                0);
    }
  }
  EXPECT_EQ(dot_count, 4);
}

TEST_F(WindowedEinsumHandlerTest, AgLoopsMultipleConsumersAreChained) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,512,24576]{2,1,0}, bf16[24576,24576]{1,0}, bf16[24576,24576]{1,0})->bf16[2,2048,24576]{2,1,0}}, num_partitions=4

windowed_dot_general_body_ag {
  param.1 = (bf16[2,512,24576]{2,1,0}, bf16[24576,24576]{1,0}, bf16[2,2048,24576]{2,1,0}, bf16[2,2048,24576]{2,1,0}, u32[]) parameter(0)
  get-tuple-element.lhs = bf16[2,512,24576]{2,1,0} get-tuple-element(param.1), index=0
  collective-permute.send_first_lhs_shard = bf16[2,512,24576]{2,1,0} collective-permute(get-tuple-element.lhs), channel_id=2, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  collective-permute.send_second_lhs_shard = bf16[2,512,24576]{2,1,0} collective-permute(collective-permute.send_first_lhs_shard), channel_id=3, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  get-tuple-element.rhs = bf16[24576,24576]{1,0} get-tuple-element(param.1), index=1
  get-tuple-element.3 = bf16[2,2048,24576]{2,1,0} get-tuple-element(param.1), index=2
  dot = bf16[2,512,24576]{2,1,0} dot(get-tuple-element.lhs, get-tuple-element.rhs), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  constant.2 = s32[] constant(0)
  constant.3 = s32[4]{0} constant({0, 512, 1024, 1536})
  get-tuple-element.5 = u32[] get-tuple-element(param.1), index=4
  partition-id = u32[] partition-id()
  add = u32[] add(get-tuple-element.5, partition-id)
  constant.1 = u32[] constant(4)
  remainder = u32[] remainder(add, constant.1)
  dynamic-slice = s32[1]{0} dynamic-slice(constant.3, remainder), dynamic_slice_sizes={1}
  reshape = s32[] reshape(dynamic-slice)
  dynamic-update-slice = bf16[2,2048,24576]{2,1,0} dynamic-update-slice(get-tuple-element.3, dot, constant.2, reshape, constant.2)
  dot.1 = bf16[2,512,24576]{2,1,0} dot(collective-permute.send_first_lhs_shard, get-tuple-element.rhs), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  constant.5 = u32[] constant(1)
  add.1 = u32[] add(get-tuple-element.5, constant.5)
  add.2 = u32[] add(add.1, partition-id)
  remainder.1 = u32[] remainder(add.2, constant.1)
  dynamic-slice.1 = s32[1]{0} dynamic-slice(constant.3, remainder.1), dynamic_slice_sizes={1}
  reshape.1 = s32[] reshape(dynamic-slice.1)
  dynamic-update-slice.1 = bf16[2,2048,24576]{2,1,0} dynamic-update-slice(dynamic-update-slice, dot.1, constant.2, reshape.1, constant.2)
  get-tuple-element.4 = bf16[2,2048,24576]{2,1,0} get-tuple-element(param.1), index=3
  add.3 = u32[] add(add.1, constant.5)
  ROOT tuple = (bf16[2,512,24576]{2,1,0}, bf16[24576,24576]{1,0}, bf16[2,2048,24576]{2,1,0}, bf16[2,2048,24576]{2,1,0}, u32[]) tuple(collective-permute.send_second_lhs_shard, get-tuple-element.rhs, dynamic-update-slice.1, get-tuple-element.4, add.3)
} // windowed_dot_general_body_ag

windowed_dot_general_cond_ag {
  param = (bf16[2,512,24576]{2,1,0}, bf16[24576,24576]{1,0}, bf16[2,2048,24576]{2,1,0}, bf16[2,2048,24576]{2,1,0}, u32[]) parameter(0)
  get-tuple-element = u32[] get-tuple-element(param), index=4
  constant = u32[] constant(4)
  ROOT compare = pred[] compare(get-tuple-element, constant), direction=LT
}

ENTRY main.12_spmd {
  param.4 = bf16[2,512,24576]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
  param.5 = bf16[24576,24576]{1,0} parameter(1), sharding={devices=[1,4]<=[4]}
  constant.22 = bf16[] constant(0)
  broadcast = bf16[2,2048,24576]{2,1,0} broadcast(constant.22), dimensions={}
  constant.24 = u32[] constant(0)
  tuple.2 = (bf16[2,512,24576]{2,1,0}, bf16[24576,24576]{1,0}, bf16[2,2048,24576]{2,1,0}, bf16[2,2048,24576]{2,1,0}, u32[]) tuple(param.4, param.5, broadcast, broadcast, constant.24)
  while = (bf16[2,512,24576]{2,1,0}, bf16[24576,24576]{1,0}, bf16[2,2048,24576]{2,1,0}, bf16[2,2048,24576]{2,1,0}, u32[]) while(tuple.2), condition=windowed_dot_general_cond_ag, body=windowed_dot_general_body_ag
  get-tuple-element.13 = bf16[2,2048,24576]{2,1,0} get-tuple-element(while), index=2
  copy.1 = bf16[2,2048,24576]{2,1,0} copy(get-tuple-element.13)
  all-gather = bf16[2,2048,24576]{2,1,0} all-gather(param.4), channel_id=1, replica_groups={{0,1,2,3}}, dimensions={1}, use_global_device_ids=true
  param.6 = bf16[24576,24576]{1,0} parameter(2), sharding={devices=[1,4]<=[4]}
  ROOT dot.7 = bf16[2,2048,24576]{2,1,0} dot(all-gather, param.6), lhs_contracting_dims={2}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  WindowedEinsumHandler gpu_handler;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* inst =
      FindInstructionByName(module->entry_computation(), "dot.7");
  // dot.7 should now consume output of the windowed einsum while loop.
  EXPECT_EQ(inst->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(inst->operand(0)->tuple_index(), 5);
  const HloInstruction* while_loop = inst->operand(0)->operand(0);
  EXPECT_EQ(while_loop->opcode(), HloOpcode::kWhile);
  HloComputation* while_body = while_loop->while_body();
  int64_t dot_count = 0;
  for (HloInstruction* ins : while_body->MakeInstructionPostOrder()) {
    if (HloPredicateIsOp<HloOpcode::kDot>(ins)) {
      dot_count++;
      EXPECT_GT(ins->backend_config<GpuBackendConfig>()->operation_queue_id(),
                0);
    }
  }
  EXPECT_EQ(dot_count, 4);

  HloInstruction* ag_loop =
      FindInstructionByName(module->entry_computation(), "while");
  // while loop's root should now have a chain of 4 DUSes.
  HloInstruction* ag_while_root = ag_loop->while_body()->root_instruction();
  EXPECT_THAT(
      ag_while_root,
      GmockMatch(m::Tuple(
          m::Op(), m::Op(), m::Op(), m::Op(), m::Op(),
          m::DynamicUpdateSlice(
              m::DynamicUpdateSlice(
                  m::GetTupleElement(
                      m::Tuple(m::Op(), m::Op(), m::Op(), m::Op(), m::Op(),
                               m::DynamicUpdateSlice(
                                   m::DynamicUpdateSlice(
                                       m::GetTupleElement(m::Parameter())
                                           .WithPredicate(
                                               [](const HloInstruction* instr) {
                                                 return instr->tuple_index() ==
                                                        5;
                                               }),
                                       m::Op(), m::Op(), m::Op(), m::Op()),
                                   m::Op(), m::Op(), m::Op(), m::Op())))
                      .WithPredicate([](const HloInstruction* instr) {
                        return instr->tuple_index() == 5;
                      }),
                  m::Op(), m::Op(), m::Op(), m::Op()),
              m::Op(), m::Op(), m::Op(), m::Op()))));
  EXPECT_EQ(FindInstructionByName(module->entry_computation(), "all-gather"),
            nullptr);
}

TEST_F(WindowedEinsumHandlerTest, A2aGemmHaveStreamIds) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[1,8192,32768]{2,1,0}, bf16[1,4,2048,8192]{3,2,1,0})->bf16[1,4,2048,32768]{3,2,1,0}}, num_partitions=8

ENTRY main.9_spmd {
  param0 = bf16[1,8192,32768]{2,1,0} parameter(0)
  param1 = bf16[1,4,2048,8192]{3,2,1,0} parameter(1)
  all-to-all = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(param1), channel_id=4, replica_groups={{0,1,2,3},{4,5,6,7}}, dimensions={1}
  ROOT dot.12 = bf16[1,4,2048,32768]{3,2,1,0} dot(all-to-all, param0), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}
}
)";

  const char* kExpected = R"(
CHECK: ENTRY
CHECK-DAG: %[[P1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} parameter(1)

CHECK-DAG: %[[SLICE0:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(%[[P1]]), slice={[0:1], [0:4], [0:2048], [6144:8192]}
CHECK: %[[A2A0:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(%[[SLICE0]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3},{4,5,6,7}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[P0:.*]] = bf16[1,8192,32768]{2,1,0} parameter(0)
CHECK-DAG: %[[SLICE4:.*]] = bf16[1,2048,32768]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [6144:8192], [0:32768]}
CHECK-DAG: %[[DOT0:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(%[[A2A0:.*]], %[[SLICE4:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"8","wait_on_operation_queues":[],"force_earliest_schedule":false}

CHECK-DAG: %[[SLICE1:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(%[[P1]]), slice={[0:1], [0:4], [0:2048], [4096:6144]}
CHECK: %[[A2A1:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(%[[SLICE1]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3},{4,5,6,7}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[SLICE5:.*]] = bf16[1,2048,32768]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [4096:6144], [0:32768]}
CHECK-DAG: %[[DOT1:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(%[[A2A1:.*]], %[[SLICE5:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"7","wait_on_operation_queues":[],"force_earliest_schedule":false}

CHECK-DAG: %[[SLICE2:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(%[[P1]]), slice={[0:1], [0:4], [0:2048], [2048:4096]}
CHECK: %[[A2A2:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(%[[SLICE2]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3},{4,5,6,7}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[SLICE6:.*]] = bf16[1,2048,32768]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [2048:4096], [0:32768]}
CHECK-DAG: %[[DOT2:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(%[[A2A2:.*]], %[[SLICE6:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"6","wait_on_operation_queues":[],"force_earliest_schedule":false}

CHECK-DAG: %[[SLICE3:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(%[[P1]]), slice={[0:1], [0:4], [0:2048], [0:2048]}
CHECK: %[[A2A3:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(%[[SLICE3]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3},{4,5,6,7}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[SLICE7:.*]] = bf16[1,2048,32768]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [0:2048], [0:32768]}
CHECK-DAG: %[[DOT3:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(%[[A2A3:.*]], %[[SLICE7:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"5","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK-DAG: %[[CONSTANT:.*]] = bf16[] constant(0)
CHECK-DAG: %[[BROADCAST:.*]] = bf16[1,4,2048,32768]{3,2,1,0} broadcast(%[[CONSTANT:.*]]), dimensions={}
CHECK-DAG: %[[ADD0:.*]] = bf16[1,4,2048,32768]{3,2,1,0} add(%[[DOT0:.*]], %[[BROADCAST:.*]]), backend_config={"operation_queue_id":"0","wait_on_operation_queues":["5"],"force_earliest_schedule":false}
CHECK-DAG: %[[ADD1:.*]] = bf16[1,4,2048,32768]{3,2,1,0} add(%[[DOT1:.*]], %[[ADD0:.*]]), backend_config={"operation_queue_id":"0","wait_on_operation_queues":["6"],"force_earliest_schedule":false}
CHECK-DAG: %[[ADD2:.*]] = bf16[1,4,2048,32768]{3,2,1,0} add(%[[DOT2:.*]], %[[ADD1:.*]]), backend_config={"operation_queue_id":"0","wait_on_operation_queues":["7"],"force_earliest_schedule":false}

CHECK: ROOT {{.*}} = bf16[1,4,2048,32768]{3,2,1,0} add(%[[DOT3:.*]], %[[ADD2:.*]])
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  WindowedEinsumHandler gpu_handler;
  bool changed;
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_enable_alltoall_windowed_einsum(true);
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(module->ToString(), kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(WindowedEinsumHandlerTest, GemmA2aHaveStreamIds) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[1,8192,32768]{2,1,0}, bf16[1,4,2048,32768]{3,2,1,0})->bf16[1,4,2048,8192]{3,2,1,0}}, num_partitions=4

ENTRY main.9_spmd {
  param.9 = bf16[1,8192,32768]{2,1,0} parameter(0)
  param.10 = bf16[1,4,2048,32768]{3,2,1,0} parameter(1)
  dot.12 = bf16[1,4,2048,8192]{3,2,1,0} dot(param.10, param.9), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={2}
  ROOT all-to-all = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(dot.12), channel_id=4, replica_groups={{0,1,2,3}}, dimensions={1}
}
)";

  const char* kExpected = R"(
CHECK: ENTRY
CHECK-DAG: %[[P1:.*]] = bf16[1,4,2048,32768]{3,2,1,0} parameter(1)

CHECK-DAG: %[[SLICE0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(%[[P1]]), slice={[0:1], [0:4], [0:2048], [24576:32768]}
CHECK-DAG: %[[P0:.*]] = bf16[1,8192,32768]{2,1,0} parameter(0)
CHECK-DAG: %[[SLICE4:.*]] = bf16[1,8192,8192]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [0:8192], [24576:32768]}
CHECK-DAG: %[[DOT0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(%[[SLICE0:.*]], %[[SLICE4:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={2}, backend_config={"operation_queue_id":"8","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(%[[DOT0:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}

CHECK-DAG: %[[SLICE1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(%[[P1]]), slice={[0:1], [0:4], [0:2048], [16384:24576]}
CHECK-DAG: %[[SLICE5:.*]] = bf16[1,8192,8192]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [0:8192], [16384:24576]}
CHECK-DAG: %[[DOT1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(%[[SLICE1:.*]], %[[SLICE5:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={2}, backend_config={"operation_queue_id":"7","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(%[[DOT1:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}

CHECK-DAG: %[[SLICE2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(%[[P1]]), slice={[0:1], [0:4], [0:2048], [8192:16384]}
CHECK-DAG: %[[SLICE6:.*]] = bf16[1,8192,8192]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [0:8192], [8192:16384]}
CHECK-DAG: %[[DOT2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(%[[SLICE2:.*]], %[[SLICE6:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={2}, backend_config={"operation_queue_id":"6","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(%[[DOT2:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}

CHECK-DAG: %[[SLICE3:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(%[[P1]]), slice={[0:1], [0:4], [0:2048], [0:8192]}
CHECK-DAG: %[[SLICE7:.*]] = bf16[1,8192,8192]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [0:8192], [0:8192]}
CHECK-DAG: %[[DOT3:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(%[[SLICE3:.*]], %[[SLICE7:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={2}, backend_config={"operation_queue_id":"5","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A3:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(%[[DOT3:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[CONSTANT:.*]] = bf16[] constant(0)
CHECK-DAG: %[[BROADCAST:.*]] = bf16[1,4,2048,8192]{3,2,1,0} broadcast(%[[CONSTANT:.*]]), dimensions={}
CHECK-DAG: %[[ADD0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(%[[A2A0:.*]], %[[BROADCAST:.*]])
CHECK-DAG: %[[ADD1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(%[[A2A1:.*]], %[[ADD0:.*]])
CHECK-DAG: %[[ADD2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(%[[A2A2:.*]], %[[ADD1:.*]])

CHECK: ROOT {{.*}} = bf16[1,4,2048,8192]{3,2,1,0} add(%[[A2A3:.*]], %[[ADD2:.*]])
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  WindowedEinsumHandler gpu_handler;
  bool changed;
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_enable_alltoall_windowed_einsum(true);
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(module->ToString(), kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(WindowedEinsumHandlerTest, A2aTransposeLoopsHaveStreamIds) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[1,8192,32768]{2,1,0}, bf16[1,1,8192,4,1,2048]{5,4,3,2,1,0})->bf16[1,4,2048,32768]{3,2,1,0}}, num_partitions=4

ENTRY main.9_spmd {
  param.9 = bf16[1,8192,32768]{2,1,0} parameter(0)
  param.10 = bf16[1,1,8192,4,1,2048]{5,4,3,2,1,0} parameter(1)
  all-to-all = bf16[1,1,8192,4,1,2048]{5,4,3,2,1,0} all-to-all(param.10), channel_id=4, replica_groups={{0,1,2,3}}, dimensions={3}
  transpose.15 = bf16[1,4,1,8192,1,2048]{5,4,1,3,2,0} transpose(all-to-all), dimensions={0,3,1,2,4,5}
  reshape.2170 = bf16[1,4,8192,1,2048]{4,3,2,1,0} reshape(transpose.15)
  reshape.2173 = bf16[4,8192,1,2048]{3,2,1,0} reshape(reshape.2170)
  transpose.16 = bf16[1,4,2048,8192]{2,0,3,1} transpose(reshape.2173), dimensions={2,0,3,1}
  copy.53 = bf16[1,4,2048,8192]{3,2,1,0} copy(transpose.16)
  ROOT dot.12 = bf16[1,4,2048,32768]{3,2,1,0} dot(copy.53, param.9), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}
}
)";

  const char* kExpected = R"(
CHECK: ENTRY
CHECK-DAG: %[[P1:.*]] = bf16[1,1,8192,4,1,2048]{5,4,3,2,1,0} parameter(1)
CHECK-DAG: %[[TRANSPOSE0:.*]] = bf16[1,4,1,8192,1,2048]{5,4,1,3,2,0} transpose(%[[P1:.*]]), dimensions={0,3,1,2,4,5}
CHECK-DAG: %[[RESHAPE0:.*]] = bf16[1,4,8192,1,2048]{4,3,2,1,0} reshape(%[[TRANSPOSE0:.*]])
CHECK-DAG: %[[RESHAPE1:.*]] = bf16[4,8192,1,2048]{3,2,1,0} reshape(%[[RESHAPE0:.*]])
CHECK-DAG: %[[TRANSPOSE1:.*]] = bf16[1,4,2048,8192]{2,0,3,1} transpose(%[[RESHAPE1:.*]]), dimensions={2,0,3,1}
CHECK-DAG: %[[COPY:.*]] = bf16[1,4,2048,8192]{3,2,1,0} copy(%[[TRANSPOSE1:.*]])

CHECK-DAG: %[[SLICE0:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(%[[COPY:.*]]), slice={[0:1], [0:4], [0:2048], [6144:8192]}
CHECK: %[[A2A0:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(%[[SLICE0]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[P0:.*]] = bf16[1,8192,32768]{2,1,0} parameter(0)
CHECK-DAG: %[[SLICE4:.*]] = bf16[1,2048,32768]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [6144:8192], [0:32768]}
CHECK-DAG: %[[DOT0:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(%[[A2A0:.*]], %[[SLICE4:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"9","wait_on_operation_queues":[],"force_earliest_schedule":false}

CHECK-DAG: %[[SLICE1:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(%[[COPY:.*]]), slice={[0:1], [0:4], [0:2048], [4096:6144]}
CHECK: %[[A2A1:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(%[[SLICE1]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[SLICE5:.*]] = bf16[1,2048,32768]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [4096:6144], [0:32768]}
CHECK-DAG: %[[DOT1:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(%[[A2A1:.*]], %[[SLICE5:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"8","wait_on_operation_queues":[],"force_earliest_schedule":false}

CHECK-DAG: %[[SLICE2:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(%[[COPY:.*]]), slice={[0:1], [0:4], [0:2048], [2048:4096]}
CHECK: %[[A2A2:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(%[[SLICE2]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[SLICE6:.*]] = bf16[1,2048,32768]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [2048:4096], [0:32768]}
CHECK-DAG: %[[DOT2:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(%[[A2A2:.*]], %[[SLICE6:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"7","wait_on_operation_queues":[],"force_earliest_schedule":false}

CHECK-DAG: %[[SLICE3:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(%[[COPY:.*]]), slice={[0:1], [0:4], [0:2048], [0:2048]}
CHECK: %[[A2A3:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(%[[SLICE3]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[SLICE7:.*]] = bf16[1,2048,32768]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [0:2048], [0:32768]}
CHECK-DAG: %[[DOT3:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(%[[A2A3:.*]], %[[SLICE7:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"6","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK-DAG: %[[CONSTANT:.*]] = bf16[] constant(0)
CHECK-DAG: %[[BROADCAST:.*]] = bf16[1,4,2048,32768]{3,2,1,0} broadcast(%[[CONSTANT:.*]]), dimensions={}
CHECK-DAG: %[[ADD0:.*]] = bf16[1,4,2048,32768]{3,2,1,0} add(%[[DOT0:.*]], %[[BROADCAST:.*]]), backend_config={"operation_queue_id":"0","wait_on_operation_queues":["6"],"force_earliest_schedule":false}
CHECK-DAG: %[[ADD1:.*]] = bf16[1,4,2048,32768]{3,2,1,0} add(%[[DOT1:.*]], %[[ADD0:.*]]), backend_config={"operation_queue_id":"0","wait_on_operation_queues":["7"],"force_earliest_schedule":false}
CHECK-DAG: %[[ADD2:.*]] = bf16[1,4,2048,32768]{3,2,1,0} add(%[[DOT2:.*]], %[[ADD1:.*]]), backend_config={"operation_queue_id":"0","wait_on_operation_queues":["8"],"force_earliest_schedule":false}

CHECK: ROOT {{.*}} = bf16[1,4,2048,32768]{3,2,1,0} add(%[[DOT3:.*]], %[[ADD2:.*]])
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  WindowedEinsumHandler gpu_handler;
  bool changed;
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_enable_alltoall_windowed_einsum(true);
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(module->ToString(), kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(WindowedEinsumHandlerTest, GemmA2aTransposeLoopsHaveStreamIds) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[1,4,2048,32768]{3,2,1,0}, bf16[1,32768,8192]{2,1,0})->bf16[1,4,1,1,2048,8192]{5,4,3,2,1,0}}, num_partitions=4

ENTRY main.9_spmd {
  param.9 = bf16[1,4,2048,32768]{3,2,1,0} parameter(0)
  param.10 = bf16[1,32768,8192]{2,1,0} parameter(1)
  dot.13 = bf16[1,4,2048,8192]{3,2,1,0} dot(param.9, param.10), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}
  copy.55 = bf16[1,4,2048,8192]{3,2,1,0} copy(dot.13)
  transpose.17 = bf16[4,1,2048,8192]{3,2,0,1} transpose(copy.55), dimensions={1,0,2,3}
  copy.56 = bf16[4,1,2048,8192]{3,2,1,0} copy(transpose.17)
  reshape.2216 = bf16[1,4,1,2048,8192]{4,3,2,1,0} reshape(copy.56)
  reshape.2219 = bf16[1,4,1,1,2048,8192]{5,4,3,2,1,0} reshape(reshape.2216)
  ROOT all-to-all.1 = bf16[1,4,1,1,2048,8192]{5,4,3,2,1,0} all-to-all(reshape.2219), channel_id=7, replica_groups={{0,1,2,3}}, dimensions={1}
}
)";

  const char* kExpected = R"(
CHECK: ENTRY
CHECK-DAG: %[[P1:.*]] = bf16[1,4,2048,32768]{3,2,1,0} parameter(0)

CHECK-DAG: %[[SLICE0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(%[[P1]]), slice={[0:1], [0:4], [0:2048], [24576:32768]}
CHECK-DAG: %[[P0:.*]] = bf16[1,32768,8192]{2,1,0} parameter(1)
CHECK-DAG: %[[SLICE4:.*]] = bf16[1,8192,8192]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [24576:32768], [0:8192]}
CHECK-DAG: %[[DOT0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(%[[SLICE0:.*]], %[[SLICE4:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"12","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(%[[DOT0:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}

CHECK-DAG: %[[SLICE1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(%[[P1]]), slice={[0:1], [0:4], [0:2048], [16384:24576]}
CHECK-DAG: %[[SLICE5:.*]] = bf16[1,8192,8192]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [16384:24576], [0:8192]}
CHECK-DAG: %[[DOT1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(%[[SLICE1:.*]], %[[SLICE5:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"11","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(%[[DOT1:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}

CHECK-DAG: %[[SLICE2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(%[[P1]]), slice={[0:1], [0:4], [0:2048], [8192:16384]}
CHECK-DAG: %[[SLICE6:.*]] = bf16[1,8192,8192]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [8192:16384], [0:8192]}
CHECK-DAG: %[[DOT2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(%[[SLICE2:.*]], %[[SLICE6:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"10","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(%[[DOT2:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}

CHECK-DAG: %[[SLICE3:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(%[[P1]]), slice={[0:1], [0:4], [0:2048], [0:8192]}
CHECK-DAG: %[[SLICE7:.*]] = bf16[1,8192,8192]{2,1,0} slice(%[[P0:.*]]), slice={[0:1], [0:8192], [0:8192]}
CHECK-DAG: %[[DOT3:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(%[[SLICE3:.*]], %[[SLICE7:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"9","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A3:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(%[[DOT3:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[CONSTANT:.*]] = bf16[] constant(0)
CHECK-DAG: %[[BROADCAST:.*]] = bf16[1,4,2048,8192]{3,2,1,0} broadcast(%[[CONSTANT:.*]]), dimensions={}
CHECK-DAG: %[[ADD0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(%[[A2A0:.*]], %[[BROADCAST:.*]])
CHECK-DAG: %[[ADD1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(%[[A2A1:.*]], %[[ADD0:.*]])
CHECK-DAG: %[[ADD2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(%[[A2A2:.*]], %[[ADD1:.*]])
CHECK-DAG: %[[ADD3:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(%[[A2A3:.*]], %[[ADD2:.*]])

CHECK-DAG: %[[COPY:.*]] = bf16[1,4,2048,8192]{3,2,1,0} copy(%[[ADD3:.*]])
CHECK-DAG: %[[TRANSPOSE0:.*]] = bf16[4,1,2048,8192]{3,2,0,1} transpose(%[[COPY:.*]]), dimensions={1,0,2,3}
CHECK-DAG: %[[COPY1:.*]] = bf16[4,1,2048,8192]{3,2,1,0} copy(%[[TRANSPOSE0:.*]])
CHECK-DAG: %[[RESHAPE0:.*]] = bf16[1,4,1,2048,8192]{4,3,2,1,0} reshape(%[[COPY1:.*]])

CHECK: ROOT {{.*}} = bf16[1,4,1,1,2048,8192]{5,4,3,2,1,0} reshape(%[[RESHAPE0:.*]])
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  WindowedEinsumHandler gpu_handler;
  bool changed;
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_enable_alltoall_windowed_einsum(true);
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(module->ToString(), kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(WindowedEinsumHandlerTest, AllGatherF8) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(f8e4m3fn[2,512,24576]{2,1,0}, f8e4m3fn[1536,24576]{1,0}, f32[], f32[])->f32[2,2048,24576]{2,1,0}}, num_partitions=4

windowed_dot_general_body_ag {
  input = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) parameter(0)
  lhs = f32[2,512,24576]{2,1,0} get-tuple-element(input), index=0
  permuted_lhs0 = f32[2,512,24576]{2,1,0} collective-permute(lhs), channel_id=4, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  permuted_lhs1 = f32[2,512,24576]{2,1,0} collective-permute(permuted_lhs0), channel_id=5, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  rhs = f32[24576,24576]{1,0} get-tuple-element(input), index=1
  partial_dot_output = f32[2,2048,24576]{2,1,0} get-tuple-element(input), index=2
  dot0 = f32[2,512,24576]{2,1,0} dot(lhs, rhs), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  c0 = s32[] constant(0)
  dot_update_slice_offsets = s32[4]{0} constant({0, 512, 1024, 1536})
  loop_counter = u32[] get-tuple-element(input), index=4
  partition_id = u32[] partition-id()
  loop_counter_plus_partition_id = u32[] add(loop_counter, partition_id)
  c4 = u32[] constant(4)
  dot_update_slice_offsets_index0 = u32[] remainder(loop_counter_plus_partition_id, c4)
  dot_update_slice_offset0 = s32[1]{0} dynamic-slice(dot_update_slice_offsets, dot_update_slice_offsets_index0), dynamic_slice_sizes={1}
  dot_update_slice_offset_scalar0 = s32[] reshape(dot_update_slice_offset0)
  updated_dot_output0 = f32[2,2048,24576]{2,1,0} dynamic-update-slice(partial_dot_output, dot0, c0, dot_update_slice_offset_scalar0, c0)
  dot1 = f32[2,512,24576]{2,1,0} dot(permuted_lhs0, rhs), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  c1 = u32[] constant(1)
  loop_counter_plus_one = u32[] add(loop_counter, c1)
  loop_counter_plus_partiion_id_plus_one = u32[] add(loop_counter_plus_one, partition_id)
  dot_update_slice_offsets_index1 = u32[] remainder(loop_counter_plus_partiion_id_plus_one, c4)
  dot_update_slice_offset1 = s32[1]{0} dynamic-slice(dot_update_slice_offsets, dot_update_slice_offsets_index1), dynamic_slice_sizes={1}
  dot_update_slice_offset1_scalar = s32[] reshape(dot_update_slice_offset1)
  updated_dot_output1 = f32[2,2048,24576]{2,1,0} dynamic-update-slice(updated_dot_output0, dot1, c0, dot_update_slice_offset1_scalar, c0)
  pass_through = f32[2,2048,24576]{2,1,0} get-tuple-element(input), index=3
  next_loop_counter = u32[] add(loop_counter_plus_one, c1)
  ROOT tuple = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) tuple(permuted_lhs1, rhs, updated_dot_output1, pass_through, next_loop_counter)
} // windowed_dot_general_body_ag

windowed_dot_general_cond_ag {
  input = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) parameter(0)
  loop_counter = u32[] get-tuple-element(input), index=4
  loop_limit = u32[] constant(4)
  ROOT compare = pred[] compare(loop_counter, loop_limit), direction=LT
}

ENTRY main {
  lhs = f8e4m3fn[2,512,24576]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
  rhs = f8e4m3fn[1536,24576]{1,0} parameter(1), sharding={devices=[1,4]<=[4]}
  c0_f32 = f32[] constant(0)
  c0_f32_bcast = f32[2,2048,24576]{2,1,0} broadcast(c0_f32), dimensions={}
  c0_u32 = u32[] constant(0)
  scale_lhs = f32[] parameter(2)
  scale_lhs_bcast = f32[2,512,24576]{2,1,0} broadcast(scale_lhs), dimensions={}
  lhs_f32 = f32[2,512,24576]{2,1,0} convert(lhs)
  lhs_scaled = f32[2,512,24576]{2,1,0} multiply(lhs_f32, scale_lhs_bcast)
  scale_rhs = f32[] parameter(3)
  scale_rhs_bcast = f32[1536,24576]{1,0} broadcast(scale_rhs), dimensions={}
  rhs_f32 = f32[1536,24576]{1,0} convert(rhs)
  rhs_scaled = f32[1536,24576]{1,0} multiply(rhs_f32, scale_rhs_bcast)
  rhs_bcast = f32[16,1536,24576]{2,1,0} broadcast(rhs_scaled), dimensions={1,2}
  rhs_reshaped = f32[24576,24576]{1,0} reshape(rhs_bcast)
  while_input = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) tuple(lhs_scaled, rhs_reshaped, c0_f32_bcast, c0_f32_bcast, c0_u32)
  while = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) while(while_input), condition=windowed_dot_general_cond_ag, body=windowed_dot_general_body_ag
  ROOT get-tuple-element.13 = f32[2,2048,24576]{2,1,0} get-tuple-element(while), index=2
}
)";

  RunAndFilecheckHloRewrite(kHloString, WindowedEinsumHandler(),
                            R"(
; CHECK-LABEL: %unrolled_windowed_dot_general_body_ag
; CHECK-NEXT:    [[INPUT:%[^ ]+]] = (f8e4m3fn[2,512,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[], /*index=5*/f32[], f32[]) parameter(0)
; CHECK-NEXT:    [[LHS:%[^ ]+]] = f8e4m3fn[2,512,24576]{2,1,0} get-tuple-element([[INPUT]]), index=0
; CHECK-NEXT:    [[PERMUTED_LHS0:%[^ ]+]] = f8e4m3fn[2,512,24576]{2,1,0} collective-permute([[LHS]]), channel_id=6
; CHECK-NEXT:    [[PERMUTED_LHS1:%[^ ]+]] = f8e4m3fn[2,512,24576]{2,1,0} collective-permute([[PERMUTED_LHS0]]), channel_id=7
; CHECK-NEXT:    [[RHS:%[^ ]+]] = f8e4m3fn[24576,24576]{1,0} get-tuple-element([[INPUT]]), index=1
; CHECK-NEXT:    [[PARTIAL_DOT_OUTPUT:%[^ ]+]] = f32[2,2048,24576]{2,1,0} get-tuple-element([[INPUT]]), index=2
; CHECK-NEXT:    [[LHS_F32:%[^ ]+]] = f32[2,512,24576]{2,1,0} convert([[LHS]])
; CHECK-NEXT:    [[SCALE_LHS:%[^ ]+]] = f32[] get-tuple-element([[INPUT]]), index=5
; CHECK-NEXT:    [[SCALE_LHS_BCAST:%[^ ]+]] = f32[2,512,24576]{2,1,0} broadcast([[SCALE_LHS]]), dimensions={}
; CHECK-NEXT:    [[LHS_SCALED:%[^ ]+]] = f32[2,512,24576]{2,1,0} multiply([[LHS_F32]], [[SCALE_LHS_BCAST]])
; CHECK-NEXT:    [[RHS_F32:%[^ ]+]] = f32[24576,24576]{1,0} convert([[RHS]])
; CHECK-NEXT:    [[SCALE_RHS:%[^ ]+]] = f32[] get-tuple-element([[INPUT]]), index=6
; CHECK-NEXT:    [[SCALE_RHS_BCAST:%[^ ]+]] = f32[24576,24576]{1,0} broadcast([[SCALE_RHS]]), dimensions={}
; CHECK-NEXT:    [[RHS_SCALED:%[^ ]+]] = f32[24576,24576]{1,0} multiply([[RHS_F32]], [[SCALE_RHS_BCAST]])
; CHECK-NEXT:    [[DOT0:%[^ ]+]] = f32[2,512,24576]{2,1,0} dot([[LHS_SCALED]], [[RHS_SCALED]]),
; CHECK-DAG:       lhs_contracting_dims={2},
; CHECK-DAG:       rhs_contracting_dims={0},
; CHECK-DAG:       backend_config={
; CHECK-DAG:         "operation_queue_id":"[[OPQUEUEID:[0-9]+]]",
; CHECK-DAG:         "wait_on_operation_queues":[],
; CHECK-DAG:         "force_earliest_schedule":false}
; CHECK-NEXT:    [[C0_S32:%[^ ]+]] = s32[] constant(0)
; CHECK-NEXT:    [[C0_U32:%[^ ]+]] = u32[] constant(0)
; CHECK-NEXT:    [[C5:%[^ ]+]] = u32[] constant(0)
; CHECK-NEXT:    [[PARTITION_ID:%[^ ]+]] = u32[] partition-id()
; CHECK-NEXT:    [[ADD0:%[^ ]+]] = u32[] add([[C5]], [[PARTITION_ID]])
; CHECK-NEXT:    [[C3:%[^ ]+]] = u32[] constant(3)
; CHECK-NEXT:    [[AND0:%[^ ]+]] = u32[] and([[ADD0]], [[C3]])
; CHECK-NEXT:    [[CLAMP0:%[^ ]+]] = u32[] clamp([[C0_U32]], [[AND0]], [[C3]])
; CHECK-NEXT:    [[CONVERT3:%[^ ]+]] = s32[] convert([[CLAMP0]])
; CHECK-NEXT:    [[C512:%[^ ]+]] = s32[] constant(512)
; CHECK-NEXT:    [[MUL3:%[^ ]+]] = s32[] multiply([[CONVERT3]], [[C512]])
; CHECK-NEXT:    [[RESHAPE0:%[^ ]+]] = s32[] reshape([[MUL3]])
; CHECK-NEXT:    [[UPDATED_DOT_OUTPUT0:%[^ ]+]] = f32[2,2048,24576]{2,1,0} dynamic-update-slice([[PARTIAL_DOT_OUTPUT]], [[DOT0]], [[C0_S32]], [[RESHAPE0]], [[C0_S32]]),
; CHECK-DAG:       backend_config={
; CHECK-DAG:         "operation_queue_id":"0",
; CHECK-DAG:         "wait_on_operation_queues":["[[OPQUEUEID]]"],
; CHECK-DAG:         "force_earliest_schedule":false}
; CHECK-NEXT:    [[PERMUTED_LHS0_F32:%[^ ]+]] = f32[2,512,24576]{2,1,0} convert([[PERMUTED_LHS0]])
; CHECK-NEXT:    [[PERMUTED_LHS_SCALED:%[^ ]+]] = f32[2,512,24576]{2,1,0} multiply([[PERMUTED_LHS0_F32]], [[SCALE_LHS_BCAST]])
; CHECK-NEXT:    [[DOT1:%[^ ]+]] = f32[2,512,24576]{2,1,0} dot([[PERMUTED_LHS_SCALED]], [[RHS_SCALED]]),
; CHECK-DAG:       lhs_contracting_dims={2},
; CHECK-DAG:       rhs_contracting_dims={0}
; CHECK-NEXT:    [[LOOP_COUNTER:%[^ ]+]] = u32[] get-tuple-element([[INPUT]]), index=4
; CHECK-NEXT:    [[C1:%[^ ]+]] = u32[] constant(1)
; CHECK-NEXT:    [[LOOP_COUNTER_PLUS_ONE:%[^ ]+]] = u32[] add([[LOOP_COUNTER]], [[C1]])
; CHECK-NEXT:    [[LOOP_COUNTER_PLUS_ONE_PLUS_PARTITION_ID:%[^ ]+]] = u32[] add([[LOOP_COUNTER_PLUS_ONE]], [[PARTITION_ID]])
; CHECK-NEXT:    [[AND1:%[^ ]+]] = u32[] and([[LOOP_COUNTER_PLUS_ONE_PLUS_PARTITION_ID]], [[C3]])
; CHECK-NEXT:    [[CLAMP1:%[^ ]+]] = u32[] clamp([[C0_U32]], [[AND1]], [[C3]])
; CHECK-NEXT:    [[CONVERT4:%[^ ]+]] = s32[] convert([[CLAMP1]])
; CHECK-NEXT:    [[MUL4:%[^ ]+]] = s32[] multiply([[CONVERT4]], [[C512]])
; CHECK-NEXT:    [[RESHAPE1:%[^ ]+]] = s32[] reshape([[MUL4]])
; CHECK-NEXT:    [[UPDATED_DOT_OUTPUT1:%[^ ]+]] = f32[2,2048,24576]{2,1,0} dynamic-update-slice([[UPDATED_DOT_OUTPUT0]], [[DOT1]], [[C0_S32]], [[RESHAPE1]], [[C0_S32]])
; CHECK-NEXT:    [[PASS_THROUGH:%[^ ]+]] = f32[2,2048,24576]{2,1,0} get-tuple-element([[INPUT]]), index=3
; CHECK-NEXT:    [[C2:%[^ ]+]] = u32[] constant(2)
; CHECK-NEXT:    [[NEXT_LOOP_COUNTER:%[^ ]+]] = u32[] add([[LOOP_COUNTER]], [[C2]])
; CHECK-NEXT:    [[TUPLE:%[^ ]+]] = (f8e4m3fn[2,512,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[], /*index=5*/f32[], f32[]) tuple([[PERMUTED_LHS1]], [[RHS]], [[UPDATED_DOT_OUTPUT1]], [[PASS_THROUGH]], [[NEXT_LOOP_COUNTER]], /*index=5*/[[SCALE_LHS]], [[SCALE_RHS]])
; CHECK-LABEL: ENTRY %main
; CHECK:         [[LHS:%[^ ]+]] = f8e4m3fn[2,512,24576]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
; CHECK-NEXT:    [[RHS:%[^ ]+]] = f8e4m3fn[1536,24576]{1,0} parameter(1), sharding={devices=[1,4]<=[4]}
; CHECK-NEXT:    [[RHS_BCAST:%[^ ]+]] = f8e4m3fn[16,1536,24576]{2,1,0} broadcast([[RHS]]), dimensions={1,2}
; CHECK-NEXT:    [[RHS_RESHAPED:%[^ ]+]] = f8e4m3fn[24576,24576]{1,0} reshape([[RHS_BCAST]])
; CHECK-NEXT:    [[C0:%[^ ]+]] = f32[] constant(0)
; CHECK-NEXT:    [[C0_BCAST:%[^ ]+]] = f32[2,2048,24576]{2,1,0} broadcast([[C0]]), dimensions={}
; CHECK-NEXT:    [[C0_U32:%[^ ]+]] = u32[] constant(0)
; CHECK-NEXT:    [[SCALE_LHS:%[^ ]+]] = f32[] parameter(2)
; CHECK-NEXT:    [[SCALE_RHS:%[^ ]+]] = f32[] parameter(3)
; CHECK-NEXT:    [[WHILE_INPUT:%[^ ]+]] = (f8e4m3fn[2,512,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[], /*index=5*/f32[], f32[]) tuple([[LHS]], [[RHS_RESHAPED]], [[C0_BCAST]], [[C0_BCAST]], [[C0_U32]], /*index=5*/[[SCALE_LHS]], [[SCALE_RHS]])
; CHECK:         [[WHILE:%[^ ]+]] = (f8e4m3fn[2,512,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[], /*index=5*/f32[], f32[]) while([[WHILE_INPUT]]),
; CHECK-DAG:       condition=%unrolled_windowed_dot_general_cond_ag,
; CHECK-DAG:       body=%unrolled_windowed_dot_general_body_ag
)");
}

TEST_F(WindowedEinsumHandlerTest, ReduceScatterF8) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(f8e4m3fn[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f8e4m3fn[2,2048,24576]{2,1,0}, f32[], f32[])->f32[2,512,24576]{2,1,0}}, num_partitions=4

windowed_dot_general_body_rs {
  param.3 = (f32[2,2048,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f32[2,512,24576]{2,1,0}, u32[]) parameter(0)
  get-tuple-element.lhs = f32[2,2048,24576]{2,1,0} get-tuple-element(param.3), index=0
  get-tuple-element.rhs = f32[24576,24576]{1,0} get-tuple-element(param.3), index=1
  get-tuple-element.output = f32[2,512,24576]{2,1,0} get-tuple-element(param.3), index=2
  collective-permute.send_shard = f32[2,512,24576]{2,1,0} collective-permute(get-tuple-element.output), channel_id=9, source_target_pairs={{0,2},{1,3},{2,0},{3,1}}
  constant.zero = s32[] constant(0)
  constant.loop_index = s32[4]{0} constant({0, 512, 1024, 1536})
  get-tuple-element.loop_iter = u32[] get-tuple-element(param.3), index=4
  constant.iter_increment = u32[] constant(2)
  add.8 = u32[] add(get-tuple-element.loop_iter, constant.iter_increment)
  constant.27 = u32[] constant(1)
  add.9 = u32[] add(add.8, constant.27)
  partition-id.3 = u32[] partition-id()
  add.shard_index = u32[] add(add.9, partition-id.3)
  constant.22 = u32[] constant(4)
  remainder.shard_index = u32[] remainder(add.shard_index, constant.22)
  dynamic-slice.shard_start_index = s32[1]{0} dynamic-slice(constant.loop_index, remainder.shard_index), dynamic_slice_sizes={1}
  reshape.3 = s32[] reshape(dynamic-slice.shard_start_index)
  dynamic-slice.shard_to_compute = f32[2,512,24576]{2,1,0} dynamic-slice(get-tuple-element.lhs, constant.zero, reshape.3, constant.zero), dynamic_slice_sizes={2,512,24576}
  dot.first_shard_dot = f32[2,512,24576]{2,1,0} dot(dynamic-slice.shard_to_compute, get-tuple-element.rhs), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  add.shard_partial_result = f32[2,512,24576]{2,1,0} add(collective-permute.send_shard, dot.first_shard_dot)
  get-tuple-element.10 = f32[2,512,24576]{2,1,0} get-tuple-element(param.3), index=3
  add.6 = u32[] add(get-tuple-element.loop_iter, partition-id.3)
  remainder.2 = u32[] remainder(add.6, constant.22)
  dynamic-slice.2 = s32[1]{0} dynamic-slice(constant.loop_index, remainder.2), dynamic_slice_sizes={1}
  reshape.2 = s32[] reshape(dynamic-slice.2)
  dynamic-slice.3 = f32[2,512,24576]{2,1,0} dynamic-slice(get-tuple-element.lhs, constant.zero, reshape.2, constant.zero), dynamic_slice_sizes={2,512,24576}
  dot.second_shard_dot = f32[2,512,24576]{2,1,0} dot(dynamic-slice.3, get-tuple-element.rhs), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  add.7 = f32[2,512,24576]{2,1,0} add(get-tuple-element.10, dot.second_shard_dot)
  collective-permute.send_second_shard = f32[2,512,24576]{2,1,0} collective-permute(add.7), channel_id=10, source_target_pairs={{0,2},{1,3},{2,0},{3,1}}
  ROOT tuple.1 = (f32[2,2048,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f32[2,512,24576]{2,1,0}, u32[]) tuple(get-tuple-element.lhs, get-tuple-element.rhs, add.shard_partial_result, collective-permute.send_second_shard, add.8)
} // windowed_dot_general_body_rs

windowed_dot_general_cond_rs {
  param.2 = (f32[2,2048,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f32[2,512,24576]{2,1,0}, u32[]) parameter(0)
  get-tuple-element.6 = u32[] get-tuple-element(param.2), index=4
  constant.21 = u32[] constant(4)
  ROOT compare.1 = pred[] compare(get-tuple-element.6, constant.21), direction=LT
}

ENTRY main.9_spmd {
  param.6 = f8e4m3fn[24576,24576]{1,0} parameter(0), sharding={devices=[4,1]<=[4]}
  param.7 = f32[2,512,24576]{2,1,0} parameter(1)
  param.8 = f8e4m3fn[2,2048,24576]{2,1,0} parameter(2)
  constant.20 = u32[] constant(0)
  scale_lhs = f32[] parameter(3)
  scale_lhs_bcast = f32[2,2048,24576]{2,1,0} broadcast(scale_lhs), dimensions={}
  lhs_bf16 = f32[2,2048,24576]{2,1,0} convert(param.8)
  lhs_scaled = f32[2,2048,24576]{2,1,0} multiply(lhs_bf16, scale_lhs_bcast)
  scale_rhs = f32[] parameter(4)
  scale_rhs_bcast = f32[24576,24576]{1,0} broadcast(scale_rhs), dimensions={}
  rhs_bf16 = f32[24576,24576]{1,0} convert(param.6)
  rhs_scaled = f32[24576,24576]{1,0} multiply(rhs_bf16, scale_rhs_bcast)
  tuple.3 = (f32[2,2048,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f32[2,512,24576]{2,1,0}, u32[]) tuple(lhs_scaled, rhs_scaled, param.7, param.7, constant.20)
  while.1 = (f32[2,2048,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f32[2,512,24576]{2,1,0}, u32[]) while(tuple.3), condition=windowed_dot_general_cond_rs, body=windowed_dot_general_body_rs
  ROOT get-tuple-element.14 = f32[2,512,24576]{2,1,0} get-tuple-element(while.1), index=2
}
)";

  RunAndFilecheckHloRewrite(kHloString, WindowedEinsumHandler(),
                            R"(
; CHECK-LABEL: unrolled_windowed_dot_general_body_rs
; CHECK-NEXT:    [[P0:%[^ ]+]] = (f8e4m3fn[2,2048,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f32[2,512,24576]{2,1,0}, u32[], /*index=5*/f32[], f32[]) parameter(0)
; CHECK-NEXT:    [[GTE0:%[^ ]+]] = f8e4m3fn[2,2048,24576]{2,1,0} get-tuple-element([[P0]]), index=0
; CHECK-NEXT:    [[GTE1:%[^ ]+]] = f8e4m3fn[24576,24576]{1,0} get-tuple-element([[P0]]), index=1
; CHECK-NEXT:    [[GTE2:%[^ ]+]] = f32[2,512,24576]{2,1,0} get-tuple-element([[P0]]), index=2
; CHECK-NEXT:    [[CP0:%[^ ]+]] = f32[2,512,24576]{2,1,0} collective-permute([[GTE2]]), channel_id=11
; CHECK-NEXT:    [[CONVERT0:%[^ ]+]] = f32[2,2048,24576]{2,1,0} convert([[GTE0]])
; CHECK-NEXT:    [[GTE3:%[^ ]+]] = f32[] get-tuple-element([[P0]]), index=5
; CHECK-NEXT:    [[BCAST0:%[^ ]+]] = f32[2,2048,24576]{2,1,0} broadcast([[GTE3]]), dimensions={}
; CHECK-NEXT:    [[MUL0:%[^ ]+]] = f32[2,2048,24576]{2,1,0} multiply([[CONVERT0]], [[BCAST0]])
; CHECK-NEXT:    [[C0:%[^ ]+]] = s32[] constant(0)
; CHECK-NEXT:    [[C1:%[^ ]+]] = u32[] constant(0)
; CHECK-NEXT:    [[GTE4:%[^ ]+]] = u32[] get-tuple-element([[P0]]), index=4
; CHECK-NEXT:    [[C2:%[^ ]+]] = u32[] constant(3)
; CHECK-NEXT:    [[ADD0:%[^ ]+]] = u32[] add([[GTE4]], [[C2]])
; CHECK-NEXT:    [[PID:%[^ ]+]] = u32[] partition-id()
; CHECK-NEXT:    [[ADD2:%[^ ]+]] = u32[] add([[ADD0]], [[PID]])
; CHECK-NEXT:    [[AND0:%[^ ]+]] = u32[] and([[ADD2]], [[C2]])
; CHECK-NEXT:    [[CLAMP0:%[^ ]+]] = u32[] clamp([[C1]], [[AND0]], [[C2]])
; CHECK-NEXT:    [[CONVERT10:%[^ ]+]] = s32[] convert([[CLAMP0]])
; CHECK-NEXT:    [[C10:%[^ ]+]] = s32[] constant(512)
; CHECK-NEXT:    [[MUL10:%[^ ]+]] = s32[] multiply([[CONVERT10]], [[C10]])
; CHECK-NEXT:    [[RESHAPE0:%[^ ]+]] = s32[] reshape([[MUL10]])
; CHECK-NEXT:    [[DSLICE1:%[^ ]+]] = f32[2,512,24576]{2,1,0} dynamic-slice([[MUL0]], [[C0]], [[RESHAPE0]], [[C0]]), dynamic_slice_sizes={2,512,24576}
; CHECK-NEXT:    [[CONVERT1:%[^ ]+]] = f32[24576,24576]{1,0} convert([[GTE1]])
; CHECK-NEXT:    [[GTE5:%[^ ]+]] = f32[] get-tuple-element([[P0]]), index=6
; CHECK-NEXT:    [[BCAST1:%[^ ]+]] = f32[24576,24576]{1,0} broadcast([[GTE5]]), dimensions={}
; CHECK-NEXT:    [[MUL1:%[^ ]+]] = f32[24576,24576]{1,0} multiply([[CONVERT1]], [[BCAST1]])
; CHECK-NEXT:    [[DOT0:%[^ ]+]] = f32[2,512,24576]{2,1,0} dot([[DSLICE1]], [[MUL1]]),
; CHECK-DAG:       lhs_contracting_dims={2},
; CHECK-DAG:       rhs_contracting_dims={0},
; CHECK-DAG:       backend_config={
; CHECK-DAG:         "operation_queue_id":"[[OPQUEUEID0:[1-9][0-9]*]]",
; CHECK-DAG:         "wait_on_operation_queues":[],
; CHECK-DAG:         "force_earliest_schedule":false}
; CHECK-NEXT:    [[ADD3:%[^ ]+]] = f32[2,512,24576]{2,1,0} add([[CP0]], [[DOT0]]),
; CHECK-DAG:       backend_config={"
; CHECK-DAG:         operation_queue_id":"0",
; CHECK-DAG:         "wait_on_operation_queues":["[[OPQUEUEID0]]"],
; CHECK-DAG:         "force_earliest_schedule":false}
; CHECK-NEXT:    [[GTE6:[^ ]+]] = f32[2,512,24576]{2,1,0} get-tuple-element([[P0]]), index=3
; CHECK-NEXT:    [[C11:%[^ ]+]] = u32[] constant(0)
; CHECK-NEXT:    [[ADD6:%[^ ]+]] = u32[] add([[C11]], [[PID]])
; CHECK-NEXT:    [[AND1:%[^ ]+]] = u32[] and([[ADD6]], [[C2]])
; CHECK-NEXT:    [[CLAMP1:%[^ ]+]] = u32[] clamp([[C1]], [[AND1]], [[C2]])
; CHECK-NEXT:    [[CONVERT11:%[^ ]+]] = s32[] convert([[CLAMP1]])
; CHECK-NEXT:    [[MUL11:%[^ ]+]] = s32[] multiply([[CONVERT11]], [[C10]])
; CHECK-NEXT:    [[RESHAPE2:%[^ ]+]] = s32[] reshape([[MUL11]])
; CHECK-NEXT:    [[DSLICE3:%[^ ]+]] = f32[2,512,24576]{2,1,0} dynamic-slice([[MUL0]], [[C0]], [[RESHAPE2]], [[C0]]), dynamic_slice_sizes={2,512,24576}
; CHECK-NEXT:    [[DOT1:%[^ ]+]] = f32[2,512,24576]{2,1,0} dot([[DSLICE3]], [[MUL1]]),
; CHECK-DAG:       lhs_contracting_dims={2},
; CHECK-DAG:       rhs_contracting_dims={0}
; CHECK-DAG:       backend_config={
; CHECK-DAG:         "operation_queue_id":"[[OPQUEUEID:[0-9]+]]",
; CHECK-DAG:         "wait_on_operation_queues":[],
; CHECK-DAG:         "force_earliest_schedule":false}
; CHECK-NEXT:    [[ADD5:%[^ ]+]] = f32[2,512,24576]{2,1,0} add([[GTE6]], [[DOT1]])
; CHECK-NEXT:    [[CP1:[^ ]+]] = f32[2,512,24576]{2,1,0} collective-permute([[ADD5]]), channel_id=12
; CHECK-NEXT:    [[C3:%[^ ]+]] = u32[] constant(2)
; CHECK-NEXT:    [[ADD7:%[^ ]+]] = u32[] add([[GTE4]], [[C3]])
; CHECK-NEXT:    [[TUPLE0:[^ ]+]] = (f8e4m3fn[2,2048,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f32[2,512,24576]{2,1,0}, u32[], /*index=5*/f32[], f32[]) tuple([[GTE0]], [[GTE1]], [[ADD3]], [[CP1]], [[ADD7]], /*index=5*/[[GTE3]], [[GTE5]])
; CHECK-NEXT:    [[GTE0:%[^ ]+]] = f8e4m3fn[2,2048,24576]{2,1,0} get-tuple-element([[TUPLE0]]), index=0
; CHECK-NEXT:    [[GTE1:%[^ ]+]] = f8e4m3fn[24576,24576]{1,0} get-tuple-element([[TUPLE0]]), index=1
; CHECK-NEXT:    [[GTE2:%[^ ]+]] = f32[2,512,24576]{2,1,0} get-tuple-element([[TUPLE0]]), index=2
; CHECK-NEXT:    [[CP0:%[^ ]+]] = f32[2,512,24576]{2,1,0} collective-permute([[GTE2]]), channel_id=13
; CHECK-NEXT:    [[CONVERT0:%[^ ]+]] = f32[2,2048,24576]{2,1,0} convert([[GTE0]])
; CHECK-NEXT:    [[GTE3:%[^ ]+]] = f32[] get-tuple-element([[TUPLE0]]), index=5
; CHECK-NEXT:    [[BCAST0:%[^ ]+]] = f32[2,2048,24576]{2,1,0} broadcast([[GTE3]]), dimensions={}
; CHECK-NEXT:    [[MUL0:%[^ ]+]] = f32[2,2048,24576]{2,1,0} multiply([[CONVERT0]], [[BCAST0]])
; CHECK-NEXT:    [[C0:%[^ ]+]] = s32[] constant(0)
; CHECK-NEXT:    [[C1:%[^ ]+]] = u32[] constant(0)
; CHECK-NEXT:    [[GTE4:%[^ ]+]] = u32[] get-tuple-element([[TUPLE0]]), index=4
; CHECK-NEXT:    [[C2:%[^ ]+]] = u32[] constant(3)
; CHECK-NEXT:    [[ADD0:%[^ ]+]] = u32[] add([[GTE4]], [[C2]])
; CHECK-NEXT:    [[PID:%[^ ]+]] = u32[] partition-id()
; CHECK-NEXT:    [[ADD2:%[^ ]+]] = u32[] add([[ADD0]], [[PID]])
; CHECK-NEXT:    [[AND0:%[^ ]+]] = u32[] and([[ADD2]], [[C2]])
; CHECK-NEXT:    [[CLAMP0:%[^ ]+]] = u32[] clamp([[C1]], [[AND0]], [[C2]])
; CHECK-NEXT:    [[CONVERT10:%[^ ]+]] = s32[] convert([[CLAMP0]])
; CHECK-NEXT:    [[C10:%[^ ]+]] = s32[] constant(512)
; CHECK-NEXT:    [[MUL10:%[^ ]+]] = s32[] multiply([[CONVERT10]], [[C10]])
; CHECK-NEXT:    [[RESHAPE0:%[^ ]+]] = s32[] reshape([[MUL10]])
; CHECK-NEXT:    [[DSLICE1:%[^ ]+]] = f32[2,512,24576]{2,1,0} dynamic-slice([[MUL0]], [[C0]], [[RESHAPE0]], [[C0]]), dynamic_slice_sizes={2,512,24576}
; CHECK-NEXT:    [[CONVERT1:%[^ ]+]] = f32[24576,24576]{1,0} convert([[GTE1]])
; CHECK-NEXT:    [[GTE5:%[^ ]+]] = f32[] get-tuple-element([[TUPLE0]]), index=6
; CHECK-NEXT:    [[BCAST1:%[^ ]+]] = f32[24576,24576]{1,0} broadcast([[GTE5]]), dimensions={}
; CHECK-NEXT:    [[MUL1:%[^ ]+]] = f32[24576,24576]{1,0} multiply([[CONVERT1]], [[BCAST1]])
; CHECK-NEXT:    [[DOT0:%[^ ]+]] = f32[2,512,24576]{2,1,0} dot([[DSLICE1]], [[MUL1]]),
; CHECK-DAG:       lhs_contracting_dims={2},
; CHECK-DAG:       rhs_contracting_dims={0},
; CHECK-DAG:       backend_config={
; CHECK-DAG:         "operation_queue_id":"[[OPQUEUEID:[0-9]+]]",
; CHECK-DAG:         "wait_on_operation_queues":[],
; CHECK-DAG:         "force_earliest_schedule":false}
; CHECK-NEXT:    [[ADD3:%[^ ]+]] = f32[2,512,24576]{2,1,0} add([[CP0]], [[DOT0]]),
; CHECK-DAG:       backend_config={"
; CHECK-DAG:         operation_queue_id":"0",
; CHECK-DAG:         "wait_on_operation_queues":["[[OPQUEUEID]]"],
; CHECK-DAG:         "force_earliest_schedule":false}
; CHECK-NEXT:    [[GTE6:[^ ]+]] = f32[2,512,24576]{2,1,0} get-tuple-element([[TUPLE0]]), index=3
; CHECK-NEXT:    [[C11:%[^ ]+]] = u32[] constant(1)
; CHECK-NEXT:    [[ADD6:%[^ ]+]] = u32[] add([[C11]], [[PID]])
; CHECK-NEXT:    [[AND1:%[^ ]+]] = u32[] and([[ADD6]], [[C2]])
; CHECK-NEXT:    [[CLAMP1:%[^ ]+]] = u32[] clamp([[C1]], [[AND1]], [[C2]])
; CHECK-NEXT:    [[CONVERT11:%[^ ]+]] = s32[] convert([[CLAMP1]])
; CHECK-NEXT:    [[MUL11:%[^ ]+]] = s32[] multiply([[CONVERT11]], [[C10]])
; CHECK-NEXT:    [[RESHAPE2:%[^ ]+]] = s32[] reshape([[MUL11]])
; CHECK-NEXT:    [[DSLICE3:%[^ ]+]] = f32[2,512,24576]{2,1,0} dynamic-slice([[MUL0]], [[C0]], [[RESHAPE2]], [[C0]]), dynamic_slice_sizes={2,512,24576}
; CHECK-NEXT:    [[DOT1:%[^ ]+]] = f32[2,512,24576]{2,1,0} dot([[DSLICE3]], [[MUL1]]),
; CHECK-DAG:       lhs_contracting_dims={2},
; CHECK-DAG:       rhs_contracting_dims={0}
; CHECK-DAG:       backend_config={
; CHECK-DAG:         "operation_queue_id":"[[OPQUEUEID:[0-9]+]]",
; CHECK-DAG:         "wait_on_operation_queues":[],
; CHECK-DAG:         "force_earliest_schedule":false}
; CHECK-NEXT:    [[ADD5:%[^ ]+]] = f32[2,512,24576]{2,1,0} add([[GTE6]], [[DOT1]])
; CHECK-NEXT:    [[CP1:[^ ]+]] = f32[2,512,24576]{2,1,0} collective-permute([[ADD5]]), channel_id=14
; CHECK-NEXT:    [[C3:%[^ ]+]] = u32[] constant(2)
; CHECK-NEXT:    [[ADD7:%[^ ]+]] = u32[] add([[GTE4]], [[C3]])
)");
}

TEST_F(WindowedEinsumHandlerTest, AllGatherMultipleConsumersF8) {
  constexpr absl::string_view kHloString = R"(
HloModule all_gather_multiple_consumers_f8, entry_computation_layout={(f8e4m3fn[2,512,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f8e4m3fn[24576,24576]{1,0}, f8e4m3fn[24576,24576]{1,0}, f32[], f32[], f32[], f32[])->f32[2,2048,24576]{2,1,0}}, num_partitions=4
windowed_dot_general_body_ag {
  input = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) parameter(0)
  lhs = f32[2,512,24576]{2,1,0} get-tuple-element(input), index=0
  permuted_lhs0 = f32[2,512,24576]{2,1,0} collective-permute(lhs), channel_id=2, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  permuted_lhs1 = f32[2,512,24576]{2,1,0} collective-permute(permuted_lhs0), channel_id=3, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  rhs = f32[24576,24576]{1,0} get-tuple-element(input), index=1
  partial_dot_output = f32[2,2048,24576]{2,1,0} get-tuple-element(input), index=2
  dot0 = f32[2,512,24576]{2,1,0} dot(lhs, rhs), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  c0 = s32[] constant(0)
  dot_update_slice_offsets = s32[4]{0} constant({0, 512, 1024, 1536})
  loop_counter = u32[] get-tuple-element(input), index=4
  partition_id = u32[] partition-id()
  loop_counter_plus_partition_id = u32[] add(loop_counter, partition_id)
  c4 = u32[] constant(4)
  dot_update_slice_offsets_index0 = u32[] remainder(loop_counter_plus_partition_id, c4)
  dot_update_slice_offset0 = s32[1]{0} dynamic-slice(dot_update_slice_offsets, dot_update_slice_offsets_index0), dynamic_slice_sizes={1}
  dot_update_slice_offset_scalar0 = s32[] reshape(dot_update_slice_offset0)
  updated_dot_output0 = f32[2,2048,24576]{2,1,0} dynamic-update-slice(partial_dot_output, dot0, c0, dot_update_slice_offset_scalar0, c0)
  dot1 = f32[2,512,24576]{2,1,0} dot(permuted_lhs0, rhs), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  c1 = u32[] constant(1)
  loop_counter_plus_one = u32[] add(loop_counter, c1)
  loop_counter_plus_partition_id_plus_one = u32[] add(loop_counter_plus_one, partition_id)
  dot_update_slice_offsets_index1 = u32[] remainder(loop_counter_plus_partition_id_plus_one, c4)
  dot_update_slice_offset1 = s32[1]{0} dynamic-slice(dot_update_slice_offsets, dot_update_slice_offsets_index1), dynamic_slice_sizes={1}
  dot_update_slice_offset1_scalar = s32[] reshape(dot_update_slice_offset1)
  updated_dot_output1 = f32[2,2048,24576]{2,1,0} dynamic-update-slice(updated_dot_output0, dot1, c0, dot_update_slice_offset1_scalar, c0)
  pass_through = f32[2,2048,24576]{2,1,0} get-tuple-element(input), index=3
  next_loop_counter = u32[] add(loop_counter_plus_one, c1)
  ROOT tuple = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) tuple(permuted_lhs1, rhs, updated_dot_output1, pass_through, next_loop_counter)
} // windowed_dot_general_body_ag

windowed_dot_general_cond_ag {
  input = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) parameter(0)
  loop_counter = u32[] get-tuple-element(input), index=4
  loop_limit = u32[] constant(4)
  ROOT compare = pred[] compare(loop_counter, loop_limit), direction=LT
}

ENTRY main {
  lhs = f8e4m3fn[2,512,24576]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
  rhs0 = f8e4m3fn[24576,24576]{1,0} parameter(1), sharding={devices=[1,4]<=[4]}
  c0_f32 = f32[] constant(0)
  c0_f32_bcast = f32[2,2048,24576]{2,1,0} broadcast(c0_f32), dimensions={}
  c0_u32 = u32[] constant(0)
  // Dequantization of LHS and RHS:
  scale_lhs = f32[] parameter(4)
  scale_lhs_bcast = f32[2,512,24576]{2,1,0} broadcast(scale_lhs), dimensions={}
  lhs_f32 = f32[2,512,24576]{2,1,0} convert(lhs)
  lhs_scaled = f32[2,512,24576]{2,1,0} multiply(lhs_f32, scale_lhs_bcast)
  scale_rhs0 = f32[] parameter(5)
  scale_rhs0_bcast = f32[24576,24576]{1,0} broadcast(scale_rhs0), dimensions={}
  rhs0_f32 = f32[24576,24576]{1,0} convert(rhs0)
  rhs0_scaled = f32[24576,24576]{1,0} multiply(rhs0_f32, scale_rhs0_bcast)
  // While loop of all-gather windowed einsum:
  while_input = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) tuple(lhs_scaled, rhs0_scaled, c0_f32_bcast, c0_f32_bcast, c0_u32)
  while = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) while(while_input), condition=windowed_dot_general_cond_ag, body=windowed_dot_general_body_ag
  // Additional all-gather FP8 dot operating on a dequantized RHS and the LHS also consumed by the windowed einsum.
  all-gather1 = f32[2,2048,24576]{2,1,0} all-gather(lhs_scaled), channel_id=1, replica_groups={{0,1,2,3}}, dimensions={1}, use_global_device_ids=true
  rhs1 = f8e4m3fn[24576,24576]{1,0} parameter(2), sharding={devices=[1,4]<=[4]}
  scale_rhs1 = f32[] parameter(6)
  scale_rhs1_bcast = f32[24576,24576]{1,0} broadcast(scale_rhs1), dimensions={}
  rhs1_f32 = f32[24576,24576]{1,0} convert(rhs1)
  rhs1_scaled = f32[24576,24576]{1,0} multiply(rhs1_f32, scale_rhs1_bcast)
  dot1 = f32[2,2048,24576]{2,1,0} dot(all-gather1, rhs1_scaled), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  // Another all-gather FP8 dot operating on a dequantized RHS and the LHS also consumed by the windowed einsum.
  all-gather2 = f32[2,2048,24576]{2,1,0} all-gather(lhs_scaled), channel_id=1, replica_groups={{0,1,2,3}}, dimensions={1}, use_global_device_ids=true
  rhs2 = f8e4m3fn[24576,24576]{1,0} parameter(3), sharding={devices=[1,4]<=[4]}
  scale_rhs2 = f32[] parameter(7)
  scale_rhs2_bcast = f32[24576,24576]{1,0} broadcast(scale_rhs2), dimensions={}
  rhs2_f32 = f32[24576,24576]{1,0} convert(rhs2)
  rhs2_scaled = f32[24576,24576]{1,0} multiply(rhs2_f32, scale_rhs2_bcast)
  dot2 = f32[2,2048,24576]{2,1,0} dot(all-gather2, rhs2_scaled), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT product = f32[2,2048,24576]{2,1,0} multiply(dot1, dot2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  RunAndFilecheckHloRewrite(kHloString, WindowedEinsumHandler(),
                            R"(
; CHECK-LABEL: %main
; CHECK:         [[WHILE0:%[^ ]+]] = (f8e4m3fn[2,512,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[], /*index=5*/f32[], f32[], f8e4m3fn[2,2048,24576]{2,1,0}) while([[TUPLE0:%[^ ]+]]),
; CHECK-DAG:       condition=%unrolled_windowed_dot_general_cond_ag,
; CHECK-DAG:       body=%unrolled_windowed_dot_general_body_ag
; CHECK:         [[LHS1:%[^ ]+]] = f8e4m3fn[2,2048,24576]{2,1,0} get-tuple-element([[WHILE0]]), index=7
; CHECK-NEXT:    [[LHS1_F32:%[^ ]+]] = f32[2,2048,24576]{2,1,0} convert([[LHS1]])
; CHECK-NEXT:    [[SCALE_LHS1_BCAST:%[^ ]+]] = f32[2,2048,24576]{2,1,0} broadcast([[SCALE_LHS1:%[^ ]+]]), dimensions={}
; CHECK-NEXT:    [[LHS1_SCALED:%[^ ]+]] = f32[2,2048,24576]{2,1,0} multiply([[LHS1_F32]], [[SCALE_LHS1_BCAST]])
; CHECK-NEXT:    [[RHS1:%[^ ]+]] = f8e4m3fn[24576,24576]{1,0} parameter(2), sharding={devices=[1,4]<=[4]}
; CHECK-NEXT:    [[RHS1_F32:%[^ ]+]] = f32[24576,24576]{1,0} convert([[RHS1]])
; CHECK:         [[SCALE_RHS1_BCAST:%[^ ]+]] = f32[24576,24576]{1,0} broadcast([[SCALE_RHS1:%[^ ]+]]), dimensions={}
; CHECK-NEXT:    [[RHS1_SCALED:%[^ ]+]] = f32[24576,24576]{1,0} multiply([[RHS1_F32]], [[SCALE_RHS1_BCAST]])
; CHECK-NEXT:    [[DOT1:%[^ ]+]] = f32[2,2048,24576]{2,1,0} dot([[LHS1_SCALED]], [[RHS1_SCALED]]),
; CHECK-DAG:       lhs_contracting_dims={2},
; CHECK-DAG:       rhs_contracting_dims={0}
; CHECK:         [[LHS2:%[^ ]+]] = f8e4m3fn[2,2048,24576]{2,1,0} get-tuple-element([[WHILE0]]), index=7
; CHECK-NEXT:    [[LHS2_F32:%[^ ]+]] = f32[2,2048,24576]{2,1,0} convert([[LHS2]])
; CHECK-NEXT:    [[SCALE_LHS2_BCAST:%[^ ]+]] = f32[2,2048,24576]{2,1,0} broadcast([[SCALE_LHS2:%[^ ]+]]), dimensions={}
; CHECK-NEXT:    [[LHS2_SCALED:%[^ ]+]] = f32[2,2048,24576]{2,1,0} multiply([[LHS2_F32]], [[SCALE_LHS2_BCAST]])
; CHECK-NEXT:    [[RHS2:%[^ ]+]] = f8e4m3fn[24576,24576]{1,0} parameter(3), sharding={devices=[1,4]<=[4]}
; CHECK-NEXT:    [[RHS2_F32:%[^ ]+]] = f32[24576,24576]{1,0} convert([[RHS2]])
; CHECK-NEXT:    [[SCALE_RHS2:%[^ ]+]] = f32[] parameter(7)
; CHECK-NEXT:    [[SCALE_RHS2_BCAST:%[^ ]+]] = f32[24576,24576]{1,0} broadcast([[SCALE_RHS2]]), dimensions={}
; CHECK-NEXT:    [[RHS2_SCALED:%[^ ]+]] = f32[24576,24576]{1,0} multiply([[RHS2_F32]], [[SCALE_RHS2_BCAST]])
; CHECK-NEXT:    [[DOT2:%[^ ]+]] = f32[2,2048,24576]{2,1,0} dot([[LHS2_SCALED]], [[RHS2_SCALED]]),
; CHECK-DAG:       lhs_contracting_dims={2},
; CHECK-DAG:       rhs_contracting_dims={0}
; CHECK-NEXT:  ROOT [[OUT:[^ ]+]] = f32[2,2048,24576]{2,1,0} multiply([[DOT1]], [[DOT2]])
)");
}

TEST_F(WindowedEinsumHandlerTest,
       AgLoopsMultipleConsumersAreChainedWithShardedContratingDim) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[16,2048,512]{2,1,0}, bf16[4096,6288]{1,0}, bf16[16,2048,6288]{2,1,0})->(bf16[16,2048,6288]{2,1,0}, bf16[4096,6288]{1,0})}, num_partitions=8

windowed_dot_general_body_ag {
  param.195 = (bf16[16,2048,512]{2,1,0}, bf16[4096,6288]{1,0}, bf16[16,2048,6288]{2,1,0}, bf16[16,2048,6288]{2,1,0}, u32[]) parameter(0)
  get-tuple-element.588 = bf16[16,2048,512]{2,1,0} get-tuple-element(param.195), index=0
  collective-permute.194 = bf16[16,2048,512]{2,1,0} collective-permute(get-tuple-element.588), channel_id=446, source_target_pairs={{0,7},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6}}
  collective-permute.195 = bf16[16,2048,512]{2,1,0} collective-permute(collective-permute.194), channel_id=447, source_target_pairs={{0,7},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6}}
  get-tuple-element.589 = bf16[4096,6288]{1,0} get-tuple-element(param.195), index=1
  get-tuple-element.590 = bf16[16,2048,6288]{2,1,0} get-tuple-element(param.195), index=2
  constant.11432 = s32[8]{0} constant({0, 512, 1024, 1536, 2048, 2560, 3072, 3584})
  get-tuple-element.592 = u32[] get-tuple-element(param.195), index=4
  partition-id.194 = u32[] partition-id()
  add.4309 = u32[] add(get-tuple-element.592, partition-id.194)
  constant.11431 = u32[] constant(8)
  remainder.194 = u32[] remainder(add.4309, constant.11431)
  dynamic-slice.388 = s32[1]{0} dynamic-slice(constant.11432, remainder.194), dynamic_slice_sizes={1}
  reshape.12959 = s32[] reshape(dynamic-slice.388)
  constant.11433 = s32[] constant(0)
  dynamic-slice.389 = bf16[512,6288]{1,0} dynamic-slice(get-tuple-element.589, reshape.12959, constant.11433), dynamic_slice_sizes={512,6288}
  dot.244 = bf16[16,2048,6288]{2,1,0} dot(get-tuple-element.588, dynamic-slice.389), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  add.4310 = bf16[16,2048,6288]{2,1,0} add(get-tuple-element.590, dot.244)
  constant.11434 = u32[] constant(1)
  add.4312 = u32[] add(get-tuple-element.592, constant.11434)
  add.4313 = u32[] add(add.4312, partition-id.194)
  remainder.195 = u32[] remainder(add.4313, constant.11431)
  dynamic-slice.390 = s32[1]{0} dynamic-slice(constant.11432, remainder.195), dynamic_slice_sizes={1}
  reshape.12960 = s32[] reshape(dynamic-slice.390)
  dynamic-slice.391 = bf16[512,6288]{1,0} dynamic-slice(get-tuple-element.589, reshape.12960, constant.11433), dynamic_slice_sizes={512,6288}
  dot.245 = bf16[16,2048,6288]{2,1,0} dot(collective-permute.194, dynamic-slice.391), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  add.4314 = bf16[16,2048,6288]{2,1,0} add(add.4310, dot.245)
  get-tuple-element.591 = bf16[16,2048,6288]{2,1,0} get-tuple-element(param.195), index=3
  add.4315 = u32[] add(add.4312, constant.11434)
  ROOT tuple.98 = (bf16[16,2048,512]{2,1,0}, bf16[4096,6288]{1,0}, bf16[16,2048,6288]{2,1,0}, bf16[16,2048,6288]{2,1,0}, u32[]) tuple(collective-permute.195, get-tuple-element.589, add.4314, get-tuple-element.591, add.4315)
} // windowed_dot_general_body_ag

windowed_dot_general_cond_ag {
  param = (bf16[16,2048,512]{2,1,0}, bf16[4096,6288]{1,0}, bf16[16,2048,6288]{2,1,0}, bf16[16,2048,6288]{2,1,0}, u32[]) parameter(0)
  get-tuple-element = u32[] get-tuple-element(param), index=4
  constant = u32[] constant(4)
  ROOT compare = pred[] compare(get-tuple-element, constant), direction=LT
}

ENTRY main.12_spmd {
  param.4 = bf16[16,2048,512]{2,1,0} parameter(0)
  param.5 = bf16[4096,6288]{1,0} parameter(1)
  constant.22 = bf16[] constant(0)
  broadcast = bf16[16,2048,6288]{2,1,0} broadcast(constant.22), dimensions={}
  constant.24 = u32[] constant(0)
  tuple.2 = (bf16[16,2048,512]{2,1,0}, bf16[4096,6288]{1,0}, bf16[16,2048,6288]{2,1,0}, bf16[16,2048,6288]{2,1,0}, u32[]) tuple(param.4, param.5, broadcast, broadcast, constant.24)
  while = (bf16[16,2048,512]{2,1,0}, bf16[4096,6288]{1,0}, bf16[16,2048,6288]{2,1,0}, bf16[16,2048,6288]{2,1,0}, u32[]) while(tuple.2), condition=windowed_dot_general_cond_ag, body=windowed_dot_general_body_ag
  get-tuple-element.result = bf16[16,2048,6288]{2,1,0} get-tuple-element(while), index=2
  all-gather = bf16[16,2048,4096]{2,1,0} all-gather(param.4), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={2}, use_global_device_ids=true
  param.6 = bf16[16,2048,6288]{2,1,0} parameter(2)
  dot.7 = bf16[4096,6288]{1,0} dot(all-gather, param.6), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
  ROOT tuple.output = (bf16[16,2048,6288]{2,1,0}, bf16[4096,6288]{1,0}) tuple(get-tuple-element.result, dot.7)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  WindowedEinsumHandler gpu_handler;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* ag_loop =
      FindInstructionByName(module->entry_computation(), "while");
  HloInstruction* inst =
      FindInstructionByName(module->entry_computation(), "dot.7");
  // dot.7 should now consume output of the windowed einsum while loop.
  EXPECT_EQ(inst->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(inst->operand(0)->tuple_index(), 5);
  EXPECT_EQ(inst->operand(0)->operand(0), ag_loop);

  EXPECT_EQ(ag_loop->operand(0)->shape().tuple_shapes_size(), 7);
  // The root instruction's first operand should now be a reduction.
  EXPECT_EQ(
      module->entry_computation()->root_instruction()->operand(0)->opcode(),
      HloOpcode::kReduce);
}

}  // namespace
}  // namespace xla::gpu
