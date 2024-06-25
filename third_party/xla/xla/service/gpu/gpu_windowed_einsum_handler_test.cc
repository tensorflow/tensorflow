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

#include "xla/service/gpu/gpu_windowed_einsum_handler.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

using GpuWindowedEinsumHanlderTest = HloTestBase;

HloInstruction* FindInstructionByName(HloComputation* comp, std::string name) {
  for (auto inst : comp->instructions()) {
    if (inst->name() == name) {
      return inst;
    }
  }
  return nullptr;
}

TEST_F(GpuWindowedEinsumHanlderTest, AgLoopsHaveStreamIds) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[1,512,24576]{2,1,0}, bf16[24576,24576]{1,0})->bf16[2048,24576]{1,0}}, num_partitions=4

windowed_dot_general_body_ag.1 {
  param = (bf16[512,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[2048,24576]{1,0}, bf16[2048,24576]{1,0}, u32[]) parameter(0)
  get-tuple-element = bf16[512,24576]{1,0} get-tuple-element(param), index=0
  collective-permute = bf16[512,24576]{1,0} collective-permute(get-tuple-element), channel_id=2, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
  get-tuple-element.1 = bf16[24576,24576]{1,0} get-tuple-element(param), index=1
  get-tuple-element.2 = bf16[2048,24576]{1,0} get-tuple-element(param), index=2
  dot.2 = bf16[512,24576]{1,0} dot(get-tuple-element, get-tuple-element.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
  constant.1 = s32[4]{0} constant({0, 512, 1024, 1536})
  get-tuple-element.4 = u32[] get-tuple-element(param), index=4
  partition-id = u32[] partition-id()
  add = u32[] add(get-tuple-element.4, partition-id)
  constant = u32[] constant(4)
  remainder = u32[] remainder(add, constant)
  dynamic-slice = s32[1]{0} dynamic-slice(constant.1, remainder), dynamic_slice_sizes={1}
  reshape.4 = s32[] reshape(dynamic-slice)
  constant.2 = s32[] constant(0)
  dynamic-update-slice = bf16[2048,24576]{1,0} dynamic-update-slice(get-tuple-element.2, dot.2, reshape.4, constant.2), backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
  dot.3 = bf16[512,24576]{1,0} dot(collective-permute, get-tuple-element.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  constant.3 = u32[] constant(1)
  add.1 = u32[] add(get-tuple-element.4, constant.3)
  add.2 = u32[] add(add.1, partition-id)
  remainder.1 = u32[] remainder(add.2, constant)
  dynamic-slice.1 = s32[1]{0} dynamic-slice(constant.1, remainder.1), dynamic_slice_sizes={1}
  reshape.5 = s32[] reshape(dynamic-slice.1)
  dynamic-update-slice.1 = bf16[2048,24576]{1,0} dynamic-update-slice(dynamic-update-slice, dot.3, reshape.5, constant.2)
  get-tuple-element.3 = bf16[2048,24576]{1,0} get-tuple-element(param), index=3
  add.3 = u32[] add(add.1, constant.3)
  ROOT tuple = (bf16[512,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[2048,24576]{1,0}, bf16[2048,24576]{1,0}, u32[]) tuple(collective-permute, get-tuple-element.1, dynamic-update-slice.1, get-tuple-element.3, add.3)
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

  GpuWindowedEinsumHandler gpu_handler;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* ag_loop =
      module->entry_computation()->root_instruction()->mutable_operand(0);
  HloComputation* ag_loop_body = ag_loop->while_body();
  HloInstruction* inst = FindInstructionByName(ag_loop_body, "dot.2");
  EXPECT_GT(inst->backend_config<GpuBackendConfig>()->operation_queue_id(), 0);
  EXPECT_TRUE(
      inst->backend_config<GpuBackendConfig>()->force_earliest_schedule());

  HloInstruction* cp1 =
      FindInstructionByName(ag_loop_body, "collective-permute");
  EXPECT_TRUE(
      cp1->backend_config<GpuBackendConfig>()->force_earliest_schedule());
}

TEST_F(GpuWindowedEinsumHanlderTest, RsLoopsHaveStreamIds) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[24576,24576]{1,0}, bf16[512,24576]{1,0}, bf16[2048,24576]{1,0})->bf16[512,24576]{1,0}}, num_partitions=4

windowed_dot_general_body_rs_clone.1 {
  param.2 = (bf16[2048,24576]{1,0}, bf16[24576,24576]{1,0}, bf16[512,24576]{1,0}, bf16[512,24576]{1,0}, u32[]) parameter(0)
  get-tuple-element.6 = bf16[2048,24576]{1,0} get-tuple-element(param.2), index=0
  get-tuple-element.7 = bf16[24576,24576]{1,0} get-tuple-element(param.2), index=1
  get-tuple-element.9 = bf16[512,24576]{1,0} get-tuple-element(param.2), index=2
  collective-permute.1 = bf16[512,24576]{1,0} collective-permute(get-tuple-element.9), channel_id=4, source_target_pairs={{0,2},{1,3},{2,0},{3,1}}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
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
  add.11 = bf16[512,24576]{1,0} add(collective-permute.1, dot.7), backend_config={"operation_queue_id":"0","wait_on_operation_queues":[]}
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

  GpuWindowedEinsumHandler gpu_handler;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* rs_loop =
      module->entry_computation()->root_instruction()->mutable_operand(0);
  HloComputation* rs_loop_body = rs_loop->while_body();
  HloInstruction* inst = FindInstructionByName(rs_loop_body, "dot.7");
  EXPECT_TRUE(inst->backend_config<GpuBackendConfig>()->operation_queue_id() >
              0);

  HloInstruction* cp1 =
      FindInstructionByName(rs_loop_body, "collective-permute.1");
  EXPECT_TRUE(
      cp1->backend_config<GpuBackendConfig>()->force_earliest_schedule());
}

TEST_F(GpuWindowedEinsumHanlderTest, AgLoopsMultipleConsumersAreChained) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(bf16[2,512,24576]{2,1,0}, bf16[24576,24576]{1,0}, bf16[24576,24576]{1,0})->bf16[2,2048,24576]{2,1,0}}, num_partitions=4

windowed_dot_general_body_ag {
  param.1 = (bf16[2,512,24576]{2,1,0}, bf16[24576,24576]{1,0}, bf16[2,2048,24576]{2,1,0}, bf16[2,2048,24576]{2,1,0}, u32[]) parameter(0)
  get-tuple-element.1 = bf16[2,512,24576]{2,1,0} get-tuple-element(param.1), index=0
  collective-permute = bf16[2,512,24576]{2,1,0} collective-permute(get-tuple-element.1), channel_id=2, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  collective-permute.1 = bf16[2,512,24576]{2,1,0} collective-permute(collective-permute), channel_id=3, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  get-tuple-element.2 = bf16[24576,24576]{1,0} get-tuple-element(param.1), index=1
  get-tuple-element.3 = bf16[2,2048,24576]{2,1,0} get-tuple-element(param.1), index=2
  dot = bf16[2,512,24576]{2,1,0} dot(get-tuple-element.1, get-tuple-element.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}
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
  dot.1 = bf16[2,512,24576]{2,1,0} dot(collective-permute, get-tuple-element.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  constant.5 = u32[] constant(1)
  add.1 = u32[] add(get-tuple-element.5, constant.5)
  add.2 = u32[] add(add.1, partition-id)
  remainder.1 = u32[] remainder(add.2, constant.1)
  dynamic-slice.1 = s32[1]{0} dynamic-slice(constant.3, remainder.1), dynamic_slice_sizes={1}
  reshape.1 = s32[] reshape(dynamic-slice.1)
  dynamic-update-slice.1 = bf16[2,2048,24576]{2,1,0} dynamic-update-slice(dynamic-update-slice, dot.1, constant.2, reshape.1, constant.2)
  get-tuple-element.4 = bf16[2,2048,24576]{2,1,0} get-tuple-element(param.1), index=3
  add.3 = u32[] add(add.1, constant.5)
  ROOT tuple = (bf16[2,512,24576]{2,1,0}, bf16[24576,24576]{1,0}, bf16[2,2048,24576]{2,1,0}, bf16[2,2048,24576]{2,1,0}, u32[]) tuple(collective-permute.1, get-tuple-element.2, dynamic-update-slice.1, get-tuple-element.4, add.3)
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

  GpuWindowedEinsumHandler gpu_handler;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  EXPECT_TRUE(changed);
  HloInstruction* ag_loop =
      FindInstructionByName(module->entry_computation(), "while");
  HloInstruction* inst =
      FindInstructionByName(module->entry_computation(), "dot.7");
  // dot.7 should now consume output of the windowed einsum while loop.
  EXPECT_EQ(inst->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(inst->operand(0)->tuple_index(), 3);
  EXPECT_EQ(inst->operand(0)->operand(0), ag_loop);

  // while loop's root should now have a chain of DUS.
  HloInstruction* ag_while_root = ag_loop->while_body()->root_instruction();
  EXPECT_THAT(ag_while_root,
              GmockMatch(m::Tuple(
                  m::Op(), m::Op(), m::Op(),
                  m::DynamicUpdateSlice(
                      m::DynamicUpdateSlice(
                          m::GetTupleElement(m::Parameter())
                              .WithPredicate([](const HloInstruction* instr) {
                                return instr->tuple_index() == 3;
                              }),
                          m::Op(), m::Op(), m::Op(), m::Op()),
                      m::Op(), m::Op(), m::Op(), m::Op()),
                  m::Op())));
}
TEST_F(GpuWindowedEinsumHanlderTest, A2aGemmHaveStreamIds) {
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

CHECK-DAG: %[[SLICE0:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(bf16[1,4,2048,8192]{3,2,1,0} %[[P1]]), slice={[0:1], [0:4], [0:2048], [6144:8192]}
CHECK: %[[A2A0:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(bf16[1,4,2048,2048]{3,2,1,0} %[[SLICE0]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3},{4,5,6,7}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[P0:.*]] = bf16[1,8192,32768]{2,1,0} parameter(0)
CHECK-DAG: %[[SLICE4:.*]] = bf16[1,2048,32768]{2,1,0} slice(bf16[1,8192,32768]{2,1,0} %[[P0:.*]]), slice={[0:1], [6144:8192], [0:32768]}
CHECK-DAG: %[[DOT0:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(bf16[1,4,2048,2048]{3,2,1,0} %[[A2A0:.*]], bf16[1,2048,32768]{2,1,0} %[[SLICE4:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"8","wait_on_operation_queues":[],"force_earliest_schedule":false}

CHECK-DAG: %[[SLICE1:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(bf16[1,4,2048,8192]{3,2,1,0} %[[P1]]), slice={[0:1], [0:4], [0:2048], [4096:6144]}
CHECK: %[[A2A1:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(bf16[1,4,2048,2048]{3,2,1,0} %[[SLICE1]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3},{4,5,6,7}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[SLICE5:.*]] = bf16[1,2048,32768]{2,1,0} slice(bf16[1,8192,32768]{2,1,0} %[[P0:.*]]), slice={[0:1], [4096:6144], [0:32768]}
CHECK-DAG: %[[DOT1:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(bf16[1,4,2048,2048]{3,2,1,0} %[[A2A1:.*]], bf16[1,2048,32768]{2,1,0} %[[SLICE5:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"7","wait_on_operation_queues":[],"force_earliest_schedule":false}

CHECK-DAG: %[[SLICE2:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(bf16[1,4,2048,8192]{3,2,1,0} %[[P1]]), slice={[0:1], [0:4], [0:2048], [2048:4096]}
CHECK: %[[A2A2:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(bf16[1,4,2048,2048]{3,2,1,0} %[[SLICE2]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3},{4,5,6,7}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[SLICE6:.*]] = bf16[1,2048,32768]{2,1,0} slice(bf16[1,8192,32768]{2,1,0} %[[P0:.*]]), slice={[0:1], [2048:4096], [0:32768]}
CHECK-DAG: %[[DOT2:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(bf16[1,4,2048,2048]{3,2,1,0} %[[A2A2:.*]], bf16[1,2048,32768]{2,1,0} %[[SLICE6:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"6","wait_on_operation_queues":[],"force_earliest_schedule":false}

CHECK-DAG: %[[SLICE3:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(bf16[1,4,2048,8192]{3,2,1,0} %[[P1]]), slice={[0:1], [0:4], [0:2048], [0:2048]}
CHECK: %[[A2A2:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(bf16[1,4,2048,2048]{3,2,1,0} %[[SLICE3]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3},{4,5,6,7}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[SLICE7:.*]] = bf16[1,2048,32768]{2,1,0} slice(bf16[1,8192,32768]{2,1,0} %[[P0:.*]]), slice={[0:1], [0:2048], [0:32768]}
CHECK-DAG: %[[DOT3:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(bf16[1,4,2048,2048]{3,2,1,0} %[[A2A3:.*]], bf16[1,2048,32768]{2,1,0} %[[SLICE7:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"5","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK-DAG: %[[CONSTANT:.*]] = bf16[] constant(0)
CHECK-DAG: %[[BROADCAST:.*]] = bf16[1,4,2048,32768]{3,2,1,0} broadcast(bf16[] %[[CONSTANT:.*]]), dimensions={}
CHECK-DAG: %[[ADD0:.*]] = bf16[1,4,2048,32768]{3,2,1,0} add(bf16[1,4,2048,32768]{3,2,1,0} %[[DOT0:.*]], bf16[1,4,2048,32768]{3,2,1,0} %[[BROADCAST:.*]]), backend_config={"operation_queue_id":"0","wait_on_operation_queues":["5"],"force_earliest_schedule":false}
CHECK-DAG: %[[ADD1:.*]] = bf16[1,4,2048,32768]{3,2,1,0} add(bf16[1,4,2048,32768]{3,2,1,0} %[[DOT1:.*]], bf16[1,4,2048,32768]{3,2,1,0} %[[ADD0:.*]]), backend_config={"operation_queue_id":"0","wait_on_operation_queues":["6"],"force_earliest_schedule":false}
CHECK-DAG: %[[ADD2:.*]] = bf16[1,4,2048,32768]{3,2,1,0} add(bf16[1,4,2048,32768]{3,2,1,0} %[[DOT2:.*]], bf16[1,4,2048,32768]{3,2,1,0} %[[ADD1:.*]]), backend_config={"operation_queue_id":"0","wait_on_operation_queues":["7"],"force_earliest_schedule":false}

CHECK: ROOT {{.*}} = bf16[1,4,2048,32768]{3,2,1,0} add(bf16[1,4,2048,32768]{3,2,1,0} %[[DOT3:.*]], bf16[1,4,2048,32768]{3,2,1,0} %[[ADD2:.*]])
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  GpuWindowedEinsumHandler gpu_handler;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(module->ToString(), kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(GpuWindowedEinsumHanlderTest, GemmA2aHaveStreamIds) {
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

CHECK-DAG: %[[SLICE0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(bf16[1,4,2048,32768]{3,2,1,0} %[[P1]]), slice={[0:1], [0:4], [0:2048], [24576:32768]}
CHECK-DAG: %[[P0:.*]] = bf16[1,8192,32768]{2,1,0} parameter(0)
CHECK-DAG: %[[SLICE4:.*]] = bf16[1,8192,8192]{2,1,0} slice(bf16[1,8192,32768]{2,1,0} %[[P0:.*]]), slice={[0:1], [0:8192], [24576:32768]}
CHECK-DAG: %[[DOT0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(bf16[1,4,2048,8192]{3,2,1,0} %[[SLICE0:.*]], bf16[1,8192,8192]{2,1,0} %[[SLICE4:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={2}, backend_config={"operation_queue_id":"8","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(bf16[1,4,2048,8192]{3,2,1,0} %[[DOT0:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}

CHECK-DAG: %[[SLICE1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(bf16[1,4,2048,32768]{3,2,1,0} %[[P1]]), slice={[0:1], [0:4], [0:2048], [16384:24576]}
CHECK-DAG: %[[SLICE5:.*]] = bf16[1,8192,8192]{2,1,0} slice(bf16[1,8192,32768]{2,1,0} %[[P0:.*]]), slice={[0:1], [0:8192], [16384:24576]}
CHECK-DAG: %[[DOT1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(bf16[1,4,2048,8192]{3,2,1,0} %[[SLICE1:.*]], bf16[1,8192,8192]{2,1,0} %[[SLICE5:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={2}, backend_config={"operation_queue_id":"7","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(bf16[1,4,2048,8192]{3,2,1,0} %[[DOT1:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}

CHECK-DAG: %[[SLICE2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(bf16[1,4,2048,32768]{3,2,1,0} %[[P1]]), slice={[0:1], [0:4], [0:2048], [8192:16384]}
CHECK-DAG: %[[SLICE6:.*]] = bf16[1,8192,8192]{2,1,0} slice(bf16[1,8192,32768]{2,1,0} %[[P0:.*]]), slice={[0:1], [0:8192], [8192:16384]}
CHECK-DAG: %[[DOT2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(bf16[1,4,2048,8192]{3,2,1,0} %[[SLICE2:.*]], bf16[1,8192,8192]{2,1,0} %[[SLICE6:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={2}, backend_config={"operation_queue_id":"6","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(bf16[1,4,2048,8192]{3,2,1,0} %[[DOT2:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}

CHECK-DAG: %[[SLICE3:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(bf16[1,4,2048,32768]{3,2,1,0} %[[P1]]), slice={[0:1], [0:4], [0:2048], [0:8192]}
CHECK-DAG: %[[SLICE7:.*]] = bf16[1,8192,8192]{2,1,0} slice(bf16[1,8192,32768]{2,1,0} %[[P0:.*]]), slice={[0:1], [0:8192], [0:8192]}
CHECK-DAG: %[[DOT3:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(bf16[1,4,2048,8192]{3,2,1,0} %[[SLICE3:.*]], bf16[1,8192,8192]{2,1,0} %[[SLICE7:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={2}, backend_config={"operation_queue_id":"5","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(bf16[1,4,2048,8192]{3,2,1,0} %[[DOT3:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[CONSTANT:.*]] = bf16[] constant(0)
CHECK-DAG: %[[BROADCAST:.*]] = bf16[1,4,2048,8192]{3,2,1,0} broadcast(bf16[] %[[CONSTANT:.*]]), dimensions={}
CHECK-DAG: %[[ADD0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(bf16[1,4,2048,8192]{3,2,1,0} %[[A2A0:.*]], bf16[1,4,2048,8192]{3,2,1,0} %[[BROADCAST:.*]])
CHECK-DAG: %[[ADD1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(bf16[1,4,2048,8192]{3,2,1,0} %[[A2A1:.*]], bf16[1,4,2048,8192]{3,2,1,0} %[[ADD0:.*]])
CHECK-DAG: %[[ADD2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(bf16[1,4,2048,8192]{3,2,1,0} %[[A2A2:.*]], bf16[1,4,2048,8192]{3,2,1,0} %[[ADD1:.*]])

CHECK: ROOT {{.*}} = bf16[1,4,2048,8192]{3,2,1,0} add(bf16[1,4,2048,8192]{3,2,1,0} %[[A2A3:.*]], bf16[1,4,2048,8192]{3,2,1,0} %[[ADD2:.*]])
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  GpuWindowedEinsumHandler gpu_handler;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(module->ToString(), kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(GpuWindowedEinsumHanlderTest, A2aTransposeLoopsHaveStreamIds) {
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
CHECK-DAG: %[[TRANSPOSE0:.*]] = bf16[1,4,1,8192,1,2048]{5,4,1,3,2,0} transpose(bf16[1,1,8192,4,1,2048]{5,4,3,2,1,0} %[[P1:.*]]), dimensions={0,3,1,2,4,5}
CHECK-DAG: %[[RESHAPE0:.*]] = bf16[1,4,8192,1,2048]{4,3,2,1,0} reshape(bf16[1,4,1,8192,1,2048]{5,4,1,3,2,0} %[[TRANSPOSE0:.*]])
CHECK-DAG: %[[RESHAPE1:.*]] = bf16[4,8192,1,2048]{3,2,1,0} reshape(bf16[1,4,8192,1,2048]{4,3,2,1,0} %[[RESHAPE0:.*]])
CHECK-DAG: %[[TRANSPOSE1:.*]] = bf16[1,4,2048,8192]{2,0,3,1} transpose(bf16[4,8192,1,2048]{3,2,1,0} %[[RESHAPE1:.*]]), dimensions={2,0,3,1}
CHECK-DAG: %[[COPY:.*]] = bf16[1,4,2048,8192]{3,2,1,0} copy(bf16[1,4,2048,8192]{2,0,3,1} %[[TRANSPOSE1:.*]])

CHECK-DAG: %[[SLICE0:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(bf16[1,4,2048,8192]{3,2,1,0} %[[COPY:.*]]), slice={[0:1], [0:4], [0:2048], [6144:8192]}
CHECK: %[[A2A0:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(bf16[1,4,2048,2048]{3,2,1,0} %[[SLICE0]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[P0:.*]] = bf16[1,8192,32768]{2,1,0} parameter(0)
CHECK-DAG: %[[SLICE4:.*]] = bf16[1,2048,32768]{2,1,0} slice(bf16[1,8192,32768]{2,1,0} %[[P0:.*]]), slice={[0:1], [6144:8192], [0:32768]}
CHECK-DAG: %[[DOT0:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(bf16[1,4,2048,2048]{3,2,1,0} %[[A2A0:.*]], bf16[1,2048,32768]{2,1,0} %[[SLICE4:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"9","wait_on_operation_queues":[],"force_earliest_schedule":false}

CHECK-DAG: %[[SLICE1:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(bf16[1,4,2048,8192]{3,2,1,0} %[[COPY:.*]]), slice={[0:1], [0:4], [0:2048], [4096:6144]}
CHECK: %[[A2A1:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(bf16[1,4,2048,2048]{3,2,1,0} %[[SLICE1]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[SLICE5:.*]] = bf16[1,2048,32768]{2,1,0} slice(bf16[1,8192,32768]{2,1,0} %[[P0:.*]]), slice={[0:1], [4096:6144], [0:32768]}
CHECK-DAG: %[[DOT1:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(bf16[1,4,2048,2048]{3,2,1,0} %[[A2A1:.*]], bf16[1,2048,32768]{2,1,0} %[[SLICE5:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"8","wait_on_operation_queues":[],"force_earliest_schedule":false}

CHECK-DAG: %[[SLICE2:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(bf16[1,4,2048,8192]{3,2,1,0} %[[COPY:.*]]), slice={[0:1], [0:4], [0:2048], [2048:4096]}
CHECK: %[[A2A2:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(bf16[1,4,2048,2048]{3,2,1,0} %[[SLICE2]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[SLICE6:.*]] = bf16[1,2048,32768]{2,1,0} slice(bf16[1,8192,32768]{2,1,0} %[[P0:.*]]), slice={[0:1], [2048:4096], [0:32768]}
CHECK-DAG: %[[DOT2:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(bf16[1,4,2048,2048]{3,2,1,0} %[[A2A2:.*]], bf16[1,2048,32768]{2,1,0} %[[SLICE6:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"7","wait_on_operation_queues":[],"force_earliest_schedule":false}

CHECK-DAG: %[[SLICE3:.*]] = bf16[1,4,2048,2048]{3,2,1,0} slice(bf16[1,4,2048,8192]{3,2,1,0} %[[COPY:.*]]), slice={[0:1], [0:4], [0:2048], [0:2048]}
CHECK: %[[A2A2:.*]] = bf16[1,4,2048,2048]{3,2,1,0} all-to-all(bf16[1,4,2048,2048]{3,2,1,0} %[[SLICE3]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[SLICE7:.*]] = bf16[1,2048,32768]{2,1,0} slice(bf16[1,8192,32768]{2,1,0} %[[P0:.*]]), slice={[0:1], [0:2048], [0:32768]}
CHECK-DAG: %[[DOT3:.*]] = bf16[1,4,2048,32768]{3,2,1,0} dot(bf16[1,4,2048,2048]{3,2,1,0} %[[A2A3:.*]], bf16[1,2048,32768]{2,1,0} %[[SLICE7:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"6","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK-DAG: %[[CONSTANT:.*]] = bf16[] constant(0)
CHECK-DAG: %[[BROADCAST:.*]] = bf16[1,4,2048,32768]{3,2,1,0} broadcast(bf16[] %[[CONSTANT:.*]]), dimensions={}
CHECK-DAG: %[[ADD0:.*]] = bf16[1,4,2048,32768]{3,2,1,0} add(bf16[1,4,2048,32768]{3,2,1,0} %[[DOT0:.*]], bf16[1,4,2048,32768]{3,2,1,0} %[[BROADCAST:.*]]), backend_config={"operation_queue_id":"0","wait_on_operation_queues":["6"],"force_earliest_schedule":false}
CHECK-DAG: %[[ADD1:.*]] = bf16[1,4,2048,32768]{3,2,1,0} add(bf16[1,4,2048,32768]{3,2,1,0} %[[DOT1:.*]], bf16[1,4,2048,32768]{3,2,1,0} %[[ADD0:.*]]), backend_config={"operation_queue_id":"0","wait_on_operation_queues":["7"],"force_earliest_schedule":false}
CHECK-DAG: %[[ADD2:.*]] = bf16[1,4,2048,32768]{3,2,1,0} add(bf16[1,4,2048,32768]{3,2,1,0} %[[DOT2:.*]], bf16[1,4,2048,32768]{3,2,1,0} %[[ADD1:.*]]), backend_config={"operation_queue_id":"0","wait_on_operation_queues":["8"],"force_earliest_schedule":false}

CHECK: ROOT {{.*}} = bf16[1,4,2048,32768]{3,2,1,0} add(bf16[1,4,2048,32768]{3,2,1,0} %[[DOT3:.*]], bf16[1,4,2048,32768]{3,2,1,0} %[[ADD2:.*]])
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  GpuWindowedEinsumHandler gpu_handler;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(module->ToString(), kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(GpuWindowedEinsumHanlderTest, GemmA2aTransposeLoopsHaveStreamIds) {
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

CHECK-DAG: %[[SLICE0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(bf16[1,4,2048,32768]{3,2,1,0} %[[P1]]), slice={[0:1], [0:4], [0:2048], [24576:32768]}
CHECK-DAG: %[[P0:.*]] = bf16[1,32768,8192]{2,1,0} parameter(1)
CHECK-DAG: %[[SLICE4:.*]] = bf16[1,8192,8192]{2,1,0} slice(bf16[1,32768,8192]{2,1,0} %[[P0:.*]]), slice={[0:1], [24576:32768], [0:8192]}
CHECK-DAG: %[[DOT0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(bf16[1,4,2048,8192]{3,2,1,0} %[[SLICE0:.*]], bf16[1,8192,8192]{2,1,0} %[[SLICE4:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"12","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(bf16[1,4,2048,8192]{3,2,1,0} %[[DOT0:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}

CHECK-DAG: %[[SLICE1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(bf16[1,4,2048,32768]{3,2,1,0} %[[P1]]), slice={[0:1], [0:4], [0:2048], [16384:24576]}
CHECK-DAG: %[[SLICE5:.*]] = bf16[1,8192,8192]{2,1,0} slice(bf16[1,32768,8192]{2,1,0} %[[P0:.*]]), slice={[0:1], [16384:24576], [0:8192]}
CHECK-DAG: %[[DOT1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(bf16[1,4,2048,8192]{3,2,1,0} %[[SLICE1:.*]], bf16[1,8192,8192]{2,1,0} %[[SLICE5:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"11","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(bf16[1,4,2048,8192]{3,2,1,0} %[[DOT1:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}

CHECK-DAG: %[[SLICE2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(bf16[1,4,2048,32768]{3,2,1,0} %[[P1]]), slice={[0:1], [0:4], [0:2048], [8192:16384]}
CHECK-DAG: %[[SLICE6:.*]] = bf16[1,8192,8192]{2,1,0} slice(bf16[1,32768,8192]{2,1,0} %[[P0:.*]]), slice={[0:1], [8192:16384], [0:8192]}
CHECK-DAG: %[[DOT2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(bf16[1,4,2048,8192]{3,2,1,0} %[[SLICE2:.*]], bf16[1,8192,8192]{2,1,0} %[[SLICE6:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"10","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(bf16[1,4,2048,8192]{3,2,1,0} %[[DOT2:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}

CHECK-DAG: %[[SLICE3:.*]] = bf16[1,4,2048,8192]{3,2,1,0} slice(bf16[1,4,2048,32768]{3,2,1,0} %[[P1]]), slice={[0:1], [0:4], [0:2048], [0:8192]}
CHECK-DAG: %[[SLICE7:.*]] = bf16[1,8192,8192]{2,1,0} slice(bf16[1,32768,8192]{2,1,0} %[[P0:.*]]), slice={[0:1], [0:8192], [0:8192]}
CHECK-DAG: %[[DOT3:.*]] = bf16[1,4,2048,8192]{3,2,1,0} dot(bf16[1,4,2048,8192]{3,2,1,0} %[[SLICE3:.*]], bf16[1,8192,8192]{2,1,0} %[[SLICE7:.*]]), lhs_batch_dims={0}, lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}, backend_config={"operation_queue_id":"9","wait_on_operation_queues":[],"force_earliest_schedule":false}
CHECK: %[[A2A2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} all-to-all(bf16[1,4,2048,8192]{3,2,1,0} %[[DOT3:.*]]),
CHECK: replica_groups={
CHECK:     {0,1,2,3}
CHECK: }
CHECK: dimensions={1}
CHECK-DAG: %[[CONSTANT:.*]] = bf16[] constant(0)
CHECK-DAG: %[[BROADCAST:.*]] = bf16[1,4,2048,8192]{3,2,1,0} broadcast(bf16[] %[[CONSTANT:.*]]), dimensions={}
CHECK-DAG: %[[ADD0:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(bf16[1,4,2048,8192]{3,2,1,0} %[[A2A0:.*]], bf16[1,4,2048,8192]{3,2,1,0} %[[BROADCAST:.*]])
CHECK-DAG: %[[ADD1:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(bf16[1,4,2048,8192]{3,2,1,0} %[[A2A1:.*]], bf16[1,4,2048,8192]{3,2,1,0} %[[ADD0:.*]])
CHECK-DAG: %[[ADD2:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(bf16[1,4,2048,8192]{3,2,1,0} %[[A2A2:.*]], bf16[1,4,2048,8192]{3,2,1,0} %[[ADD1:.*]])
CHECK-DAG: %[[ADD3:.*]] = bf16[1,4,2048,8192]{3,2,1,0} add(bf16[1,4,2048,8192]{3,2,1,0} %[[A2A3:.*]], bf16[1,4,2048,8192]{3,2,1,0} %[[ADD2:.*]])

CHECK-DAG: %[[COPY:.*]] = bf16[1,4,2048,8192]{3,2,1,0} copy(bf16[1,4,2048,8192]{3,2,1,0} %[[ADD3:.*]])
CHECK-DAG: %[[TRANSPOSE0:.*]] = bf16[4,1,2048,8192]{3,2,0,1} transpose(bf16[1,4,2048,8192]{3,2,1,0} %[[COPY:.*]]), dimensions={1,0,2,3}
CHECK-DAG: %[[COPY1:.*]] = bf16[4,1,2048,8192]{3,2,1,0} copy(bf16[4,1,2048,8192]{3,2,0,1} %[[TRANSPOSE0:.*]])
CHECK-DAG: %[[RESHAPE0:.*]] = bf16[1,4,1,2048,8192]{4,3,2,1,0} reshape(bf16[4,1,2048,8192]{3,2,1,0} %[[COPY1:.*]])

CHECK: ROOT {{.*}} = bf16[1,4,1,1,2048,8192]{5,4,3,2,1,0} reshape(bf16[1,4,1,2048,8192]{4,3,2,1,0} %[[RESHAPE0:.*]])
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  GpuWindowedEinsumHandler gpu_handler;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, gpu_handler.Run(module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(module->ToString(), kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(GpuWindowedEinsumHanlderTest, AllGatherF8) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(f8e4m3fn[2,512,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f32[], f32[])->f32[2,2048,24576]{2,1,0}}, num_partitions=4

windowed_dot_general_body_ag {
  param.1 = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) parameter(0)
  get-tuple-element.1 = f32[2,512,24576]{2,1,0} get-tuple-element(param.1), index=0
  collective-permute = f32[2,512,24576]{2,1,0} collective-permute(get-tuple-element.1), channel_id=4, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  collective-permute.1 = f32[2,512,24576]{2,1,0} collective-permute(collective-permute), channel_id=5, source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
  get-tuple-element.2 = f32[24576,24576]{1,0} get-tuple-element(param.1), index=1
  get-tuple-element.3 = f32[2,2048,24576]{2,1,0} get-tuple-element(param.1), index=2
  dot = f32[2,512,24576]{2,1,0} dot(get-tuple-element.1, get-tuple-element.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  constant.12 = s32[] constant(0)
  constant.13 = s32[4]{0} constant({0, 512, 1024, 1536})
  get-tuple-element.5 = u32[] get-tuple-element(param.1), index=4
  partition-id = u32[] partition-id()
  add = u32[] add(get-tuple-element.5, partition-id)
  constant.11 = u32[] constant(4)
  remainder = u32[] remainder(add, constant.11)
  dynamic-slice = s32[1]{0} dynamic-slice(constant.13, remainder), dynamic_slice_sizes={1}
  reshape = s32[] reshape(dynamic-slice)
  dynamic-update-slice = f32[2,2048,24576]{2,1,0} dynamic-update-slice(get-tuple-element.3, dot, constant.12, reshape, constant.12)
  dot.1 = f32[2,512,24576]{2,1,0} dot(collective-permute, get-tuple-element.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  constant.15 = u32[] constant(1)
  add.1 = u32[] add(get-tuple-element.5, constant.15)
  add.2 = u32[] add(add.1, partition-id)
  remainder.1 = u32[] remainder(add.2, constant.11)
  dynamic-slice.1 = s32[1]{0} dynamic-slice(constant.13, remainder.1), dynamic_slice_sizes={1}
  reshape.1 = s32[] reshape(dynamic-slice.1)
  dynamic-update-slice.1 = f32[2,2048,24576]{2,1,0} dynamic-update-slice(dynamic-update-slice, dot.1, constant.12, reshape.1, constant.12)
  get-tuple-element.4 = f32[2,2048,24576]{2,1,0} get-tuple-element(param.1), index=3
  add.3 = u32[] add(add.1, constant.15)
  ROOT tuple = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) tuple(collective-permute.1, get-tuple-element.2, dynamic-update-slice.1, get-tuple-element.4, add.3)
} // windowed_dot_general_body_ag

windowed_dot_general_cond_ag {
  param = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) parameter(0)
  get-tuple-element = u32[] get-tuple-element(param), index=4
  constant.10 = u32[] constant(4)
  ROOT compare = pred[] compare(get-tuple-element, constant.10), direction=LT
}

ENTRY test_main {
  param.4 = f8e4m3fn[2,512,24576]{2,1,0} parameter(0), sharding={devices=[1,4,1]<=[4]}
  reshape.8 = f8e4m3fn[2,512,24576]{2,1,0} reshape(param.4)
  param.5 = f8e4m3fn[24576,24576]{1,0} parameter(1), sharding={devices=[1,4]<=[4]}
  constant.18 = f32[] constant(0)
  broadcast = f32[2,2048,24576]{2,1,0} broadcast(constant.18), dimensions={}
  constant.20 = u32[] constant(0)
  scale_lhs = f32[] parameter(2)
  scale_lhs_bcast = f32[2,512,24576]{2,1,0} broadcast(scale_lhs), dimensions={}
  lhs_bf32 = f32[2,512,24576]{2,1,0} convert(reshape.8)  
  lhs_scaled = f32[2,512,24576]{2,1,0} multiply(lhs_bf32, scale_lhs_bcast)
  scale_rhs = f32[] parameter(3)
  scale_rhs_bcast = f32[24576,24576]{1,0} broadcast(scale_rhs), dimensions={}
  rhs_bf32 = f32[24576,24576]{1,0} convert(param.5)  
  rhs_scaled = f32[24576,24576]{1,0} multiply(rhs_bf32, scale_rhs_bcast)
  tuple.2 = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) tuple(lhs_scaled, rhs_scaled, broadcast, broadcast, constant.20)
  while = (f32[2,512,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[]) while(tuple.2), condition=windowed_dot_general_cond_ag, body=windowed_dot_general_body_ag
  ROOT get-tuple-element.13 = f32[2,2048,24576]{2,1,0} get-tuple-element(while), index=2
}
)";

  RunAndFilecheckHloRewrite(kHloString, GpuWindowedEinsumHandler(),
                            R"(
; CHECK-LABEL: windowed_dot_general_body_ag
; CHECK-NEXT:    [[P0:%[^ ]+]] = (f8e4m3fn[2,512,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[], /*index=5*/f32[], f32[]) parameter(0)
; CHECK-NEXT:    [[GTE0:%[^ ]+]] = f8e4m3fn[2,512,24576]{2,1,0} get-tuple-element([[P0]]), index=0
; CHECK-NEXT:    [[CP0:%[^ ]+]] = f8e4m3fn[2,512,24576]{2,1,0} collective-permute([[GTE0]]), channel_id=4
; CHECK-NEXT:    [[CP1:%[^ ]+]] = f8e4m3fn[2,512,24576]{2,1,0} collective-permute([[CP0]]), channel_id=5
; CHECK-NEXT:    [[GTE1:%[^ ]+]] = f8e4m3fn[24576,24576]{1,0} get-tuple-element([[P0]]), index=1
; CHECK-NEXT:    [[GTE2:%[^ ]+]] = f32[2,2048,24576]{2,1,0} get-tuple-element([[P0]]), index=2
; CHECK-NEXT:    [[CONVERT0:%[^ ]+]] = f32[2,512,24576]{2,1,0} convert([[GTE0]])
; CHECK-NEXT:    [[GTE3:%[^ ]+]] = f32[] get-tuple-element([[P0]]), index=5
; CHECK-NEXT:    [[BCAST0:%[^ ]+]] = f32[2,512,24576]{2,1,0} broadcast([[GTE3]]), dimensions={}
; CHECK-NEXT:    [[MUL0:%[^ ]+]] = f32[2,512,24576]{2,1,0} multiply([[CONVERT0]], [[BCAST0]])
; CHECK-NEXT:    [[CONVERT1:%[^ ]+]] = f32[24576,24576]{1,0} convert([[GTE1]])
; CHECK-NEXT:    [[GTE4:%[^ ]+]] = f32[] get-tuple-element([[P0]]), index=6
; CHECK-NEXT:    [[BCAST1:%[^ ]+]] = f32[24576,24576]{1,0} broadcast([[GTE4]]), dimensions={}
; CHECK-NEXT:    [[MUL1:%[^ ]+]] = f32[24576,24576]{1,0} multiply([[CONVERT1]], [[BCAST1]])
; CHECK-NEXT:    [[DOT0:%[^ ]+]] = f32[2,512,24576]{2,1,0} dot([[MUL0]], [[MUL1]]),
; CHECK-DAG:       lhs_contracting_dims={2},
; CHECK-DAG:       rhs_contracting_dims={0},
; CHECK-DAG:       backend_config={
; CHECK-DAG:         "operation_queue_id":"[[OPQUEUEID:[0-9]+]]",
; CHECK-DAG:         "wait_on_operation_queues":[],
; CHECK-DAG:         "force_earliest_schedule":true}
; CHECK-NEXT:    [[C0:%[^ ]+]] = s32[] constant(0)
; CHECK-NEXT:    [[C1:%[^ ]+]] = s32[4]{0} constant({0, 512, 1024, 1536})
; CHECK-NEXT:    [[GTE5:%[^ ]+]] = u32[] get-tuple-element([[P0]]), index=4
; CHECK-NEXT:    [[PID:%[^ ]+]] = u32[] partition-id()
; CHECK-NEXT:    [[ADD0:%[^ ]+]] = u32[] add([[GTE5]], [[PID]])
; CHECK-NEXT:    [[C2:%[^ ]+]] = u32[] constant(4)
; CHECK-NEXT:    [[REM0:%[^ ]+]] = u32[] remainder([[ADD0]], [[C2]])
; CHECK-NEXT:    [[DSLICE0:%[^ ]+]] = s32[1]{0} dynamic-slice([[C1]], [[REM0]]), dynamic_slice_sizes={1}
; CHECK-NEXT:    [[RESHAPE0:%[^ ]+]] = s32[] reshape([[DSLICE0]])
; CHECK-NEXT:    [[DUPDATESLICE0:%[^ ]+]] = f32[2,2048,24576]{2,1,0} dynamic-update-slice([[GTE2]], [[DOT0]], [[C0]], [[RESHAPE0]], [[C0]]),
; CHECK-DAG:       backend_config={
; CHECK-DAG:         "operation_queue_id":"0",
; CHECK-DAG:         "wait_on_operation_queues":["[[OPQUEUEID]]"],
; CHECK-DAG:         "force_earliest_schedule":false}
; CHECK-NEXT:    [[CONVERT2:%[^ ]+]] = f32[2,512,24576]{2,1,0} convert([[CP0]])
; CHECK-NEXT:    [[MUL2:%[^ ]+]] = f32[2,512,24576]{2,1,0} multiply([[CONVERT2]], [[BCAST0]])
; CHECK-NEXT:    [[DOT1:%[^ ]+]] = f32[2,512,24576]{2,1,0} dot([[MUL2]], [[MUL1]]),
; CHECK-DAG:       lhs_contracting_dims={2},
; CHECK-DAG:       rhs_contracting_dims={0}
; CHECK-NEXT:    [[C3:%[^ ]+]] = u32[] constant(1)
; CHECK-NEXT:    [[ADD1:%[^ ]+]] = u32[] add([[GTE5]], [[C3]])
; CHECK-NEXT:    [[ADD2:%[^ ]+]] = u32[] add([[ADD1]], [[PID]])
; CHECK-NEXT:    [[REM1:%[^ ]+]] = u32[] remainder([[ADD2]], [[C2]])
; CHECK-NEXT:    [[DSLICE1:%[^ ]+]] = s32[1]{0} dynamic-slice([[C1]], [[REM1]]), dynamic_slice_sizes={1}
; CHECK-NEXT:    [[RESHAPE1:%[^ ]+]] = s32[] reshape([[DSLICE1]])
; CHECK-NEXT:    [[DUPDATESLICE1:%[^ ]+]] = f32[2,2048,24576]{2,1,0} dynamic-update-slice([[DUPDATESLICE0]], [[DOT1]], [[C0]], [[RESHAPE1]], [[C0]])
; CHECK-NEXT:    [[GTE6:%[^ ]+]] = f32[2,2048,24576]{2,1,0} get-tuple-element([[P0]]), index=3
; CHECK-NEXT:    [[ADD3:%[^ ]+]] = u32[] add([[ADD1]], [[C3]])
; CHECK-NEXT: ROOT [[OUT:%[^ ]+]] = (f8e4m3fn[2,512,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f32[2,2048,24576]{2,1,0}, f32[2,2048,24576]{2,1,0}, u32[], /*index=5*/f32[], f32[]) tuple([[CP1]], [[GTE1]], [[DUPDATESLICE1]], [[GTE6]], [[ADD3]], /*index=5*/[[GTE3]], [[GTE4]])
)");
}

TEST_F(GpuWindowedEinsumHanlderTest, ReduceScatterF8) {
  constexpr absl::string_view kHloString = R"(
HloModule pjit__unnamed_wrapped_function_, entry_computation_layout={(f8e4m3fn[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f8e4m3fn[2,2048,24576]{2,1,0}, f32[], f32[])->f32[2,512,24576]{2,1,0}}, num_partitions=4

windowed_dot_general_body_rs {
  param.3 = (f32[2,2048,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f32[2,512,24576]{2,1,0}, u32[]) parameter(0)
  get-tuple-element.7 = f32[2,2048,24576]{2,1,0} get-tuple-element(param.3), index=0
  get-tuple-element.8 = f32[24576,24576]{1,0} get-tuple-element(param.3), index=1
  get-tuple-element.9 = f32[2,512,24576]{2,1,0} get-tuple-element(param.3), index=2
  collective-permute.2 = f32[2,512,24576]{2,1,0} collective-permute(get-tuple-element.9), channel_id=9, source_target_pairs={{0,2},{1,3},{2,0},{3,1}}
  constant.23 = s32[] constant(0)
  constant.24 = s32[4]{0} constant({0, 512, 1024, 1536})
  get-tuple-element.11 = u32[] get-tuple-element(param.3), index=4
  constant.26 = u32[] constant(2)
  add.8 = u32[] add(get-tuple-element.11, constant.26)
  constant.27 = u32[] constant(1)
  add.9 = u32[] add(add.8, constant.27)
  partition-id.3 = u32[] partition-id()
  add.10 = u32[] add(add.9, partition-id.3)
  constant.22 = u32[] constant(4)
  remainder.3 = u32[] remainder(add.10, constant.22)
  dynamic-slice.4 = s32[1]{0} dynamic-slice(constant.24, remainder.3), dynamic_slice_sizes={1}
  reshape.3 = s32[] reshape(dynamic-slice.4)
  dynamic-slice.5 = f32[2,512,24576]{2,1,0} dynamic-slice(get-tuple-element.7, constant.23, reshape.3, constant.23), dynamic_slice_sizes={2,512,24576}
  dot.3 = f32[2,512,24576]{2,1,0} dot(dynamic-slice.5, get-tuple-element.8), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  add.11 = f32[2,512,24576]{2,1,0} add(collective-permute.2, dot.3)
  get-tuple-element.10 = f32[2,512,24576]{2,1,0} get-tuple-element(param.3), index=3
  add.6 = u32[] add(get-tuple-element.11, partition-id.3)
  remainder.2 = u32[] remainder(add.6, constant.22)
  dynamic-slice.2 = s32[1]{0} dynamic-slice(constant.24, remainder.2), dynamic_slice_sizes={1}
  reshape.2 = s32[] reshape(dynamic-slice.2)
  dynamic-slice.3 = f32[2,512,24576]{2,1,0} dynamic-slice(get-tuple-element.7, constant.23, reshape.2, constant.23), dynamic_slice_sizes={2,512,24576}
  dot.2 = f32[2,512,24576]{2,1,0} dot(dynamic-slice.3, get-tuple-element.8), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  add.7 = f32[2,512,24576]{2,1,0} add(get-tuple-element.10, dot.2)
  collective-permute.3 = f32[2,512,24576]{2,1,0} collective-permute(add.7), channel_id=10, source_target_pairs={{0,2},{1,3},{2,0},{3,1}}
  ROOT tuple.1 = (f32[2,2048,24576]{2,1,0}, f32[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f32[2,512,24576]{2,1,0}, u32[]) tuple(get-tuple-element.7, get-tuple-element.8, add.11, collective-permute.3, add.8)
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

  RunAndFilecheckHloRewrite(kHloString, GpuWindowedEinsumHandler(),
                            R"(
; CHECK-LABEL: windowed_dot_general_body_rs
; CHECK-NEXT:    [[P0:%[^ ]+]] = (f8e4m3fn[2,2048,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f32[2,512,24576]{2,1,0}, u32[], /*index=5*/f32[], f32[]) parameter(0)
; CHECK-NEXT:    [[GTE0:%[^ ]+]] = f8e4m3fn[2,2048,24576]{2,1,0} get-tuple-element([[P0]]), index=0
; CHECK-NEXT:    [[GTE1:%[^ ]+]] = f8e4m3fn[24576,24576]{1,0} get-tuple-element([[P0]]), index=1
; CHECK-NEXT:    [[GTE2:%[^ ]+]] = f32[2,512,24576]{2,1,0} get-tuple-element([[P0]]), index=2
; CHECK-NEXT:    [[CP0:%[^ ]+]] = f32[2,512,24576]{2,1,0} collective-permute([[GTE2]]), channel_id=9
; CHECK-NEXT:    [[CONVERT0:%[^ ]+]] = f32[2,2048,24576]{2,1,0} convert([[GTE0]])
; CHECK-NEXT:    [[GTE3:%[^ ]+]] = f32[] get-tuple-element([[P0]]), index=5
; CHECK-NEXT:    [[BCAST0:%[^ ]+]] = f32[2,2048,24576]{2,1,0} broadcast([[GTE3]]), dimensions={}
; CHECK-NEXT:    [[MUL0:%[^ ]+]] = f32[2,2048,24576]{2,1,0} multiply([[CONVERT0]], [[BCAST0]])
; CHECK-NEXT:    [[C0:%[^ ]+]] = s32[] constant(0)
; CHECK-NEXT:    [[C1:%[^ ]+]] = s32[4]{0} constant({0, 512, 1024, 1536})
; CHECK-NEXT:    [[GTE4:%[^ ]+]] = u32[] get-tuple-element([[P0]]), index=4
; CHECK-NEXT:    [[C2:%[^ ]+]] = u32[] constant(2)
; CHECK-NEXT:    [[ADD0:%[^ ]+]] = u32[] add([[GTE4]], [[C2]])
; CHECK-NEXT:    [[C3:%[^ ]+]] = u32[] constant(1)
; CHECK-NEXT:    [[ADD1:%[^ ]+]] = u32[] add([[ADD0]], [[C3]])
; CHECK-NEXT:    [[PID:%[^ ]+]] = u32[] partition-id()
; CHECK-NEXT:    [[ADD2:%[^ ]+]] = u32[] add([[ADD1]], [[PID]])
; CHECK-NEXT:    [[C4:%[^ ]+]] = u32[] constant(4)
; CHECK-NEXT:    [[REM0:%[^ ]+]] = u32[] remainder([[ADD2]], [[C4]])
; CHECK-NEXT:    [[DSLICE0:%[^ ]+]] = s32[1]{0} dynamic-slice([[C1]], [[REM0]]), dynamic_slice_sizes={1}
; CHECK-NEXT:    [[RESHAPE0:%[^ ]+]] = s32[] reshape([[DSLICE0]])
; CHECK-NEXT:    [[DSLICE1:%[^ ]+]] = f32[2,512,24576]{2,1,0} dynamic-slice([[MUL0]], [[C0]], [[RESHAPE0]], [[C0]]), dynamic_slice_sizes={2,512,24576}
; CHECK-NEXT:    [[CONVERT1:%[^ ]+]] = f32[24576,24576]{1,0} convert([[GTE1]])
; CHECK-NEXT:    [[GTE5:%[^ ]+]] = f32[] get-tuple-element([[P0]]), index=6
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
; CHECK-NEXT:    [[GTE6:[^ ]+]] = f32[2,512,24576]{2,1,0} get-tuple-element([[P0]]), index=3
; CHECK-NEXT:    [[ADD4:%[^ ]+]] = u32[] add([[GTE4]], [[PID]])
; CHECK-NEXT:    [[REM1:%[^ ]+]] = u32[] remainder([[ADD4]], [[C4]])
; CHECK-NEXT:    [[DSLICE2:%[^ ]+]] = s32[1]{0} dynamic-slice([[C1]], [[REM1]]), dynamic_slice_sizes={1}
; CHECK-NEXT:    [[RESHAPE1:%[^ ]+]] = s32[] reshape([[DSLICE2]])
; CHECK-NEXT:    [[DSLICE3:%[^ ]+]] = f32[2,512,24576]{2,1,0} dynamic-slice([[MUL0]], [[C0]], [[RESHAPE1]], [[C0]]), dynamic_slice_sizes={2,512,24576}
; CHECK-NEXT:    [[DOT1:%[^ ]+]] = f32[2,512,24576]{2,1,0} dot([[DSLICE3]], [[MUL1]]),
; CHECK-DAG:       lhs_contracting_dims={2},
; CHECK-DAG:       rhs_contracting_dims={0}
; CHECK-NEXT:    [[ADD5:%[^ ]+]] = f32[2,512,24576]{2,1,0} add([[GTE6]], [[DOT1]])
; CHECK-NEXT:    [[CP1:[^ ]+]] = f32[2,512,24576]{2,1,0} collective-permute([[ADD5]]), channel_id=10
; CHECK-NEXT:  ROOT [[OUT:[^ ]+]] = (f8e4m3fn[2,2048,24576]{2,1,0}, f8e4m3fn[24576,24576]{1,0}, f32[2,512,24576]{2,1,0}, f32[2,512,24576]{2,1,0}, u32[], /*index=5*/f32[], f32[]) tuple([[GTE0]], [[GTE1]], [[ADD3]], [[CP1]], [[ADD0]], /*index=5*/[[GTE3]], [[GTE5]])
)");
}

}  // namespace
}  // namespace xla::gpu
