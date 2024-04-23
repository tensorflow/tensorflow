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
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

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

}  // namespace
}  // namespace xla::gpu
