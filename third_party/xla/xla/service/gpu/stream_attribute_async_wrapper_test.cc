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

#include "xla/service/gpu/stream_attribute_async_wrapper.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using StreamAttributeAsyncWrapperTest = HloTestBase;

TEST_F(StreamAttributeAsyncWrapperTest, NonDefaultOpIsWrapped) {
  constexpr absl::string_view kHloString = R"(
  HloModule ModuleWithAsync

  ENTRY entry {
    p1_32 = f32[1] parameter(0)
    p2_32 = f32[1] parameter(1)
    add_32 = f32[1] add(p1_32, p2_32), backend_config={"operation_queue_id":"1", "wait_on_operation_queues":[], "force_earliest_schedule":true}
    ROOT exp_32 = f32[1] exponential(add_32), backend_config={"operation_queue_id":"0", "wait_on_operation_queues":[1]}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));

  StreamAttributeAsyncWrapper async_wrapper;
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, async_wrapper.Run(module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* producer =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(producer->opcode(), HloOpcode::kAsyncDone);
  // Verify that the force_earliest_schedule is set to false for the done op.
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig done_gpu_config,
                          producer->backend_config<GpuBackendConfig>());
  EXPECT_EQ(done_gpu_config.force_earliest_schedule(), false);

  const HloInstruction* producer_start = producer->operand(0);
  EXPECT_EQ(producer_start->opcode(), HloOpcode::kAsyncStart);

  const xla::HloAsyncInstruction* async =
      Cast<HloAsyncInstruction>(producer_start);
  EXPECT_EQ(async->async_wrapped_opcode(), HloOpcode::kAdd);
  // Verify that the backend config is kept intact
  TF_ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                          async->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.operation_queue_id(), 1);
  EXPECT_EQ(gpu_config.force_earliest_schedule(), true);
  EXPECT_EQ(async->async_execution_thread(), "parallel");
}
}  // namespace
}  // namespace xla::gpu
