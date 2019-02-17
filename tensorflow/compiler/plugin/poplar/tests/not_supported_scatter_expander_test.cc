/* Copyright 2019 Graphcore. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/not_supported_scatter_expander.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using NotSupportedScatterExpanderTest = HloTestBase;

TEST_F(NotSupportedScatterExpanderTest, ExpandNotSupportedScatter) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_text, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  NotSupportedScatterExpander nsse;
  EXPECT_TRUE(nsse.Run(module).ValueOrDie());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kWhile);
}

TEST_F(NotSupportedScatterExpanderTest, ExpandNotSupportedScatterWithSharding) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[2,3] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1, sharding={maximal device=0}
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_text, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  NotSupportedScatterExpander nsse;
  EXPECT_TRUE(nsse.Run(module).ValueOrDie());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_TRUE(root->has_sharding());
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kWhile);
  HloInstruction* while_inst = root->mutable_operand(0);
  EXPECT_TRUE(while_inst->has_sharding());
  for (auto* inst : while_inst->while_condition()->MakeInstructionPostOrder()) {
    EXPECT_TRUE(inst->has_sharding());
  }
  for (auto* inst : while_inst->while_body()->MakeInstructionPostOrder()) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

TEST_F(NotSupportedScatterExpanderTest,
       ExpandNotSupportedScatterZeroSizedUpdates) {
  const string hlo_text = R"(
HloModule TensorFlowScatterV1

update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT add = s32[] add(lhs, rhs)
}

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  updates = s32[0] parameter(2)
  ROOT scatter = s32[3,3] scatter(operand, indices, updates),
      to_apply=update_s32,
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1, sharding={maximal device=0}
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_text, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  NotSupportedScatterExpander nsse;
  EXPECT_TRUE(nsse.Run(module).ValueOrDie());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kParameter);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
