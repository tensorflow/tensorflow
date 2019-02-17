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

#include "tensorflow/compiler/plugin/poplar/driver/passes/not_supported_gather_expander.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using NotSupportedGatherExpanderTest = HloTestBase;

TEST_F(NotSupportedGatherExpanderTest, ExpandNotSupportedGather) {
  const string hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3,2] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3, 1}
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_text, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  NotSupportedGatherExpander nsse;
  EXPECT_TRUE(nsse.Run(module).ValueOrDie());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTranspose);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kReshape);
  auto* reshape = root->operand(0);
  EXPECT_EQ(reshape->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(reshape->operand(0)->tuple_index(), 3);
  auto* gte = reshape->operand(0);
  EXPECT_EQ(gte->operand(0)->opcode(), HloOpcode::kWhile);
}

TEST_F(NotSupportedGatherExpanderTest, ExpandNotSupportedGatherZeroSized) {
  const string hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3,0] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3, 1}
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_text, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  NotSupportedGatherExpander nsse;
  EXPECT_FALSE(nsse.Run(module).ValueOrDie());
}

TEST_F(NotSupportedGatherExpanderTest, ExpandNotSupportedGatherWithSharding) {
  const string hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3,2] gather(operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3, 1}, sharding={maximal device=0}
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_text, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();

  NotSupportedGatherExpander nsse;
  EXPECT_TRUE(nsse.Run(module).ValueOrDie());

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTranspose);
  EXPECT_TRUE(root->has_sharding());
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kReshape);
  auto* reshape = root->operand(0);
  EXPECT_TRUE(reshape->has_sharding());
  EXPECT_EQ(reshape->operand(0)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(reshape->operand(0)->tuple_index(), 3);
  auto* gte = reshape->operand(0);
  EXPECT_TRUE(gte->has_sharding());
  EXPECT_EQ(gte->operand(0)->opcode(), HloOpcode::kWhile);
  auto* while_inst = gte->operand(0);
  EXPECT_TRUE(while_inst->has_sharding());
  for (auto* inst : while_inst->while_condition()->MakeInstructionPostOrder()) {
    EXPECT_TRUE(inst->has_sharding());
  }
  for (auto* inst : while_inst->while_body()->MakeInstructionPostOrder()) {
    EXPECT_TRUE(inst->has_sharding());
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
