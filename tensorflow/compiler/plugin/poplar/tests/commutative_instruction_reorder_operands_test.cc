/* Copyright 2018 Graphcore. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/commutative_instruction_reorder_operands.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using CommutativeInstructionReorderOperandsTest = HloTestBase;

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderUnary) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[] parameter(0)
  i2 = f16[2, 2] parameter(1)
  b1 = f16[2, 2] broadcast(i1), dimensions={0, 1}
  ROOT a1 = f16[2, 2] add(b1, i2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  CommutativeInstructionReorderOperands ciro;
  EXPECT_TRUE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kBroadcast);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderWithAddDependency1) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[] parameter(0)
  aa = token[] after-all()
  i2 = f16[2, 2] parameter(1)
  ad = f16[2, 2] add-dependency(i2, aa)
  b1 = f16[2, 2] broadcast(i1), dimensions={0, 1}
  ROOT a1 = f16[2, 2] add(b1, ad)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  CommutativeInstructionReorderOperands ciro;
  EXPECT_TRUE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kAddDependency);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kBroadcast);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderWithAddDependency2) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[] parameter(0)
  aa = token[] after-all()
  i2 = f16[2, 2] parameter(1)
  b1 = f16[2, 2] broadcast(i1), dimensions={0, 1}
  ad = f16[2, 2] add-dependency(b1, aa)
  ROOT a1 = f16[2, 2] add(ad, i2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  CommutativeInstructionReorderOperands ciro;
  EXPECT_TRUE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kAddDependency);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, DontReorderUnaryElementwise) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[2, 2] parameter(0)
  i2 = f16[2, 2] parameter(1)
  e1 = f16[2, 2] exponential(i1)
  ROOT a1 = f16[2, 2] add(e1, i2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  CommutativeInstructionReorderOperands ciro;
  EXPECT_FALSE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kExp);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kParameter);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, ReorderBinary) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  zero = f16[] constant(0)
  i1 = f16[2, 1] parameter(0)
  i2 = f16[2, 2] parameter(1)
  p1 = f16[2, 2] pad(i1, zero), padding=0_0x0_1
  ROOT a1 = f16[2, 2] multiply(p1, i2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  CommutativeInstructionReorderOperands ciro;
  EXPECT_TRUE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kParameter);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kPad);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, DontReorderBinary) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[2, 2] parameter(0)
  i2 = f16[2, 2] parameter(1)
  i3 = f16[2, 2] parameter(2)
  a1 = f16[2, 2] add(i1, i2)
  ROOT m1 = f16[2, 2] multiply(a1, i3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  CommutativeInstructionReorderOperands ciro;
  EXPECT_FALSE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kAdd);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kParameter);
  }
}

TEST_F(CommutativeInstructionReorderOperandsTest, DontReorderBothReshaping) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  i1 = f16[] parameter(0)
  i2 = f16[] parameter(1)
  b1 = f16[2, 2] broadcast(i1), dimensions={0, 1}
  b2 = f16[2, 2] broadcast(i2), dimensions={0, 1}
  ROOT a1 = f16[2, 2] add(b1, b2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());
  auto* module = module_or_status.ValueOrDie().get();
  auto* comp = module->entry_computation();

  auto* b1 = comp->GetInstructionWithName("b1");
  auto* b2 = comp->GetInstructionWithName("b2");

  CommutativeInstructionReorderOperands ciro;
  EXPECT_FALSE(ciro.Run(module).ValueOrDie());

  {
    const auto* root_inst = comp->root_instruction();
    EXPECT_THAT(root_inst->operand(0)->opcode(), HloOpcode::kBroadcast);
    EXPECT_THAT(root_inst->operand(1)->opcode(), HloOpcode::kBroadcast);
    EXPECT_THAT(root_inst->operand(0), b1);
    EXPECT_THAT(root_inst->operand(1), b2);
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
