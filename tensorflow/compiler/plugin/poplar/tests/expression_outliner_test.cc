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

#include "tensorflow/compiler/plugin/poplar/driver/passes/expression_outliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/tests/test_utils.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ExpressionOutlinerTest = HloTestBase;

// General extraction
//
//  i i  i  i
//  \ /  \ /
//   a    b
//    \  /
//     c
//     |
TEST_F(ExpressionOutlinerTest, OutlineSimpleTree) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] parameter(2)
  a3 = f16[] parameter(3)
  add1 = f16[] add(a0, a1)
  sub1 = f16[] subtract(a2, a3)
  mul1 = f16[] multiply(add1, sub1), sharding={maximal device=1}
  ROOT %tuple = (f16[]) tuple(mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ExpressionOutliner eo(annotations);
  EXPECT_TRUE(eo.Run(module).ValueOrDie());

  auto* comp = module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 6);
  ASSERT_THAT(inst->operand_count(), 1);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 4);
  ASSERT_TRUE(inst->operand(0)->has_sharding());
  EXPECT_THAT(inst->operand(0)->sharding().UniqueDevice(), 1);
}

// Shared inputs to outlined section (a+b+c)
//
//  i   i   i
//  \  / \ /
//   a    b
//    \  /
//     c
//     |
TEST_F(ExpressionOutlinerTest, OutlineTreeWithSharedInputs) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] parameter(2)
  add1 = f16[] add(a0, a1)
  sub1 = f16[] subtract(a0, a2)
  mul1 = f16[] multiply(add1, sub1), sharding={maximal device=1}
  ROOT %tuple = (f16[]) tuple(mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ExpressionOutliner eo(annotations);
  EXPECT_TRUE(eo.Run(module).ValueOrDie());

  auto* comp = module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 5);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 3);
}

// Don't outline a single operation
TEST_F(ExpressionOutlinerTest, DontOutlineSingleOps) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  add1 = f16[] add(a0, a1)
  ROOT %tuple = (f16[]) tuple(add1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ExpressionOutliner eo(annotations);
  EXPECT_TRUE(eo.Run(module).ValueOrDie());

  auto* comp = module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 4);
  ASSERT_THAT(inst->operand_count(), 1);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kAdd);
}

// Test correct tree outlining in a DAG (either a or b are not outlined)
//
//  i i i i
//  \ / \ /
//   a   b
//  / \ /
// c  d
// \  /
//  e
//  |
TEST_F(ExpressionOutlinerTest, OutlineTreeInDAG) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] parameter(2)
  a3 = f16[] parameter(3)
  add1 = f16[] add(a0, a1)
  add2 = f16[] add(a2, a3)
  sin1 = f16[] sine(add1)
  sub1 = f16[] subtract(add1, add2)
  mul1 = f16[] multiply(sin1, sub1), sharding={maximal device=1}
  ROOT %tuple = (f16[]) tuple(mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ExpressionOutliner eo(annotations);
  EXPECT_TRUE(eo.Run(module).ValueOrDie());

  auto* comp = module->entry_computation();
  auto* inst = comp->root_instruction();
  auto* add1 = comp->GetInstructionWithName("add1");
  auto* add2 = comp->GetInstructionWithName("add2");

  EXPECT_THAT(comp->instruction_count(), 7);
  ASSERT_THAT(inst->operand_count(), 1);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 3);
  EXPECT_TRUE(HasOperandIn(inst->operand(0), {add1, add2}));
}

// Don't outline op 'b' where 'X' is not part of the outline (only outline c+d)
//
//    i i
//    \ /
//     a
//    / \
//   b  c
//  / \ /
// X   d
// |   |
TEST_F(ExpressionOutlinerTest, DontOutlineOpsWithOutputsOutsideOfTheSubgraph) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  add1 = f16[] add(a0, a1)
  sin1 = f16[] sine(add1)
  cos1 = f16[] cosine(add1)
  mul1 = f16[] multiply(sin1, cos1)
  ROOT %tuple = (f16[], f16[]) tuple(mul1, cos1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ExpressionOutliner eo(annotations);
  EXPECT_TRUE(eo.Run(module).ValueOrDie());

  auto* comp = module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 6);
  ASSERT_THAT(inst->operand_count(), 2);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 2);
  EXPECT_THAT(inst->operand(1)->opcode(), HloOpcode::kCos);
}

// Do two independent outlines
TEST_F(ExpressionOutlinerTest, OutlineTwoSubgraphs) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  mul1 = f16[] multiply(a0, a1)
  sub1 = f16[] subtract(a0, a1)
  sin1 = f16[] sine(mul1)
  cos1 = f16[] cosine(sub1)
  ROOT %tuple = (f16[], f16[]) tuple(sin1, cos1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ExpressionOutliner eo(annotations);
  EXPECT_TRUE(eo.Run(module).ValueOrDie());

  auto* comp = module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 5);
  ASSERT_THAT(inst->operand_count(), 2);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 2);
  EXPECT_THAT(inst->operand(1)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(1)->operand_count(), 2);
}

// Two independent networks outlined separately
//
//  i i  i i
//  | |  | |
//  a b  c d
//  \ /  \ /
//   e    f
//   --\/--
//     o
TEST_F(ExpressionOutlinerTest, OutlineTwoExpressions) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] parameter(2)
  a3 = f16[] parameter(3)
  sin1 = f16[] sine(a0)
  sin2 = f16[] sine(a1)
  sin3 = f16[] sine(a2)
  sin4 = f16[] sine(a3)
  sub1 = f16[] subtract(sin1, sin2)
  sub2 = f16[] subtract(sin3, sin4)
  ROOT %tuple = (f16[], f16[]) tuple(sub1, sub2)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ExpressionOutliner eo(annotations);
  EXPECT_TRUE(eo.Run(module).ValueOrDie());

  auto* comp = module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 7);
  ASSERT_THAT(inst->operand_count(), 2);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 2);
  EXPECT_THAT(inst->operand(1)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(1)->operand_count(), 2);
}

// Don't outline expressions on different shards
//
//  i i  i  i
//  | |  |  |
//  a b  c* d*
//  \ /  \ /
//   e    f*
//   --\/--
//     g
//     |
TEST_F(ExpressionOutlinerTest, DontOutlineDifferentShardsTogether) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = f16[] parameter(0)
  a1 = f16[] parameter(1)
  a2 = f16[] parameter(2)
  a3 = f16[] parameter(3)
  sin1 = f16[] sine(a0), sharding={maximal device=0}
  sin2 = f16[] sine(a1), sharding={maximal device=0}
  sin3 = f16[] sine(a2), sharding={maximal device=1}
  sin4 = f16[] sine(a3), sharding={maximal device=1}
  sub1 = f16[] subtract(sin1, sin2), sharding={maximal device=0}
  sub2 = f16[] subtract(sin3, sin4), sharding={maximal device=1}
  mul1 = f16[] multiply(sub1, sub2), sharding={maximal device=0}
  ROOT %tuple = (f16[]) tuple(mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  CompilerAnnotations annotations(module);
  ExpressionOutliner eo(annotations);
  EXPECT_TRUE(eo.Run(module).ValueOrDie());

  auto* comp = module->entry_computation();
  auto* inst = comp->root_instruction();

  EXPECT_THAT(comp->instruction_count(), 7);
  ASSERT_THAT(inst->operand_count(), 1);
  EXPECT_THAT(inst->operand(0)->opcode(), HloOpcode::kCall);
  EXPECT_THAT(inst->operand(0)->operand_count(), 3);
  EXPECT_THAT(inst->operand(0)->to_apply()->instruction_count(), 7);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
