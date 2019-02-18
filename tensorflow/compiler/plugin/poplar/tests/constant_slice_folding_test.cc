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

#include "tensorflow/compiler/plugin/poplar/driver/passes/constant_slice_folding.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ConstantSliceFoldingTest = HloTestBase;

TEST_F(ConstantSliceFoldingTest, TestSliceFolding) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  arg0 = f32[4] constant({0.0, 1.0, 2.0, 3.0})
  s = f32[1] slice(arg0), slice={[1:2]}
  ROOT t = (f32[]) tuple(s)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ConstantSliceFolding shardingPass;
  EXPECT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kConstant);

  float const_val = root->operand(0)->literal().GetFirstElement<float>();
  EXPECT_EQ(const_val, 1.0f);
}

TEST_F(ConstantSliceFoldingTest, TestSliceAndReshapeFolding) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  arg0 = f32[4] constant({0.0, 1.0, 2.0, 3.0})
  s = f32[1] slice(arg0), slice={[2:3]}
  r = f32[1] reshape(s)
  ROOT t = (f32[]) tuple(r)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ConstantSliceFolding shardingPass;
  while (shardingPass.Run(module).ValueOrDie())
    ;

  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kConstant);

  float const_val = root->operand(0)->literal().GetFirstElement<float>();
  EXPECT_EQ(const_val, 2.0f);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
