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

#include "tensorflow/compiler/plugin/poplar/driver/passes/sharding_pass.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using ShardingPassTest = HloTestBase;

TEST_F(ShardingPassTest, TestNoSharding) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  sin0 = f16[4] sine(arg0)
  mul0 = f16[4] multiply(sin0, arg1)
  mul1 = f16[4] multiply(mul0, arg2)
  ROOT %tuple = (f16[4], f16[4]) tuple(mul0, mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  EXPECT_FALSE(shardingPass.Run(module).ValueOrDie());

  auto insts = module->entry_computation()->instructions();
  for (auto* inst : insts) {
    EXPECT_FALSE(inst->has_sharding());
  }
}

TEST_F(ShardingPassTest, TestAddShardingSimple) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  sin0 = f16[4] sine(arg0), sharding={maximal device=0}
  mul0 = f16[4] multiply(sin0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(mul0, arg2), sharding={maximal device=0}
  ROOT %tuple = (f16[4], f16[4]) tuple(mul0, mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  EXPECT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = module->entry_computation()->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
    const auto& sharding = inst->sharding();
    EXPECT_TRUE(sharding.HasUniqueDevice());
  }
}

TEST_F(ShardingPassTest, UnsupportedSharding) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  a0 = s32[] parameter(0)
  a1 = f16[4] parameter(1)
  ROOT %tuple = () tuple(), sharding={replicated}
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  EXPECT_TRUE(shardingPass.Run(module).ValueOrDie());
}

TEST_F(ShardingPassTest, UnsupportedAndSupportedShardingMixed) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  arg0 = f16[4] parameter(0)
  arg1 = f16[4] parameter(1)
  arg2 = f16[4] parameter(2)
  sin0 = f16[4] sine(arg0), sharding={replicated}
  mul0 = f16[4] multiply(sin0, arg1), sharding={maximal device=0}
  mul1 = f16[4] multiply(mul0, arg2), sharding={maximal device=0}
  ROOT %tuple = (f16[4], f16[4]) tuple(mul0, mul1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  ShardingPass shardingPass;
  EXPECT_TRUE(shardingPass.Run(module).ValueOrDie());

  auto insts = module->entry_computation()->instructions();
  for (auto* inst : insts) {
    EXPECT_TRUE(inst->has_sharding());
    const auto& sharding = inst->sharding();
    EXPECT_TRUE(sharding.HasUniqueDevice());
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
