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

#include "tensorflow/compiler/plugin/poplar/driver/passes/scheduler.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using SchedulerTest = HloTestBase;

TEST_F(SchedulerTest, TestScheduler) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %arg1 = f16[4] parameter(1)
  %arg2 = f16[4] parameter(2)
  %sin.0 = f16[4] sine(f16[4] %arg0)
  %mul.0 = f16[4] multiply(f16[4] %sin.0, f16[4] %arg1)
  %mul.1 = f16[4] multiply(f16[4] %mul.0, f16[4] %arg2)
  ROOT %tuple = (f16[4], f16[4]) tuple(f16[4] %mul.0, f16[4] %mul.1)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  Scheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 7);
  EXPECT_EQ(seq[0]->name(), "arg0");
  EXPECT_EQ(seq[1]->name(), "sin.0");
  EXPECT_EQ(seq[2]->name(), "arg1");
  EXPECT_EQ(seq[3]->name(), "mul.0");
  EXPECT_EQ(seq[4]->name(), "arg2");
  EXPECT_EQ(seq[5]->name(), "mul.1");
  EXPECT_EQ(seq[6]->name(), "tuple");
}

TEST_F(SchedulerTest, TestDisconnectedInstructions) {
  std::string hlo_string = R"(
HloModule top

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %arg1 = f16[4] parameter(1)
  ROOT %tuple = (f16[4], f16[4]) tuple(f16[4] %arg0, f16[4] %arg1)
  %const = f16[] constant(1)
  %sum = f16[] add(f16[] const, f16[] const)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  Scheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  // The HloComputation schedules non-root trees of instructions before
  // the actual root

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 5);
  EXPECT_EQ(seq[0]->name(), "const");
  EXPECT_EQ(seq[1]->name(), "sum");
  EXPECT_EQ(seq[2]->name(), "arg0");
  EXPECT_EQ(seq[3]->name(), "arg1");
  EXPECT_EQ(seq[4]->name(), "tuple");
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
