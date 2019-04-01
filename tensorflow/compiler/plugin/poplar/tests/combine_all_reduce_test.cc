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

#include "tensorflow/compiler/plugin/poplar/driver/passes/combine_all_reduce.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/sync_list_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include <algorithm>

namespace xla {
namespace poplarplugin {
namespace {

using SchedulerTest = HloTestBase;

TEST_F(SchedulerTest, TestSyncScheduler) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

%cluster_1  {
  %arg0 = f16[4] parameter(0)
  %arg1 = f16[4] parameter(1)
  %arg2 = f16[4] parameter(2)
  %a1 = f16[4] all-reduce(arg0), to_apply=add
  %a2 = f16[4] all-reduce(arg1), to_apply=add
  %a3 = f16[4] all-reduce(arg2), to_apply=add
  ROOT %tuple = (f16[4], f16[4], f16[4]) tuple(f16[4] %a1, f16[4] %a2, f16[4] %a3)
}
  )";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto* module = module_or_status.ValueOrDie().get();

  HloMemoryScheduler scheduler(
      [](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), 1);
      },
      CreateSyncListMemoryScheduler(64 * 1024));
  EXPECT_TRUE(scheduler.Run(module).ValueOrDie());

  CombineAllReduce combine_all_reduce;
  EXPECT_TRUE(combine_all_reduce.Run(module).ValueOrDie());

  auto s = module->schedule().sequence(module->entry_computation());
  auto seq = s.instructions();
  ASSERT_EQ(seq.size(), 8);

  auto pred = [](HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kAllReduce;
  };
  ASSERT_EQ(std::count_if(seq.begin(), seq.end(), pred), 1);
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
