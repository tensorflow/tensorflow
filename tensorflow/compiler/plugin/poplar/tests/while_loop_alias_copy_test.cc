/* Copyright 2019 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_compiler.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include <stdlib.h>

namespace xla {
namespace poplarplugin {
namespace {

using WhileLoopAliasCopyTest = HloTestBase;

/*
 * If an output is a partial alias of another input, then it needs to be copied
 * into a temporary tensor before being copied back to the input.
 *
 * Parameter 0 of the loop body is not aliased, while parameter 1 is an alias of
 * both parameter 0 and parameter 1.  Therefore it should have a temporary copy
 * made before the loop output->input copy is done.  If this does not happen
 * then it will try to copy the output 1 tensor over part of itself.
 */
TEST_F(WhileLoopAliasCopyTest, CopyAlias) {
  const char* const hlo_string = R"(
HloModule ModuleWithWhile

body {
  p = (s32[4],s32[4]) parameter(0)
  p.0 = s32[4] get-tuple-element(p), index=0
  p.1 = s32[4] get-tuple-element(p), index=1
  c.0 = s32[8] concatenate(p.0, p.1), dimensions={0}
  s.0 = s32[4] slice(c.0), slice={[2:6]}
  a.0 = s32[4] add(p.0, p.0)
  ROOT root = (s32[4],s32[4]) tuple(a.0, s.0)
}

condition {
  p_cond = (s32[4],s32[4]) parameter(0)
  p_cond.0 = s32[4] get-tuple-element(p_cond), index=0
  p_s0 = s32[1] slice(p_cond.0), slice={[0:1]}
  p_s1 = s32[] reshape(p_s0)
  p_const = s32[] constant(10)
  ROOT result = pred[] compare(p_s1, p_const), direction=LT
}

ENTRY entry {
  const_0 = s32[4] constant({0, 0, 0, 0})
  const_1 = s32[4] constant({10, 10, 10, 10})
  repeat_init = (s32[4],s32[4]) tuple(const_0, const_1)
  ROOT while = (s32[4],s32[4]) while(repeat_init), condition=condition, body=body
}
)";

  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());

  auto module_or_status = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module_or_status.ok());

  auto& module = module_or_status.ValueOrDie();

  auto* platform =
      se::MultiPlatformManager::PlatformWithName("Poplar").ConsumeValueOrDie();
  auto* stream_executor = platform->ExecutorForDevice(0).ConsumeValueOrDie();

  PoplarCompiler compiler;
  EXPECT_IS_OK(
      compiler.RunBackend(std::move(module), stream_executor, nullptr));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
