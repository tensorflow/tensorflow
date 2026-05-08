/* Copyright 2026 The OpenXLA Authors.

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

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

using CpuProfilingTest = HloPjRtTestBase;

TEST_F(CpuProfilingTest, ProfilingNWayConditionalDoesNotCrash) {
  const std::string hlo_text = R"(
HloModule module

branch0 {
  p0 = f32[] parameter(0)
  ROOT c0 = f32[] constant(1.0)
}

branch1 {
  p1 = f32[] parameter(0)
  ROOT c1 = f32[] constant(2.0)
}

branch2 {
  p2 = f32[] parameter(0)
  ROOT c2 = f32[] constant(3.0)
}

ENTRY main {
  index = s32[] parameter(0)
  arg = f32[] parameter(1)
  ROOT sel = f32[] conditional(index, arg, arg, arg),
    branch_computations={branch0, branch1, branch2}
}
)";

  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_hlo_profile(true);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  // We just want to ensure that compilation (which includes profiling
  // instrumentation) does not crash.
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
}

}  // namespace
}  // namespace cpu
}  // namespace xla
