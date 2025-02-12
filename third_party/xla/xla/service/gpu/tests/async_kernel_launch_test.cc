/* Copyright 2024 The OpenXLA Authors.

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

#include <utility>

#include <gtest/gtest.h>
#include "xla/debug_options_flags.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"

namespace xla::gpu {
namespace {

class AsyncKernelLaunchTest : public HloTestBase {};

HloModuleConfig GetModuleConfig() {
  // Allow even small graphs to be launched on the GPU.
  DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);

  HloModuleConfig config;
  config.set_debug_options(debug_options);
  return config;
}

// Run with CUDA to export `xprof` graphs showing the concurrent fusions.
// xla/service/gpu/tests/async_kernel_launch_test \
//   --test_arg=--xprof_end_2_end_upload
TEST_F(AsyncKernelLaunchTest, BasicFusion) {
  const char* hlo_text = R"(
  HloModule m, is_scheduled=true

  add_fusion1 {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] add(p0, p0)
  }

  add_fusion2 {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] add(p0, p0)
  }

  ENTRY main {
    p0 = f32[2,2] parameter(0)
    start1 = ((f32[2,2]), f32[2,2], s32[]) fusion-start(p0),
        kind=kLoop, calls=add_fusion1
    start2 = ((f32[2,2]), f32[2,2], s32[]) fusion-start(p0),
        kind=kLoop, calls=add_fusion2
    done1 = f32[2,2] fusion-done(start1)
    done2 = f32[2,2] fusion-done(start2)
    ROOT done = f32[2,2] add(done1, done2)
  })";

  auto module =
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfig()).value();

  Literal argument = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  Literal expected = LiteralUtil::CreateR2<float>({{4.0, 8.0}, {12.0, 16.0}});

  Literal result = ExecuteNoHloPasses(std::move(module), {&argument});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

}  // namespace
}  // namespace xla::gpu
