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

#include "xla/service/gpu/flag_utils.h"

#include <gtest/gtest.h>
#include "xla/backends/gpu/transforms/double_buffer_loop_unrolling.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

TEST(FlagUtilsTest, IsPassEnabledAtOptimizationEffort) {
  HloModuleConfig config;
  config.set_exec_time_optimization_effort(kExtraCollectiveOptimizations + 1);
  HloModule module("test_module", config);

  // Collective optimization passes.
  EXPECT_TRUE(IsPassEnabledAtOptimizationEffort<CollectivePipeliner>(module));
  EXPECT_TRUE(
      IsPassEnabledAtOptimizationEffort<DoubleBufferLoopUnrolling>(module));
  EXPECT_TRUE(
      IsPassEnabledAtOptimizationEffort<LatencyHidingScheduler>(module));

  // Other passes.
  EXPECT_TRUE(IsPassEnabledAtOptimizationEffort<HloDCE>(module));

  config.set_exec_time_optimization_effort(kExtraCollectiveOptimizations - 1);
  module.set_config(config);

  // Collective optimization passes.
  EXPECT_FALSE(IsPassEnabledAtOptimizationEffort<CollectivePipeliner>(module));
  EXPECT_FALSE(
      IsPassEnabledAtOptimizationEffort<DoubleBufferLoopUnrolling>(module));
  EXPECT_FALSE(
      IsPassEnabledAtOptimizationEffort<LatencyHidingScheduler>(module));
}

TEST(FlagUtilsTest, IsPassEnabledAtOptimizationLevel) {
  HloModule module("test_module", {});

  for (ExecutionOptions::EffortLevel level :
       {ExecutionOptions::EFFORT_O1, ExecutionOptions::EFFORT_O2,
        ExecutionOptions::EFFORT_O3}) {
    HloModuleConfig config;
    config.set_optimization_level(level);
    module.set_config(config);

    // Collective optimization passes.
    EXPECT_TRUE(IsPassEnabledAtOptimizationEffort<CollectivePipeliner>(module));
    EXPECT_TRUE(
        IsPassEnabledAtOptimizationEffort<DoubleBufferLoopUnrolling>(module));
    EXPECT_TRUE(
        IsPassEnabledAtOptimizationEffort<LatencyHidingScheduler>(module));

    // Other passes.
    EXPECT_TRUE(IsPassEnabledAtOptimizationEffort<HloDCE>(module));
  }

  HloModuleConfig config;
  config.set_optimization_level(ExecutionOptions::EFFORT_O0);
  module.set_config(config);

  // Collective optimization passes.
  EXPECT_FALSE(IsPassEnabledAtOptimizationEffort<CollectivePipeliner>(module));
  EXPECT_FALSE(
      IsPassEnabledAtOptimizationEffort<DoubleBufferLoopUnrolling>(module));
  EXPECT_FALSE(
      IsPassEnabledAtOptimizationEffort<LatencyHidingScheduler>(module));
}

TEST(FlagUtilsTest, HostOffloadingNotAutoEnabledAtO1) {
  // This test ensures that the host offloading collective pipeliner remains
  // opt-in only and is not automatically enabled by optimization levels.
  DebugOptions default_options = DefaultDebugOptionsIgnoringFlags();
  EXPECT_FALSE(default_options.xla_gpu_enable_pipelined_host_offloading())
      << "Host offloading must be disabled by default";

  // Test across optimization levels
  for (ExecutionOptions::EffortLevel level :
       {ExecutionOptions::EFFORT_O0, ExecutionOptions::EFFORT_O1,
        ExecutionOptions::EFFORT_O2, ExecutionOptions::EFFORT_O3}) {
    HloModuleConfig config;
    config.set_optimization_level(level);
    HloModule module("test_module", config);

    bool collective_pipeliner_enabled =
        IsPassEnabledAtOptimizationEffort<CollectivePipeliner>(module);

    if (level >= ExecutionOptions::EFFORT_O1) {
      // Standard CollectivePipeliner is enabled at O1+
      EXPECT_TRUE(collective_pipeliner_enabled);
    } else {
      EXPECT_FALSE(collective_pipeliner_enabled);
    }
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
