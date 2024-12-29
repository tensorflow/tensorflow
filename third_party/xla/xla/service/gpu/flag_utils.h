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

#ifndef XLA_SERVICE_GPU_FLAG_UTILS_H_
#define XLA_SERVICE_GPU_FLAG_UTILS_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/collective_pipeliner.h"
#include "xla/service/gpu/transforms/double_buffer_loop_unrolling.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/latency_hiding_scheduler.h"

namespace xla {
namespace gpu {

// Defines the optimization effort to trigger additional passes which optimize
// communication compute overlap.
constexpr float kExtraCollectiveOptimizations = 0.2;

// Returns true if the pass is enabled via `exec_time_optimization_effort` at
// the potential expense of compile time.
template <typename Pass>
bool IsPassEnabledAtOptimizationEffort(const HloModule& module) {
  float exec_effort = module.config().exec_time_optimization_effort();

  bool is_collective_optimization_pass =
      std::is_same_v<Pass, CollectivePipeliner> ||
      std::is_same_v<Pass, DoubleBufferLoopUnrolling> ||
      std::is_same_v<Pass, LatencyHidingScheduler>;

  if (is_collective_optimization_pass) {
    return exec_effort >= kExtraCollectiveOptimizations;
  }

  return true;
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FLAG_UTILS_H_
