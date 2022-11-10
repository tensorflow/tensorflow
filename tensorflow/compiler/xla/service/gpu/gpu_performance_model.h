/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_PERFORMANCE_MODEL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_PERFORMANCE_MODEL_H_

#include <cstdint>
#include <vector>

#include "absl/time/time.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_cost_analysis.h"

namespace xla {
namespace gpu {

class GpuPerformanceModel {
  // Estimated values in the absence of easy ways to query them.
  static constexpr absl::Duration kKernelLaunchOverhead = absl::Microseconds(1);
  static constexpr float kL2CacheSpeedup = 2.5;
  static constexpr float kL1CacheSpeedup = 8;
  // A very conservative estimate. L1 size varies because it can be dynamically
  // configured as shared memory; there is no easy way to query its actual size;
  // also we do not count what occupies cache, but rather claim that what is
  // much smaller than the cache size will likely stay in it.
  // For reference, it can be up to 256 kB per SM on RTX A6000.
  static constexpr float kL1CacheSizePerSM = 2 * 1024;

 public:
  struct RunTimes {
    absl::Duration time_unfused;
    absl::Duration time_fused;
  };
  static struct RunTimes EstimateRunTimes(
      const HloInstruction* producer, const GpuHloCostAnalysis* cost_analysis,
      const GpuDeviceInfo& gpu_device_info,
      const std::vector<HloInstruction*> fused_users = {},
      bool multi_output = false);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_PERFORMANCE_MODEL_H_
