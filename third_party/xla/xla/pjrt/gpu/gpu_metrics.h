/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PJRT_GPU_GPU_METRICS_H_
#define XLA_PJRT_GPU_GPU_METRICS_H_

#include <cstdint>

#include "absl/strings/string_view.h"

namespace xla {
namespace gpu_metrics {

inline constexpr absl::string_view freeGpuSystemMemoryMetricName =
    "/pjrt/gpu/free_gpu_system_memory";

void RecordFreeGpuSystemMemory(int device_ordinal, int64_t free_memory);

int64_t GetFreeGpuSystemMemory(int gpu_id);

}  // namespace gpu_metrics
}  // namespace xla

#endif  // XLA_PJRT_GPU_GPU_METRICS_H_
