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

#include "xla/pjrt/gpu/gpu_metrics.h"

#include <cstdint>

#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/monitoring/gauge.h"

namespace xla {
namespace {
auto* free_gpu_system_memory = tsl::monitoring::Gauge<int64_t, 1>::New(
    gpu_metrics::freeGpuSystemMemoryMetricName,
    "Record the free GPU system memory.", "gpu_id");
}  // namespace

namespace gpu_metrics {

void RecordFreeGpuSystemMemory(const int device_ordinal,
                               const int64_t free_memory) {
  free_gpu_system_memory->GetCell(absl::StrCat(device_ordinal))
      ->Set(free_memory);
}

int64_t GetFreeGpuSystemMemory(int gpu_id) {
  return free_gpu_system_memory->GetCell(absl::StrCat(gpu_id))->value();
}

}  // namespace gpu_metrics
}  // namespace xla
