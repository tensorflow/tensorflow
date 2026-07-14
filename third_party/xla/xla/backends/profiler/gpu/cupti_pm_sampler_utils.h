/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_UTILS_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_UTILS_H_

#include <string>

#include "absl/strings/string_view.h"

namespace xla {
namespace profiler {

// Returns the human-readable name for GPU sampling/profiling metrics. If the
// metric name is not found in the map, returns the original metric name.
std::string GetGpuProfileMetricName(absl::string_view metric_name);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_UTILS_H_
