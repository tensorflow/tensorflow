/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_PJRT_METRICS_H_
#define XLA_PJRT_METRICS_H_

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "tsl/lib/monitoring/counter.h"

// Simplified version of tensorflow/core/framework/metrics.h for JAX.

namespace xla {
namespace metrics {

inline constexpr absl::string_view kPjrtCompilerCompileComputationMetricName =
    "/pjrt/compiler/is_compiling_computation";
inline constexpr absl::string_view kPjrtCompilerCompileModuleMetricName =
    "/pjrt/compiler/is_compiling_module";

void ReportExecutableEnqueueTime(uint64_t running_time_usecs);

void RecordPjrtCompilerCompileComputationStatus(bool is_compiling);

void RecordPjrtCompilerCompileModuleStatus(bool is_compiling);

}  // namespace metrics
}  // namespace xla

#endif  // XLA_PJRT_METRICS_H_
