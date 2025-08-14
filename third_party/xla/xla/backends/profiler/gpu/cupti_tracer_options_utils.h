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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_TRACER_OPTIONS_UTILS_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_TRACER_OPTIONS_UTILS_H_

#include "absl/status/status.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace xla {
namespace profiler {

// Updates the CuptiTracerOptions and CuptiTracerCollectorOptions based on the
// given profiler options.
absl::Status UpdateCuptiTracerOptionsFromProfilerOptions(
    const tensorflow::ProfileOptions& profile_options,
    CuptiTracerOptions& tracer_options,
    CuptiTracerCollectorOptions& collector_options);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_TRACER_OPTIONS_UTILS_H_
