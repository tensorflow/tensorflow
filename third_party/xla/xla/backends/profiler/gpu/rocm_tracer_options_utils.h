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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_OPTIONS_UTILS_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_OPTIONS_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

// Problems found while applying ProfileOptions.advanced_configuration.
struct RocmTracerOptionDiagnostics {
  // Unrecognized gpu_* keys, and values of the wrong type or out of range.
  std::vector<std::string> errors;
  // Keys that are recognized but have no effect on ROCm, and caveats on keys
  // that do.
  std::vector<std::string> warnings;

  bool empty() const { return errors.empty() && warnings.empty(); }
};

// Applies the gpu_* entries of profile_options.advanced_configuration() on top
// of the defaults already in `collector_options`. Keys without the gpu_ prefix
// belong to other profiler components and are ignored.
// `device_count` is the number of GPUs present; out-of-range values of
// gpu_num_chips_to_profile_per_task resolve to it.
//
// Unlike UpdateCuptiTracerOptionsFromProfilerOptions, this does not stop at
// the first bad key: every problem is returned, and every valid key is still
// applied. If the tracer failed Start() instead, ProfilerController would
// never call its CollectData(), and the reason would appear only in the log,
// not in the XSpace. The caller decides what a non-empty error list means.
RocmTracerOptionDiagnostics UpdateRocmTracerOptionsFromProfilerOptions(
    const tensorflow::ProfileOptions& profile_options, uint32_t device_count,
    RocmTraceCollectorOptions& collector_options);

// Appends `diagnostics` to space->errors() and space->warnings(), and logs
// them. `space` may be null, in which case they are only logged.
void AppendOptionDiagnostics(const RocmTracerOptionDiagnostics& diagnostics,
                             tensorflow::profiler::XSpace* space);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_TRACER_OPTIONS_UTILS_H_
