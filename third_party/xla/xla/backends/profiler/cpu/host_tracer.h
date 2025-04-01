/* Copyright 2022 The OpenXLA Authors.

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
#ifndef XLA_BACKENDS_PROFILER_CPU_HOST_TRACER_H_
#define XLA_BACKENDS_PROFILER_CPU_HOST_TRACER_H_

#include <cstdint>
#include <limits>
#include <memory>

#include "tsl/profiler/lib/profiler_interface.h"

namespace xla {
namespace profiler {

struct HostTracerOptions {
  // Levels of host tracing:
  // - Level 0 is used to disable host traces.
  // - Level 1 enables tracing of only user instrumented (or default) TraceMe.
  // - Level 2 enables tracing of all level 1 TraceMe(s) and instrumented high
  //           level program execution details (expensive TF ops, XLA ops, etc).
  //           This is the default.
  // - Level 3 enables tracing of all level 2 TraceMe(s) and more verbose
  //           (low-level) program execution details (cheap TF ops, etc).
  int trace_level = 2;
  uint64_t filter_mask = std::numeric_limits<uint64_t>::max();
};

std::unique_ptr<tsl::profiler::ProfilerInterface> CreateHostTracer(
    const HostTracerOptions& options);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_CPU_HOST_TRACER_H_
