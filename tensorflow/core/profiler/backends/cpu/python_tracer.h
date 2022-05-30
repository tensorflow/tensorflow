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
#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_CPU_HOST_TRACER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_CPU_HOST_TRACER_H_

#include <memory>

#include "tensorflow/core/profiler/lib/profiler_interface.h"

namespace tensorflow {
namespace profiler {

struct PythonTracerOptions {
  // Whether to enable python function calls tracing.
  // NOTE: Runtime overhead ensues if enabled.
  bool enable_trace_python_function = false;

  // Whether to enable python TraceMe instrumentation.
  bool enable_python_traceme = true;

  // Whether profiling stops within an atexit handler.
  bool end_to_end_mode = false;
};

std::unique_ptr<ProfilerInterface> CreatePythonTracer(
    const PythonTracerOptions& options);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_CPU_HOST_TRACER_H_
