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
#include <memory>

#include "xla/backends/profiler/cpu/python_tracer.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace xla {
namespace profiler {
namespace {

std::unique_ptr<tsl::profiler::ProfilerInterface> CreatePythonTracer(
    const tensorflow::ProfileOptions& profile_options) {
  PythonTracerOptions options;
  options.enable_trace_python_function = profile_options.python_tracer_level();
  options.enable_python_traceme = profile_options.host_tracer_level();
  return CreatePythonTracer(options);
}

auto register_python_tracer_factory = [] {
  RegisterProfilerFactory(&CreatePythonTracer);
  return 0;
}();

}  // namespace
}  // namespace profiler
}  // namespace xla
