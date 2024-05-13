/* Copyright 2020 The OpenXLA Authors.

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
#include "xla/backends/profiler/cpu/python_tracer.h"

#include <memory>

#include "absl/status/status.h"
#include "xla/python/profiler/internal/python_hooks.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

// This profiler interface enables Python function call tracing.
class PythonTracer : public tsl::profiler::ProfilerInterface {
 public:
  explicit PythonTracer(const PythonHooksOptions& options)
      : options_(options) {}
  ~PythonTracer() override;

  absl::Status Start() override;  // TENSORFLOW_STATUS_OK

  absl::Status Stop() override;  // TENSORFLOW_STATUS_OK

  absl::Status CollectData(  // TENSORFLOW_STATUS_OK
      tensorflow::profiler::XSpace* space) override;

 private:
  bool recording_ = false;
  const PythonHooksOptions options_;
  std::unique_ptr<PythonHookContext> context_;

  PythonTracer(const PythonTracer&) = delete;
  void operator=(const PythonTracer&) = delete;
};

PythonTracer::~PythonTracer() { Stop().IgnoreError(); }  // NOLINT

absl::Status PythonTracer::Start() {  // TENSORFLOW_STATUS_OK
  if (recording_) {
    return tsl::errors::Internal("PythonTracer already started");
  }
  VLOG(1) << __FUNCTION__;
  recording_ = true;
  PythonHooks::GetSingleton()->Start(options_);
  return absl::OkStatus();
}

absl::Status PythonTracer::Stop() {  // TENSORFLOW_STATUS_OK
  if (!recording_) {
    return tsl::errors::Internal("PythonTracer not started");
  }
  VLOG(1) << __FUNCTION__;
  context_ = PythonHooks::GetSingleton()->Stop();
  recording_ = false;
  return absl::OkStatus();
}

absl::Status PythonTracer::CollectData(  // TENSORFLOW_STATUS_OK
    tensorflow::profiler::XSpace* space) {
  VLOG(2) << "Collecting data to XSpace from PythonTracer.";
  if (context_) {
    context_->Finalize(space);
    context_.reset();
  }
  return absl::OkStatus();
}

}  // namespace

std::unique_ptr<tsl::profiler::ProfilerInterface> CreatePythonTracer(
    const PythonTracerOptions& options) {
  if (!options.enable_trace_python_function && !options.enable_python_traceme) {
    return nullptr;
  }
  PythonHooksOptions pyhooks_options;
  pyhooks_options.enable_trace_python_function =
      options.enable_trace_python_function;
  pyhooks_options.enable_python_traceme = options.enable_python_traceme;
  pyhooks_options.end_to_end_mode = options.end_to_end_mode;
  return std::make_unique<PythonTracer>(pyhooks_options);
}

}  // namespace profiler
}  // namespace xla
