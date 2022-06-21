/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/backends/cpu/python_tracer.h"

#include <memory>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/python/profiler/internal/python_hooks.h"

namespace tensorflow {
namespace profiler {
namespace {

// This profiler interface enables Python function call tracing.
class PythonTracer : public ProfilerInterface {
 public:
  explicit PythonTracer(const PythonHooksOptions& options)
      : options_(options) {}
  ~PythonTracer() override;

  Status Start() override;

  Status Stop() override;

  Status CollectData(XSpace* space) override;

 private:
  bool recording_ = false;
  const PythonHooksOptions options_;
  std::unique_ptr<tensorflow::profiler::PythonHookContext> context_;

  TF_DISALLOW_COPY_AND_ASSIGN(PythonTracer);
};

PythonTracer::~PythonTracer() {
  Stop().IgnoreError();
}

Status PythonTracer::Start() {
  if (recording_) {
    return errors::Internal("PythonTracer already started");
  }
  VLOG(1) << __FUNCTION__;
  recording_ = true;
  PythonHooks::GetSingleton()->Start(options_);
  return OkStatus();
}

Status PythonTracer::Stop() {
  if (!recording_) {
    return errors::Internal("PythonTracer not started");
  }
  VLOG(1) << __FUNCTION__;
  context_ = PythonHooks::GetSingleton()->Stop();
  recording_ = false;
  return OkStatus();
}

Status PythonTracer::CollectData(XSpace* space) {
  VLOG(2) << "Collecting data to XSpace from PythonTracer.";
  if (context_) {
    context_->Finalize(space);
    context_.reset();
  }
  return OkStatus();
}

}  // namespace

std::unique_ptr<ProfilerInterface> CreatePythonTracer(
    const PythonTracerOptions& options) {
  if (!options.enable_trace_python_function && !options.enable_python_traceme) {
    return nullptr;
  }
  PythonHooksOptions pyhooks_options;
  pyhooks_options.enable_trace_python_function =
      options.enable_trace_python_function;
  pyhooks_options.enable_python_traceme = options.enable_python_traceme;
  pyhooks_options.end_to_end_mode = options.end_to_end_mode;
  return absl::make_unique<PythonTracer>(pyhooks_options);
}

}  // namespace profiler
}  // namespace tensorflow
