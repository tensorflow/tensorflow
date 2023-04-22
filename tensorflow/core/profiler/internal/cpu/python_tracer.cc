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
#include <memory>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
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

  Status CollectData(RunMetadata* run_metadata) override;

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
  return Status::OK();
}

Status PythonTracer::Stop() {
  if (!recording_) {
    return errors::Internal("PythonTracer not started");
  }
  VLOG(1) << __FUNCTION__;
  context_ = PythonHooks::GetSingleton()->Stop();
  recording_ = false;
  return Status::OK();
}

Status PythonTracer::CollectData(RunMetadata* run_metadata) {
  return errors::Unimplemented(
      "CollectData to RunMetadata not supported in PythonTracer");
}

Status PythonTracer::CollectData(XSpace* space) {
  VLOG(2) << "Collecting data to XSpace from PythonTracer.";
  if (context_) {
    context_->Finalize(space);
    context_.reset();
  }
  return Status::OK();
}

}  // namespace

// Not in anonymous namespace for testing purposes.
std::unique_ptr<ProfilerInterface> CreatePythonTracer(
    const ProfileOptions& options) {
  if (options.python_tracer_level() == 0 && options.host_tracer_level() == 0) {
    return nullptr;
  }
  PythonHooksOptions pyhooks_options;
  pyhooks_options.enable_trace_python_function = options.python_tracer_level();
  pyhooks_options.enable_python_traceme = options.host_tracer_level();
  return absl::make_unique<PythonTracer>(pyhooks_options);
}

auto register_python_tracer_factory = [] {
  RegisterProfilerFactory(&CreatePythonTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace tensorflow
