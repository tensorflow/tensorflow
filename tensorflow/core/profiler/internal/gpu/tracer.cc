/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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
#include "tensorflow/core/profiler/internal/gpu/tracer.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"

namespace tensorflow {
namespace profiler {
namespace gpu {

/* static */ std::unique_ptr<ProfilerInterface> Tracer::Create() {
  return absl::WrapUnique(new Tracer());
}

Status Tracer::Start() {
  device_tracer_ = CreateDeviceTracer();
  if (!device_tracer_) {
    return Status(tensorflow::error::Code::FAILED_PRECONDITION,
                  "Failed to create device tracer.");
  }
  return device_tracer_->Start();
}

Status Tracer::Stop() {
  if (!device_tracer_) {
    return Status(tensorflow::error::Code::FAILED_PRECONDITION,
                  "No running device tracer.");
  }
  return device_tracer_->Stop();
}

Status Tracer::CollectData(RunMetadata* run_metadata) {
  if (!device_tracer_) {
    return Status(tensorflow::error::Code::FAILED_PRECONDITION,
                  "No running device tracer.");
  }
  auto step_stats_collector =
      absl::make_unique<StepStatsCollector>(run_metadata->mutable_step_stats());
  Status s = device_tracer_->Collect(step_stats_collector.get());
  step_stats_collector->Finalize();
  return s;
}

Tracer::Tracer() {}

}  // namespace gpu
}  // namespace profiler
}  // namespace tensorflow
