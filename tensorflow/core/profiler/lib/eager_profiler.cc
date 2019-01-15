/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/lib/eager_profiler.h"
#include "tensorflow/cc/profiler/profiler.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/device_tracer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

/*static*/ std::unique_ptr<EagerProfiler> EagerProfiler::Create(
    EagerContext* const context) {
  return absl::WrapUnique(new EagerProfiler(context));
}

void EagerProfiler::BeforeClearRunMetadata() {
  mutex_lock l(mutex_);
  run_metadata_.MergeFrom(*context_->RunMetadataProto());
}

Status EagerProfiler::Status() {
  mutex_lock l(mutex_);
  return status_;
}

Status EagerProfiler::SerializeToString(string* content) {
  mutex_lock l(mutex_);
  if (!status_.ok()) return status_;
  Stop();

  // Get profiling data from device tracer
  if (device_tracer_ != nullptr) {
    std::unique_ptr<StepStatsCollector> step_stats_collector(
        new StepStatsCollector(run_metadata_.mutable_step_stats()));
    tensorflow::Status s = device_tracer_->Collect(step_stats_collector.get());
    if (!s.ok()) {
      device_tracer_.reset(nullptr);
      LOG(WARNING) << "Failed to collect data from device tracer. "
                   << s.error_message();
    }
    step_stats_collector->Finalize();
  }

  // TODO(fishx): update tfprof to use a lighter representation instead of
  // GraphDef.
  GraphDef graph;
  std::unique_ptr<tfprof::Profiler> tfprof(new tfprof::Profiler(graph));
  tfprof->AddStep(0, run_metadata_);
  return tfprof->SerializeToString(content);
}

EagerProfiler::EagerProfiler(EagerContext* const context) : context_(context) {
  LOG(INFO) << "Eager Profiler started.";

  status_ = context_->RegisterRunMetadataListener(this);
  if (!status_.ok()) {
    context_ = nullptr;
    LOG(WARNING)
        << "Eager Profiler failed to start. Another profiler is running.";
    return;
  }

  // TODO(fishx): Allow user disable device tracer.
  device_tracer_ = CreateDeviceTracer();
  if (!device_tracer_) {
    LOG(WARNING) << "Continue profiling without device tracer. "
                 << "Failed to create device tracer.";
    return;
  }
  class Status s = device_tracer_->Start();
  if (!s.ok()) {
    device_tracer_.reset(nullptr);
    LOG(WARNING) << "Continue profiling without device tracer. "
                 << s.error_message();
  }
}

EagerProfiler::~EagerProfiler() { Stop(); }

void EagerProfiler::Stop() {
  if (context_ != nullptr) {
    context_->ClearRunMetadataListener();
    run_metadata_.MergeFrom(*context_->RunMetadataProto());
    context_ = nullptr;
    if (device_tracer_ != nullptr) {
      tensorflow::Status s = device_tracer_->Stop();
      if (!s.ok()) {
        device_tracer_.reset(nullptr);
        LOG(WARNING) << "Failed to stop device tracer. " << s.error_message();
      }
    }
    LOG(INFO) << "Eager Profiler ended with status:" << status_;
  }
}

}  // namespace tensorflow
