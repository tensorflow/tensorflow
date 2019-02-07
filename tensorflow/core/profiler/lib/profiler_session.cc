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

#include "tensorflow/core/profiler/lib/profiler_session.h"
#include <string>
#include "tensorflow/contrib/tpu/profiler/trace_events.pb.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/internal/gpu/tracer.h"
#include "tensorflow/core/profiler/internal/runtime/eager_profiler.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

namespace {

void ConvertRunMetadataToTraceEvent(RunMetadata* run_metadata,
                                    tpu::Trace* trace,
                                    const uint64 profile_start_time_micros) {
  auto trace_devices = trace->mutable_devices();
  // TODO(fishx): use a lighter representation instead of GraphDef to insert
  // python information into trace event.

  for (size_t device_id = 0;
       device_id < run_metadata->step_stats().dev_stats_size(); ++device_id) {
    // Create device
    auto* device_stats =
        run_metadata->mutable_step_stats()->mutable_dev_stats(device_id);
    tensorflow::tpu::Device device;
    device.set_name(device_stats->device());
    device.set_device_id(device_id);
    tensorflow::tpu::Resource resource;
    resource.set_name("0");
    resource.set_resource_id(0);
    (*device.mutable_resources())[0] = resource;
    for (const auto& thread_name : device_stats->thread_names()) {
      tensorflow::tpu::Resource resource;
      resource.set_resource_id(thread_name.first);
      resource.set_name(thread_name.second);
      (*device.mutable_resources())[thread_name.first] = resource;
    }
    (*trace_devices)[device_id] = device;

    // Emit events.
    for (auto node :
         run_metadata->step_stats().dev_stats(device_id).node_stats()) {
      auto* event = trace->add_trace_events();
      auto* args = event->mutable_args();
      event->set_device_id(device_id);
      if (device_stats->device().find("host:CPU") != string::npos) {
        event->set_resource_id(node.thread_id());
      } else {
        event->set_resource_id(0);
      }
      event->set_name(node.node_name());
      event->set_timestamp_ps(
          (node.all_start_micros() - profile_start_time_micros) *
          EnvTime::kMicrosToPicos);
      event->set_duration_ps(node.all_end_rel_micros() *
                             EnvTime::kMicrosToPicos);
      (*args)["label"] = node.timeline_label();
    }
  }

  // TODO(fishx): Convert allocation data as well.
}

}  // namespace

/*static*/ std::unique_ptr<ProfilerSession> ProfilerSession::Create(
    EagerContext* const context) {
  return absl::WrapUnique(new ProfilerSession(context));
}

Status ProfilerSession::Status() {
  mutex_lock l(mutex_);
  return status_;
}

Status ProfilerSession::SerializeToString(string* content) {
  mutex_lock l(mutex_);
  if (!status_.ok()) return status_;
  for (auto& profiler : profilers_) {
    profiler->Stop().IgnoreError();
  }
  RunMetadata run_metadata;
  for (auto& profiler : profilers_) {
    profiler->CollectData(&run_metadata).IgnoreError();
  }

  tpu::Trace trace;

  ConvertRunMetadataToTraceEvent(&run_metadata, &trace, start_time_micros_);

  trace.SerializeToString(content);
  return Status::OK();
}

ProfilerSession::ProfilerSession(EagerContext* const context)
    : start_time_micros_(Env::Default()->NowNanos() / EnvTime::kMicrosToNanos) {
  LOG(INFO) << "Profile Session started.";

  if (context != nullptr) {
    profilers_.push_back(
        tensorflow::profiler::runtime::EagerProfiler::Create(context));
  }
  profilers_.push_back(tensorflow::profiler::gpu::Tracer::Create());

  status_ = Status::OK();

  for (auto& profiler : profilers_) {
    profiler->Start().IgnoreError();
  }
}

ProfilerSession::~ProfilerSession() {
  for (auto& profiler : profilers_) {
    profiler->Stop().IgnoreError();
  }
}

}  // namespace tensorflow
