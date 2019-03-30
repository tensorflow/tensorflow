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
#include <cstddef>
#include <string>
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/internal/gpu/tracer.h"
#include "tensorflow/core/profiler/internal/runtime/eager_profiler.h"
#include "tensorflow/core/profiler/trace_events.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

namespace {

// Track whether there's an active ProfilerSession.
// Prevents another ProfilerSession from creating ProfilerInterface(s), as they
// use singletons that do not allow concurrent profiling request (e.g.,
// DeviceTracer).
std::atomic<bool> session_active = ATOMIC_VAR_INIT(false);

void AssignLanes(RunMetadata* run_metadata) {
  for (size_t device_id = 0;
       device_id < run_metadata->step_stats().dev_stats_size(); ++device_id) {
    auto* device_stats =
        run_metadata->mutable_step_stats()->mutable_dev_stats(device_id);
    if (device_stats->thread_names_size() > 0 ||
        device_stats->node_stats_size() == 0) {
      continue;
    }
    std::vector<uint64> lanes;
    for (auto ns = device_stats->mutable_node_stats()->rbegin();
         ns != device_stats->mutable_node_stats()->rend(); ns++) {
      uint64 end_micros = ns->all_start_micros() + ns->all_end_rel_micros();
      bool found_lane = false;
      for (size_t l = 0; l < lanes.size(); l++) {
        if (end_micros <= lanes[l]) {
          ns->set_thread_id(l);
          found_lane = true;
          lanes[l] = ns->all_start_micros();
          break;
        }
      }
      if (!found_lane) {
        ns->set_thread_id(lanes.size());
        lanes.push_back(ns->all_start_micros());
      }
    }
  }
}

void ConvertRunMetadataToTraceEvent(RunMetadata* run_metadata,
                                    profiler::Trace* trace,
                                    const uint64 profile_start_time_micros,
                                    const uint64 profile_end_time_micros) {
  AssignLanes(run_metadata);
  auto trace_devices = trace->mutable_devices();

  for (size_t device_id = 0;
       device_id < run_metadata->step_stats().dev_stats_size(); ++device_id) {
    // Create device
    auto* device_stats =
        run_metadata->mutable_step_stats()->mutable_dev_stats(device_id);
    profiler::Device device;
    device.set_name(device_stats->device());
    device.set_device_id(device_id);
    profiler::Resource resource;
    resource.set_name("0");
    resource.set_resource_id(0);
    (*device.mutable_resources())[0] = resource;
    for (const auto& thread_name : device_stats->thread_names()) {
      profiler::Resource resource;
      resource.set_resource_id(thread_name.first);
      resource.set_name(thread_name.second);
      (*device.mutable_resources())[thread_name.first] = resource;
    }
    (*trace_devices)[device_id] = device;

    // Emit events.
    for (auto node :
         run_metadata->step_stats().dev_stats(device_id).node_stats()) {
      if (node.all_start_micros() < profile_start_time_micros ||
          node.all_start_micros() + node.all_end_rel_micros() >
              profile_end_time_micros) {
        continue;
      }
      auto* event = trace->add_trace_events();
      auto* args = event->mutable_args();
      event->set_device_id(device_id);
      event->set_resource_id(node.thread_id());
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
    ProfilerContext* const context) {
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

  if (active_) {
    // Allow another session to start.
    session_active.store(false);
    active_ = false;
  }

  profiler::Trace trace;

  ConvertRunMetadataToTraceEvent(
      &run_metadata, &trace, start_time_micros_,
      Env::Default()->NowNanos() / EnvTime::kMicrosToNanos);

  trace.SerializeToString(content);
  return Status::OK();
}

ProfilerSession::ProfilerSession(ProfilerContext* const context)
    : active_(!session_active.exchange(true)),
      start_time_micros_(Env::Default()->NowNanos() / EnvTime::kMicrosToNanos) {
  if (!active_) {
    status_ = tensorflow::Status(tensorflow::error::Code::UNAVAILABLE,
                                 "Another profiling session is active.");
    return;
  }

  LOG(INFO) << "Profile Session started.";

  if (context->eager_context != nullptr) {
    profilers_.push_back(tensorflow::profiler::runtime::EagerProfiler::Create(
        context->eager_context));
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

  if (active_) {
    // Allow another session to start.
    session_active.store(false);
  }
}

}  // namespace tensorflow
