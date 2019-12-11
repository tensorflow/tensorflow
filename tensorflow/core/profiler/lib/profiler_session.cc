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

#include <stddef.h>

#include <vector>

#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/profiler/internal/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_utils.h"
#endif
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/trace_events.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace {

// Given a node_name in the format "op_name:op_type", returns the "op_type".
// If the "op_type" is missing, returns the node_name.
// This is done so all ops with the same type appear in the same color in trace
// viewer.
inline string EventName(absl::string_view node_name) {
  // NOTE: open source device tracer now append cupti kernel name after
  // annotation as node_name, @@ is used as separator. kernel name is
  // demangled and possibly contains "::" patterns.
  std::vector<absl::string_view> segments = absl::StrSplit(node_name, "@@");
  if (segments.size() > 1) {  // unparsed
    // find the last annotation.
    std::vector<absl::string_view> annotation_stack =
        absl::StrSplit(segments.front(), "::");
    // strip trace argument.
    std::vector<absl::string_view> annotation_parts =
        absl::StrSplit(annotation_stack.back(), '#');
    std::vector<absl::string_view> parts =
        absl::StrSplit(annotation_parts.front(), ':');
    return string(parts.back());
  } else {
    std::vector<absl::string_view> parts = absl::StrSplit(node_name, ':');
    return string(parts.back());
  }
}

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
    // Don't generate trace events for "derived or aggregated" devices, the
    // events in these devices are duplicated from other streams.
    if (absl::EndsWith(device_stats->device(), "stream:all") ||
        absl::EndsWith(device_stats->device(), "sync") ||
        absl::EndsWith(device_stats->device(), "memcpy"))
      continue;
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
    for (auto node : device_stats->node_stats()) {
      if (node.all_start_micros() < profile_start_time_micros ||
          node.all_start_micros() + node.all_end_rel_micros() >
              profile_end_time_micros) {
        continue;
      }
      auto* event = trace->add_trace_events();
      auto* args = event->mutable_args();
      event->set_device_id(device_id);
      event->set_resource_id(node.thread_id());
      event->set_name(EventName(node.node_name()));
      event->set_timestamp_ps(
          (node.all_start_micros() - profile_start_time_micros) *
          EnvTime::kMicrosToPicos);
      event->set_duration_ps(node.all_end_rel_micros() *
                             EnvTime::kMicrosToPicos);
      if (!node.timeline_label().empty()) {
        std::vector<absl::string_view> label_parts =
            absl::StrSplit(node.timeline_label(), "@@");
        (*args)["label"] = string(label_parts.front());
        if (label_parts.size() == 2) {
          // NOTE: we can further parse annotation here.
          (*args)["annotation"] = string(label_parts.back());
        }
      }
      if (event->name() != node.node_name()) {
        (*args)["long name"] = node.node_name();
      }
    }
  }

  // TODO(fishx): Convert allocation data as well.
}
}  // namespace

/*static*/ std::unique_ptr<ProfilerSession> ProfilerSession::Create(
    const profiler::ProfilerOptions& options) {
  return absl::WrapUnique(new ProfilerSession(options));
}

/*static*/ std::unique_ptr<ProfilerSession> ProfilerSession::Create() {
  int64 host_tracer_level = 2;
  tensorflow::Status s = ReadInt64FromEnvVar("TF_PROFILER_HOST_TRACER_LEVEL", 2,
                                             &host_tracer_level);
  if (!s.ok()) {
    LOG(WARNING) << "ProfilerSession: " << s.error_message();
  }
  profiler::ProfilerOptions options;
  options.host_tracer_level = host_tracer_level;
  return Create(options);
}

Status ProfilerSession::Status() {
  mutex_lock l(mutex_);
  return status_;
}

Status ProfilerSession::CollectData(profiler::XSpace* space) {
  mutex_lock l(mutex_);
  if (!status_.ok()) return status_;
  for (auto& profiler : profilers_) {
    profiler->Stop().IgnoreError();
  }

  for (auto& profiler : profilers_) {
    profiler->CollectData(space).IgnoreError();
  }

  if (active_) {
    // Allow another session to start.
#if !defined(IS_MOBILE_PLATFORM)
    profiler::ReleaseProfilerLock();
#endif
    active_ = false;
  }

  return Status::OK();
}

Status ProfilerSession::CollectData(RunMetadata* run_metadata) {
  mutex_lock l(mutex_);
  if (!status_.ok()) return status_;
  for (auto& profiler : profilers_) {
    profiler->Stop().IgnoreError();
  }

  for (auto& profiler : profilers_) {
    profiler->CollectData(run_metadata).IgnoreError();
  }

  if (active_) {
    // Allow another session to start.
#if !defined(IS_MOBILE_PLATFORM)
    profiler::ReleaseProfilerLock();
#endif
    active_ = false;
  }

  return Status::OK();
}

Status ProfilerSession::SerializeToString(string* content) {
  RunMetadata run_metadata;
  TF_RETURN_IF_ERROR(CollectData(&run_metadata));

  profiler::Trace trace;
  ConvertRunMetadataToTraceEvent(
      &run_metadata, &trace, start_time_micros_,
      Env::Default()->NowNanos() / EnvTime::kMicrosToNanos);

  trace.SerializeToString(content);
  return Status::OK();
}

ProfilerSession::ProfilerSession(const profiler::ProfilerOptions& options)
#if !defined(IS_MOBILE_PLATFORM)
    : active_(profiler::AcquireProfilerLock()),
#else
    : active_(false),
#endif
      start_time_micros_(Env::Default()->NowNanos() / EnvTime::kMicrosToNanos) {
  if (!active_) {
    status_ = tensorflow::Status(error::UNAVAILABLE,
                                 "Another profiler session is active.");
    return;
  }

  LOG(INFO) << "Profiler session started.";

#if !defined(IS_MOBILE_PLATFORM)
  CreateProfilers(options, &profilers_);
#endif
  status_ = Status::OK();

  for (auto& profiler : profilers_) {
    auto start_status = profiler->Start();
    if (!start_status.ok()) {
      LOG(WARNING) << "Encountered error while starting profiler: "
                   << start_status.ToString();
    }
  }
}

ProfilerSession::~ProfilerSession() {
  for (auto& profiler : profilers_) {
    profiler->Stop().IgnoreError();
  }

  if (active_) {
    // Allow another session to start.
#if !defined(IS_MOBILE_PLATFORM)
    profiler::ReleaseProfilerLock();
#endif
  }
}
}  // namespace tensorflow
