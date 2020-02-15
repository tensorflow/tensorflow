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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/trace_events.pb.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/ptr_util.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"
#include "tensorflow/core/profiler/internal/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_utils.h"
#include "tensorflow/core/profiler/utils/derived_timeline.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#endif

namespace tensorflow {

/*static*/ std::unique_ptr<ProfilerSession> ProfilerSession::Create(
    const profiler::ProfilerOptions& options) {
  return WrapUnique(new ProfilerSession(options));
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

#if !defined(IS_MOBILE_PLATFORM)
  // Post processing the collected XSpace without hold profiler lock.
  // 1. Merge plane of host events with plane of CUPTI driver api.
  const profiler::XPlane* cupti_driver_api_plane =
      profiler::FindPlaneWithName(*space, profiler::kCuptiDriverApiPlaneName);
  if (cupti_driver_api_plane) {
    profiler::XPlane* host_plane =
        profiler::GetOrCreatePlane(space, profiler::kHostThreads);
    profiler::MergePlanes(*cupti_driver_api_plane, host_plane);
    profiler::RemovePlaneWithName(space, profiler::kCuptiDriverApiPlaneName);
  }
  // 2. Normalize all timestamps by shifting timeline to profiling start time.
  // NOTE: this have to be done before sorting XSpace due to timestamp overflow.
  profiler::NormalizeTimeLine(space, start_time_ns_);
  // 3. Sort each plane of the XSpace
  profiler::SortXSpace(space);
  // 4. Grouping (i.e. marking step number) events in the XSpace.
  profiler::EventGroupNameMap event_group_name_map;
  profiler::GroupTfEvents(space, &event_group_name_map);
  // 5. Generated miscellaneous derived time lines for device planes.
  profiler::GenerateDerivedTimeLines(event_group_name_map, space);
#endif

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
  profiler::Trace trace;
#if !defined(IS_MOBILE_PLATFORM)
  profiler::XSpace xspace;
  TF_RETURN_IF_ERROR(CollectData(&xspace));
  profiler::ConvertXSpaceToTraceEvents(xspace, &trace);
#endif
  trace.SerializeToString(content);
  return Status::OK();
}

ProfilerSession::ProfilerSession(const profiler::ProfilerOptions& options)
#if !defined(IS_MOBILE_PLATFORM)
    : active_(profiler::AcquireProfilerLock()),
#else
    : active_(false),
#endif
      start_time_ns_(EnvTime::NowNanos()) {
  if (!active_) {
#if !defined(IS_MOBILE_PLATFORM)
    status_ = tensorflow::Status(error::UNAVAILABLE,
                                 "Another profiler session is active.");
#else
    status_ =
        tensorflow::Status(error::UNIMPLEMENTED,
                           "Profiler is unimplemented for mobile platforms.");
#endif
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
