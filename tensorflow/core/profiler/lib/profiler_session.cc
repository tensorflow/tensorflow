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

#include <memory>

#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/profiler/convert/post_process_single_host_xplane.h"
#include "tensorflow/core/profiler/lib/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_lock.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#endif

namespace tensorflow {
namespace {

ProfileOptions GetOptions(const ProfileOptions& opts) {
  if (opts.version()) return opts;
  ProfileOptions options = ProfilerSession::DefaultOptions();
  options.set_include_dataset_ops(opts.include_dataset_ops());
  return options;
}

};  // namespace

/*static*/ std::unique_ptr<ProfilerSession> ProfilerSession::Create(
    const ProfileOptions& options) {
  return absl::WrapUnique(new ProfilerSession(GetOptions(options)));
}

tensorflow::Status ProfilerSession::Status() {
  mutex_lock l(mutex_);
  return status_;
}

Status ProfilerSession::CollectData(profiler::XSpace* space) {
  mutex_lock l(mutex_);
  if (!status_.ok()) return status_;
  LOG(INFO) << "Profiler session collecting data.";
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
  PostProcessSingleHostXSpace(space, start_time_ns_);
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

ProfilerSession::ProfilerSession(ProfileOptions options)
#if !defined(IS_MOBILE_PLATFORM)
    : active_(profiler::AcquireProfilerLock()),
#else
    : active_(false),
#endif
      options_(std::move(options)) {
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

  LOG(INFO) << "Profiler session initializing.";
  // Sleep until it is time to start profiling.
  const bool delayed_start = options_.start_timestamp_ns() > 0;
  if (delayed_start) {
    absl::Time scheduled_start =
        absl::FromUnixNanos(options_.start_timestamp_ns());
    auto now = absl::Now();
    if (scheduled_start < now) {
      LOG(WARNING) << "Profiling is late (" << now
                   << ") for the scheduled start (" << scheduled_start
                   << ") and will start immediately.";
    } else {
      absl::Duration sleep_duration = scheduled_start - now;
      LOG(INFO) << "Delaying start of profiler session by " << sleep_duration;
      absl::SleepFor(sleep_duration);
    }
  }

  LOG(INFO) << "Profiler session started.";
#if !defined(IS_MOBILE_PLATFORM)
  start_time_ns_ = profiler::GetCurrentTimeNanos();
  CreateProfilers(options_, &profilers_);
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
  LOG(INFO) << "Profiler session tear down.";
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
