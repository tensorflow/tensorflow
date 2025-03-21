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

#include "tsl/profiler/lib/profiler_session.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "xla/tsl/profiler/convert/post_process_single_host_xplane.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "tsl/platform/host_info.h"
#include "tsl/profiler/lib/profiler_collection.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/lib/profiler_lock.h"
#endif

namespace tsl {
namespace {

using tensorflow::ProfileOptions;
using tensorflow::profiler::XSpace;

ProfileOptions GetOptions(const ProfileOptions& opts) {
  if (opts.version()) return opts;
  ProfileOptions options = ProfilerSession::DefaultOptions();
  options.set_include_dataset_ops(opts.include_dataset_ops());
  return options;
}

};  // namespace

/*static*/ std::unique_ptr<ProfilerSession> ProfilerSession::Create(
    const ProfileOptions& options) {
  return absl::WrapUnique(new ProfilerSession(options));
}

absl::Status ProfilerSession::Status() {
  absl::MutexLock l(&mutex_);
  return status_;
}

#if !defined(IS_MOBILE_PLATFORM)
absl::Status ProfilerSession::CollectDataInternal(XSpace* space) {
  absl::MutexLock l(&mutex_);
  TF_RETURN_IF_ERROR(status_);
  LOG(INFO) << "Profiler session collecting data.";
  if (profilers_ != nullptr) {
    profilers_->Stop().IgnoreError();
    stop_time_ns_ = profiler::GetCurrentTimeNanos();
    profilers_->CollectData(space).IgnoreError();
    profilers_.reset();  // data has been collected.
  }
  // Allow another session to start.
  profiler_lock_.ReleaseIfActive();
  return absl::OkStatus();
}
#endif

absl::Status ProfilerSession::CollectData(XSpace* space) {
#if !defined(IS_MOBILE_PLATFORM)
  space->add_hostnames(port::Hostname());
  TF_RETURN_IF_ERROR(CollectDataInternal(space));
  profiler::PostProcessSingleHostXSpace(space, start_time_ns_, stop_time_ns_);
#endif
  return absl::OkStatus();
}

ProfilerSession::ProfilerSession(const ProfileOptions& options)
#if defined(IS_MOBILE_PLATFORM)
    : status_(errors::Unimplemented(
          "Profiler is unimplemented for mobile platforms.")) {
#else
    : options_(GetOptions(options)) {
  auto profiler_lock = profiler::ProfilerLock::Acquire();
  if (!profiler_lock.ok()) {
    status_ = profiler_lock.status();
    return;
  }
  profiler_lock_ = *std::move(profiler_lock);

  LOG(INFO) << "Profiler session initializing.";
  // Sleep until it is time to start profiling.
  if (options_.start_timestamp_ns() > 0) {
    int64_t sleep_duration_ns =
        options_.start_timestamp_ns() - profiler::GetCurrentTimeNanos();
    if (sleep_duration_ns < 0) {
      LOG(WARNING) << "Profiling is late by " << -sleep_duration_ns
                   << " nanoseconds and will start immediately.";
    } else {
      LOG(INFO) << "Delaying start of profiler session by "
                << sleep_duration_ns;
      profiler::SleepForNanos(sleep_duration_ns);
    }
  }

  LOG(INFO) << "Profiler session started.";
  start_time_ns_ = profiler::GetCurrentTimeNanos();

  DCHECK(profiler_lock_.Active());
  profilers_ = std::make_unique<tsl::profiler::ProfilerCollection>(
      profiler::CreateProfilers(options_));

  absl::Status status = profilers_->Start();
  if (options_.ignore_start_error()) {
    status.IgnoreError();
  } else {
    status_ = status;
  }
#endif
}

ProfilerSession::~ProfilerSession() {
#if !defined(IS_MOBILE_PLATFORM)
  LOG(INFO) << "Profiler session tear down.";
#endif
}

}  // namespace tsl
