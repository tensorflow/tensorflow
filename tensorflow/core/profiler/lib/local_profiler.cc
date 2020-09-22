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
#include "tensorflow/core/profiler/lib/local_profiler.h"

#include <memory>

#include "absl/memory/memory.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/post_process_single_host_xplane.h"
#include "tensorflow/core/profiler/internal/profiler_factory.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/profiler/lib/profiler_lock.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/derived_timeline.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace profiler {

/*static*/ std::unique_ptr<LocalProfiler> LocalProfiler::Create(
    const ProfileOptions& options, Status* out_status) {
  auto profiler = absl::WrapUnique(new LocalProfiler(options));
  Status status = profiler->Init();
  if (out_status) {
    *out_status = status;
  }
  if (!status.ok()) {
    LOG(ERROR) << status;
    return nullptr;
  }
  return profiler;
}

LocalProfiler::LocalProfiler(ProfileOptions options)
    : options_(std::move(options)) {}

LocalProfiler::~LocalProfiler() {
  mutex_lock lock(mutex_);

  for (auto& profiler : profilers_) {
    profiler->Stop().IgnoreError();
  }

  if (active_) {
    // Allow another LocalProfiler to be instantiated.
    ReleaseProfilerLock();
    active_ = false;
  }
}

Status LocalProfiler::Init() {
  mutex_lock lock(mutex_);
  VLOG(1) << "Creating a LocalProfiler.";

  bool active_ = AcquireProfilerLock();
  if (!active_) {
    return errors::Unavailable("Another LocalProfiler is active.");
  }

  CreateProfilers(options_, &profilers_);

  VLOG(1) << "LocalProfiler initialized with " << profilers_.size()
          << " profilers.";
  return Status::OK();
}

Status LocalProfiler::Start() {
  mutex_lock lock(mutex_);
  VLOG(1) << "Starting all profilers.";

  if (!active_) {
    return errors::FailedPrecondition("LocalProfiler is inactive.");
  }

  if (start_time_ns_ != 0) {
    return errors::FailedPrecondition("LocalProfiler is not restartable.");
  }

  start_time_ns_ = EnvTime::NowNanos();

  Status status;
  for (auto& profiler : profilers_) {
    Status start_status = profiler->Start();
    if (!start_status.ok()) {
      LOG(WARNING) << "Encountered error while starting profiler: "
                   << start_status.ToString();
    }
    status.Update(start_status);
  }

  VLOG(1) << "Started all profilers.";
  return status;
}

Status LocalProfiler::Stop() {
  mutex_lock lock(mutex_);
  VLOG(1) << "Stopping all profilers.";

  if (!active_) {
    return errors::FailedPrecondition("LocalProfiler is inactive.");
  }

  if (start_time_ns_ == 0) {
    return errors::FailedPrecondition(
        "LocalProfiler needs to Start() before it can stop producing data.");
  }

  Status status;
  for (auto& profiler : profilers_) {
    status.Update(profiler->Stop());
  }

  // Allow another LocalProfiler to be instantiated.
  if (active_) {
    ReleaseProfilerLock();
    active_ = false;
  }

  VLOG(1) << "Stopped all profilers.";
  return status;
}

Status LocalProfiler::CollectData(XSpace* space) {
  Status status;
  uint64 data_start_time_ns;

  {
    mutex_lock lock(mutex_);
    VLOG(1) << "Collecting data from " << profilers_.size() << " profilers.";

    if (!active_) {
      return errors::FailedPrecondition("LocalProfiler is inactive.");
    }

    if (start_time_ns_ != 0) {
      return errors::FailedPrecondition(
          "LocalProfiler needs to Stop() before collecting data.");
    }

    for (auto& profiler : profilers_) {
      VLOG(3) << "Collecting data from " << typeid(*profiler).name();
      status.Update(profiler->CollectData(space));
    }

    profilers_.clear();

    data_start_time_ns = start_time_ns_;
  }

  PostProcessSingleHostXSpace(space, data_start_time_ns);
  return status;
}

Status LocalProfiler::CollectData(RunMetadata* run_metadata) {
  return errors::Unimplemented(
      "Collecting profiler data into RunMetaData is unsupported.");
}

}  // namespace profiler
}  // namespace tensorflow
