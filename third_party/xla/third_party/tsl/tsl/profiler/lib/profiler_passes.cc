/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/profiler/lib/profiler_passes.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "xla/tsl/platform/env.h"
#include "xla/tsl/profiler/convert/post_process_single_host_xplane.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "tsl/platform/host_info.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_lock.h"
#endif

namespace tsl {
namespace {

using tensorflow::ProfileOptions;
using tensorflow::profiler::XSpace;
using ::tsl::profiler::XPlaneBuilder;

void SetProfileOptionsIntoSpace(const ProfileOptions& options, XSpace* space) {
  XPlaneBuilder xplane(profiler::FindOrAddMutablePlaneWithName(
      space, tsl::profiler::kTaskEnvPlaneName));
  xplane.AddStatValue(
      *xplane.GetOrCreateStatMetadata(tsl::profiler::GetTaskEnvStatTypeStr(
          tsl::profiler::kEnvProfileOptions)),
      options);
}

}  // namespace

std::unique_ptr<ProfilerPasses> ProfilerPasses::Create(
    const ProfileOptions& options) {
  return absl::WrapUnique(new ProfilerPasses(options));
}

absl::Status ProfilerPasses::Status() {
  absl::MutexLock l(&mutex_);
  return status_;
}

bool ProfilerPasses::NeedMorePasses() {
  absl::MutexLock l(&mutex_);
  if (!status_.ok()) return false;
  if (multi_pass_profiler_ == nullptr) {
    return false;
  }
  return multi_pass_profiler_->NeedMorePasses();
}

absl::Status ProfilerPasses::StartPass() {
  absl::MutexLock l(&mutex_);
  if (!status_.ok()) {
    return absl::InternalError("Previous operations failed");
  }
  if (multi_pass_profiler_ == nullptr) {
    return absl::OkStatus();
  }

  if (pass_count_++ == 0) {
    first_pass_start_time_ns_ = profiler::GetCurrentTimeNanos();
    absl::Status status = multi_pass_profiler_->Start();
    if (options_.raise_error_on_start_failure()) {
      status_ = status;
    } else {
      status.IgnoreError();
    }
    RETURN_IF_ERROR(status_);

    status = multi_pass_profiler_->StartPass();
    if (options_.raise_error_on_start_failure()) {
      status_ = status;
    } else {
      status.IgnoreError();
    }
    return status_;
  }

  // passes after first pass.
  return status_ = multi_pass_profiler_->StartPass();
}

absl::Status ProfilerPasses::PushRange(absl::string_view name) {
  absl::MutexLock l(&mutex_);
  RETURN_IF_ERROR(status_);
  if (multi_pass_profiler_ != nullptr) {
    return status_ = multi_pass_profiler_->PushRange(name);
  }
  return absl::OkStatus();
}

absl::Status ProfilerPasses::PopRange() {
  absl::MutexLock l(&mutex_);
  RETURN_IF_ERROR(status_);
  if (multi_pass_profiler_ != nullptr) {
    return status_ = multi_pass_profiler_->PopRange();
  }
  return absl::OkStatus();
}

absl::Status ProfilerPasses::StopPass() {
  absl::MutexLock l(&mutex_);
  if (pass_count_ == 1) {
    first_pass_stop_time_ns_ = profiler::GetCurrentTimeNanos();
  }
  if (multi_pass_profiler_ != nullptr) {
    status_.Update(multi_pass_profiler_->StopPass());
  }
  return status_;
}

#if !defined(IS_MOBILE_PLATFORM)
absl::Status ProfilerPasses::CollectDataInternal(XSpace* space) {
  absl::MutexLock l(&mutex_);
  RETURN_IF_ERROR(status_);
  VLOG(3) << "Profiler passes collecting data.";
  if (multi_pass_profiler_ != nullptr) {
    multi_pass_profiler_->Stop().IgnoreError();
    multi_pass_profiler_->CollectData(space).IgnoreError();
    multi_pass_profiler_.reset();
  }
  // Allow another session to start.
  profiler_lock_.ReleaseIfActive();
  return absl::OkStatus();
}
#endif

absl::Status ProfilerPasses::CollectData(XSpace* space) {
#if !defined(IS_MOBILE_PLATFORM)
  space->add_hostnames(port::Hostname());
  RETURN_IF_ERROR(CollectDataInternal(space));
  profiler::SetXSpacePidIfNotSet(*space, tsl::Env::Default()->GetProcessId());
  profiler::PostProcessSingleHostXSpace(space, first_pass_start_time_ns_,
                                        first_pass_stop_time_ns_);
#endif
  SetProfileOptionsIntoSpace(options_, space);
  return absl::OkStatus();
}

ProfilerPasses::ProfilerPasses(const ProfileOptions& options)
#if defined(IS_MOBILE_PLATFORM)
    : status_(absl::UnimplementedError(
          "Profiler is unimplemented for mobile platforms.")) {
#else
    : options_(options) {
  auto profiler_lock = profiler::ProfilerLock::Acquire();
  if (!profiler_lock.ok()) {
    status_ = profiler_lock.status();
    return;
  }
  profiler_lock_ = *std::move(profiler_lock);

  DCHECK(profiler_lock_.Active());
  VLOG(3) << "Profiler passes initializing. options.enable_multipass = "
          << options_.enable_multipass();
  auto multi_pass_profilers = profiler::CreateMultiPassProfilers(options_);
  if (!multi_pass_profilers.empty()) {
    multi_pass_profiler_ = std::move(multi_pass_profilers[0]);
    if (multi_pass_profilers.size() > 1) {
      LOG(ERROR)
          << "More than one multipass-planner found, only use one of them.";
    }
    VLOG(3) << "Found one multipass planner!";
  } else {
    VLOG(3) << "No multipass planner found!";
  }
#endif
}

ProfilerPasses::~ProfilerPasses() {
#if !defined(IS_MOBILE_PLATFORM)
  VLOG(3) << "Profiler passes tear down.";
#endif
}

}  // namespace tsl
