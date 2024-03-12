/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_DEVICE_PROFILER_SESSION_H_
#define TENSORFLOW_CORE_PROFILER_LIB_DEVICE_PROFILER_SESSION_H_

#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/status.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/profiler/convert/xplane_to_step_stats.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#endif
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace tensorflow {

// Wraps a ProfilerSession configured to collect only device traces.
// Returns data in StepStats format.
class DeviceProfilerSession {
 public:
  // Creates a DeviceProfilerSession and starts tracing.
  // Traces GPU devices if present.
  // Does not trace TPU devices (not supported).
  static std::unique_ptr<DeviceProfilerSession> Create() {
#if !defined(IS_MOBILE_PLATFORM)
    ProfileOptions options = ProfilerSession::DefaultOptions();
    options.set_host_tracer_level(0);
    options.set_device_type(ProfileOptions::GPU);
    return absl::WrapUnique(new DeviceProfilerSession(options));
#else
    return nullptr;
#endif
  }

  // Stops tracing and converts the data to StepStats format.
  // Should be called at most once.
  Status CollectData(StepStats* step_stats) {
#if defined(IS_MOBILE_PLATFORM)
    return errors::Unimplemented("Profiling not supported on mobile platform.");
#else
    profiler::XSpace space;
    TF_RETURN_IF_ERROR(profiler_session_->CollectData(&space));
    profiler::ConvertGpuXSpaceToStepStats(space, step_stats);
    return absl::OkStatus();
#endif
  }

 private:
  // Constructs an instance of the class and starts profiling
  explicit DeviceProfilerSession(const ProfileOptions& options)
#if !defined(IS_MOBILE_PLATFORM)
      : profiler_session_(ProfilerSession::Create(options))
#endif
  {
  }

  // DeviceProfilerSession is neither copyable nor movable.
  DeviceProfilerSession(const DeviceProfilerSession&) = delete;
  DeviceProfilerSession& operator=(const DeviceProfilerSession&) = delete;

#if !defined(IS_MOBILE_PLATFORM)
  // TODO(b/256013238)
  std::unique_ptr<ProfilerSession> profiler_session_;
#endif
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_PROFILER_LIB_DEVICE_PROFILER_SESSION_H_
