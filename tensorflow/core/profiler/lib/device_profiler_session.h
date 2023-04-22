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
#include "tensorflow/core/profiler/lib/profiler_session.h"
#endif
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// Wraps a ProfilerSession configured to collect only device traces.
// Returns data in StepStats format.
class DeviceProfilerSession {
 public:
  // Creates a DeviceProfilerSession and starts tracing.
  // If gpu_only is true, traces a GPU device but not a TPU device.
  static std::unique_ptr<DeviceProfilerSession> Create(bool gpu_only = false) {
#if !defined(IS_MOBILE_PLATFORM)
    ProfileOptions options = ProfilerSession::DefaultOptions();
    options.set_host_tracer_level(0);
    if (gpu_only) options.set_device_type(ProfileOptions::GPU);
    return absl::WrapUnique(new DeviceProfilerSession(options));
#else
    return nullptr;
#endif
  }

  // Stops tracing and converts the data to StepStats format.
  // Should be called at most once.
  Status CollectData(StepStats* step_stats) {
#if !defined(IS_MOBILE_PLATFORM)
    RunMetadata run_metadata;
    Status status = profiler_session_.CollectData(&run_metadata);
    step_stats->MergeFrom(run_metadata.step_stats());
    return status;
#else
    return errors::Unimplemented("Profiling not supported on mobile platform.");
#endif
  }

 private:
  // Constructs an instance of the class and starts profiling
  explicit DeviceProfilerSession(const ProfileOptions& options)
#if !defined(IS_MOBILE_PLATFORM)
      : profiler_session_(options)
#endif
  {
  }

  // DeviceProfilerSession is neither copyable or movable.
  DeviceProfilerSession(const DeviceProfilerSession&) = delete;
  DeviceProfilerSession& operator=(const DeviceProfilerSession&) = delete;

#if !defined(IS_MOBILE_PLATFORM)
  ProfilerSession profiler_session_;
#endif
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_PROFILER_LIB_DEVICE_PROFILER_SESSION_H_
