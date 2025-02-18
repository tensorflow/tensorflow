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
#ifndef TENSORFLOW_TSL_PROFILER_LIB_PROFILER_SESSION_H_
#define TENSORFLOW_TSL_PROFILER_LIB_PROFILER_SESSION_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/lib/profiler_lock.h"
#endif

namespace tsl {

// A profiler which will start profiling when creating the object and will stop
// when either the object is destroyed or CollectData is called.
// Multiple instances can be created, but at most one of them will profile.
// Status() will return OK only for the instance that is profiling.
// Thread-safety: ProfilerSession is thread-safe.
class ProfilerSession {
 public:
  // Creates a ProfilerSession and starts profiling.
  static std::unique_ptr<ProfilerSession> Create(
    const tensorflow::ProfileOptions& options);

  static tensorflow::ProfileOptions DefaultOptions() {
    tensorflow::ProfileOptions options;
    options.set_version(1);
    options.set_device_tracer_level(1);
    options.set_host_tracer_level(2);
    options.set_device_type(tensorflow::ProfileOptions::UNSPECIFIED);
    options.set_python_tracer_level(0);
    options.set_enable_hlo_proto(true);
    options.set_include_dataset_ops(true);
    return options;
  }

  // Deletes an existing Profiler and enables starting a new one.
  ~ProfilerSession();

  absl::Status Status() TF_LOCKS_EXCLUDED(mutex_);

  // Collects profile data into XSpace.
  absl::Status CollectData(tensorflow::profiler::XSpace* space)
      TF_LOCKS_EXCLUDED(mutex_);

 private:
  // Constructs an instance of the class and starts profiling
  explicit ProfilerSession(const tensorflow::ProfileOptions& options);

  // ProfilerSession is neither copyable or movable.
  ProfilerSession(const ProfilerSession&) = delete;
  ProfilerSession& operator=(const ProfilerSession&) = delete;

#if !defined(IS_MOBILE_PLATFORM)
  // Collects profile data into XSpace without post-processsing.
  absl::Status CollectDataInternal(tensorflow::profiler::XSpace* space);

  profiler::ProfilerLock profiler_lock_ TF_GUARDED_BY(mutex_);

  std::unique_ptr<profiler::ProfilerInterface> profilers_ TF_GUARDED_BY(mutex_);

  uint64 start_time_ns_;
  uint64 stop_time_ns_;
  tensorflow::ProfileOptions options_;
#endif
  absl::Status status_ TF_GUARDED_BY(mutex_);
  mutex mutex_;
};

}  // namespace tsl
#endif  // TENSORFLOW_TSL_PROFILER_LIB_PROFILER_SESSION_H_
