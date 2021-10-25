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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_PROFILER_SESSION_H_
#define TENSORFLOW_CORE_PROFILER_LIB_PROFILER_SESSION_H_

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {

// A profiler which will start profiling when creating the object and will stop
// when either the object is destroyed or CollectData is called.
// Multiple instances can be created, but at most one of them will profile.
// Status() will return OK only for the instance that is profiling.
// Thread-safety: ProfilerSession is thread-safe.
class ProfilerSession {
 public:
  // Creates a ProfilerSession and starts profiling.
  static std::unique_ptr<ProfilerSession> Create(const ProfileOptions& options);

  static ProfileOptions DefaultOptions() {
    ProfileOptions options;
    options.set_version(1);
    options.set_device_tracer_level(1);
    options.set_host_tracer_level(2);
    options.set_device_type(ProfileOptions::UNSPECIFIED);
    options.set_python_tracer_level(0);
    options.set_enable_hlo_proto(false);
    options.set_include_dataset_ops(true);
    return options;
  }

  // Deletes an existing Profiler and enables starting a new one.
  ~ProfilerSession();

  tensorflow::Status Status() TF_LOCKS_EXCLUDED(mutex_);

  // Collects profile data into XSpace.
  tensorflow::Status CollectData(profiler::XSpace* space)
      TF_LOCKS_EXCLUDED(mutex_);

 private:
  friend class DeviceProfilerSession;

  // Constructs an instance of the class and starts profiling
  explicit ProfilerSession(const ProfileOptions& options);

  // ProfilerSession is neither copyable or movable.
  ProfilerSession(const ProfilerSession&) = delete;
  ProfilerSession& operator=(const ProfilerSession&) = delete;

  // Collects profile data into XSpace without post-processsing.
  tensorflow::Status CollectDataInternal(profiler::XSpace* space);

  std::vector<std::unique_ptr<profiler::ProfilerInterface>> profilers_
      TF_GUARDED_BY(mutex_);

  // True if the session is active.
  bool active_ TF_GUARDED_BY(mutex_);

  tensorflow::Status status_ TF_GUARDED_BY(mutex_);
  uint64 start_time_ns_;
  mutex mutex_;
  ProfileOptions options_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_PROFILER_LIB_PROFILER_SESSION_H_
