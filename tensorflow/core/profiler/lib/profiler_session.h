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

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"

namespace tensorflow {

// A profiler which will start profiling when creating the object and will stop
// when either the object is destroyed or SerializedToString is called. It will
// profile all operations run under the given EagerContext.
// Multiple instances of it can be created, but at most one of them will profile
// for each EagerContext. Status() will return OK only for the instance that is
// profiling.
// Thread-safety: ProfilerSession is thread-safe.
class ProfilerSession {
 public:
  // Creates and ProfilerSession and starts profiling.
  static std::unique_ptr<ProfilerSession> Create(EagerContext* const context);

  // Deletes an exsiting Profiler and enables starting a new one.
  ~ProfilerSession();

  tensorflow::Status Status() LOCKS_EXCLUDED(mutex_);

  tensorflow::Status SerializeToString(string* content) LOCKS_EXCLUDED(mutex_);

 private:
  // Constructs an instance of the class and starts profiling
  explicit ProfilerSession(EagerContext* const context);

  // Profiler is neither copyable or movable.
  ProfilerSession(const ProfilerSession&) = delete;
  ProfilerSession& operator=(const ProfilerSession&) = delete;

  std::vector<std::unique_ptr<tensorflow::profiler::ProfilerInterface>>
      profilers_ GUARDED_BY(mutex_);

  tensorflow::Status status_ GUARDED_BY(mutex_);
  const uint64 start_time_micros_;
  mutex mutex_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_PROFILER_LIB_PROFILER_SESSION_H_
