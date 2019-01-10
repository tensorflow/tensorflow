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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_PROFILER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_PROFILER_H_

#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// A profiler which will start profiling when creating the object and will stop
// when the object is destroyed. It will profile all operations run under the
// given EagerContext.
// Multiple instances of it can be created, but at most one of them will profile
// for each EagerContext. Status() will return OK only for the instance that is
// profiling.
// Thread-safety: TFE_Profiler is thread-safe.
class EagerProfiler : RunMetadataListener {
 public:
  // Creates and EagerProfiler and starts profiling.
  static std::unique_ptr<EagerProfiler> Create(EagerContext* const context);

  // Deletes an exsiting Profiler and enables starting a new one.
  ~EagerProfiler() override;

  void BeforeClearRunMetadata() override LOCKS_EXCLUDED(mutex_)
      EXCLUSIVE_LOCKS_REQUIRED(context_->MetadataMu());
  tensorflow::Status Status() LOCKS_EXCLUDED(mutex_);

  tensorflow::Status SerializeToString(string* content) LOCKS_EXCLUDED(mutex_);

 private:
  // Constructs an instance of the class and starts profiling
  explicit EagerProfiler(EagerContext* const context);

  // Profiler is neither copyable or movable.
  EagerProfiler(const EagerProfiler&) = delete;
  EagerProfiler& operator=(const EagerProfiler&) = delete;

  void GetMergetRunMetadata(RunMetadata* metadata) LOCKS_EXCLUDED(mutex_);

  RunMetadata run_metadata_ GUARDED_BY(mutex_);
  tensorflow::Status status_ GUARDED_BY(mutex_);
  EagerContext* const context_;
  mutex mutex_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_PROFILER_H_
