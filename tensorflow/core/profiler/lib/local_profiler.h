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

#ifndef TENSORFLOW_CORE_PROFILER_LIB_LOCAL_PROFILER_H_
#define TENSORFLOW_CORE_PROFILER_LIB_LOCAL_PROFILER_H_

#include <memory>
#include <vector>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace profiler {

// LocalProfiler encapsulates multiple profiler backends that each implements.
// ProfilerInterface.
// Thread-safety: LocalProfiler is thread-safe.
class LocalProfiler : public ProfilerInterface {
 public:
  // Instantiates a LocalProfiler if there is not one already active.
  // Returns null on errors, which will be indicated by the Status code.
  static std::unique_ptr<LocalProfiler> Create(const ProfileOptions& options,
                                               Status* status);

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

  // Starts all profilers.
  Status Start() override TF_LOCKS_EXCLUDED(mutex_);

  // Stops all profilers.
  Status Stop() override TF_LOCKS_EXCLUDED(mutex_);

  // Collects data from all profilers into XSpace. Post-process the XSpace
  // (e.g., groups trace events per step). This is best effort profiling and
  //  XSpace may contain data collected before any errors occurred.
  Status CollectData(XSpace* space) override TF_LOCKS_EXCLUDED(mutex_);

  // Unimplemented, do not use. This will be deprecated in future.
  Status CollectData(RunMetadata* run_metadata) override;

  // Deletes an existing Profiler and enables starting a new one.
  ~LocalProfiler() override;

 private:
  // Constructs an instance of the class and starts profiling
  explicit LocalProfiler(ProfileOptions options);

  // Neither copyable or movable.
  LocalProfiler(const LocalProfiler&) = delete;
  LocalProfiler& operator=(const LocalProfiler&) = delete;

  // Initializes LocalProfiler and sets ups all profilers.
  Status Init();

  mutex mutex_;

  std::vector<std::unique_ptr<ProfilerInterface>> profilers_
      TF_GUARDED_BY(mutex_);

  // True if the LocalProfiler is active.
  bool active_ TF_GUARDED_BY(mutex_) = false;

  // Time when Start() was called.
  uint64 start_time_ns_ TF_GUARDED_BY(mutex_) = 0;

  ProfileOptions options_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_LOCAL_PROFILER_H_
