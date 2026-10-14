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

#ifndef TENSORFLOW_TSL_PROFILER_LIB_PROFILER_PASSES_H_
#define TENSORFLOW_TSL_PROFILER_LIB_PROFILER_PASSES_H_

#include <cstdint>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

#if !defined(IS_MOBILE_PLATFORM)
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/lib/profiler_lock.h"
#endif

namespace tsl {

// ProfilerPasses orchestrates multi-pass profiling for workloads that require
// programmatic replay across multiple iterations (e.g. collecting different
// sets of hardware performance counters or metrics across repeated passes).
//
// Like its counterpart `ProfilerSession` (see `profiler_session.h`), it starts
// profiling upon creation and holds a profiler lock so that at most one
// session profiles at a time. Unlike `ProfilerSession`, which performs a single
// continuous profiling session across its lifetime, `ProfilerPasses` supports
// stepping through passes via `StartPass()`, `StopPass()`, and
// `NeedMorePasses()`, and delimiting ranges with `PushRange()` / `PopRange()`.
// Profile data from all passes is aggregated into an XSpace when
// `CollectData()` is called.
class ProfilerPasses {
 public:
  // Creates a ProfilerPasses and starts profiling.
  static std::unique_ptr<ProfilerPasses> Create(
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
    options.set_enable_multipass(true);
    return options;
  }

  ~ProfilerPasses();

  absl::Status Status() ABSL_LOCKS_EXCLUDED(mutex_);

  bool NeedMorePasses() ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status StartPass() ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status PushRange(absl::string_view name) ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status PopRange() ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status StopPass() ABSL_LOCKS_EXCLUDED(mutex_);

  // Collects profile data into XSpace.
  absl::Status CollectData(tensorflow::profiler::XSpace* space)
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  explicit ProfilerPasses(const tensorflow::ProfileOptions& options);

  ProfilerPasses(const ProfilerPasses&) = delete;
  ProfilerPasses& operator=(const ProfilerPasses&) = delete;

#if !defined(IS_MOBILE_PLATFORM)
  absl::Status CollectDataInternal(tensorflow::profiler::XSpace* space);

  profiler::ProfilerLock profiler_lock_ ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<profiler::MultiPassProfilerInterface> multi_pass_profiler_
      ABSL_GUARDED_BY(mutex_);
#endif

  absl::Mutex mutex_;
  absl::Status status_ ABSL_GUARDED_BY(mutex_);
  tensorflow::ProfileOptions options_;
  int pass_count_ ABSL_GUARDED_BY(mutex_) = 0;
  int64_t first_pass_start_time_ns_ = 0;
  int64_t first_pass_stop_time_ns_ = 0;
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_PROFILER_PASSES_H_
