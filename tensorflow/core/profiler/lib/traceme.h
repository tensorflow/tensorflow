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
#ifndef TENSORFLOW_CORE_PROFILER_LIB_TRACEME_H_
#define TENSORFLOW_CORE_PROFILER_LIB_TRACEME_H_

#include <new>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"  // IWYU pragma: export

#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/profiler/internal/cpu/traceme_recorder.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#endif

namespace tensorflow {
namespace profiler {

// Predefined levels:
// - Level 1 (kCritical) is the default and used only for user instrumentation.
// - Level 2 (kInfo) is used by profiler for instrumenting high level program
//   execution details (expensive TF ops, XLA ops, etc).
// - Level 3 (kVerbose) is also used by profiler to instrument more verbose
//   (low-level) program execution details (cheap TF ops, etc).
enum TraceMeLevel {
  kCritical = 1,
  kInfo = 2,
  kVerbose = 3,
};

// This is specifically used for instrumenting Tensorflow ops.
// Takes input as whether a TF op is expensive or not and returns the TraceMe
// level to be assigned to trace that particular op. Assigns level 2 for
// expensive ops (these are high-level details and shown by default in profiler
// UI). Assigns level 3 for cheap ops (low-level details not shown by default).
inline int GetTFTraceMeLevel(bool is_expensive) {
  return is_expensive ? kInfo : kVerbose;
}

// This class permits user-specified (CPU) tracing activities. A trace activity
// is started when an object of this class is created and stopped when the
// object is destroyed.
//
// CPU tracing can be useful when trying to understand what parts of GPU
// computation (e.g., kernels and memcpy) correspond to higher level activities
// in the overall program. For instance, a collection of kernels maybe
// performing one "step" of a program that is better visualized together than
// interspersed with kernels from other "steps". Therefore, a TraceMe object
// can be created at each "step".
//
// Two APIs are provided:
//   (1) Scoped object: a TraceMe object starts tracing on construction, and
//       stops tracing when it goes out of scope.
//          {
//            TraceMe trace("step");
//            ... do some work ...
//          }
//       TraceMe objects can be members of a class, or allocated on the heap.
//   (2) Static methods: ActivityStart and ActivityEnd may be called in pairs.
//          auto id = ActivityStart("step");
//          ... do some work ...
//          ActivityEnd(id);
//       The two static methods should be called within the same thread.
class TraceMe {
 public:
  // Constructor that traces a user-defined activity labeled with name
  // in the UI. Level defines the trace priority, used for filtering TraceMe
  // events. By default, traces with TraceMe level <= 2 are recorded. Levels:
  // - Must be a positive integer.
  // - Can be a value in enum TraceMeLevel.
  // Users are welcome to use level > 3 in their code, if they wish to filter
  // out their host traces based on verbosity.
  explicit TraceMe(absl::string_view name, int level = 1) {
    DCHECK_GE(level, 1);
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(TraceMeRecorder::Active(level))) {
      new (&no_init_.name) std::string(name);
      start_time_ = GetCurrentTimeNanos();
    }
#endif
  }

  // Do not allow passing a temporary string as the overhead of generating that
  // string should only be incurred when tracing is enabled. Wrap the temporary
  // string generation (e.g., StrCat) in a lambda and use the name_generator
  // template instead.
  explicit TraceMe(std::string&& name, int level = 1) = delete;

  // Do not allow passing strings by reference or value since the caller
  // may unintentionally maintain ownership of the name.
  // Explicitly wrap the name in a string_view if you really wish to maintain
  // ownership of a string already generated for other purposes. For temporary
  // strings (e.g., result of StrCat) use the name_generator template.
  explicit TraceMe(const std::string& name, int level = 1) = delete;

  // This overload is necessary to make TraceMe's with string literals work.
  // Otherwise, the name_generator template would be used.
  explicit TraceMe(const char* raw, int level = 1)
      : TraceMe(absl::string_view(raw), level) {}

  // This overload only generates the name (and possibly metadata) if tracing is
  // enabled. Useful for avoiding expensive operations (e.g., string
  // concatenation) when tracing is disabled.
  // name_generator may be a lambda or functor that returns a type that the
  // string() constructor can take, e.g., the result of TraceMeEncode.
  // name_generator is templated, rather than a std::function to avoid
  // allocations std::function might make even if never called.
  // Example Usage:
  //   TraceMe trace_me([&]() {
  //     return StrCat("my_trace", id);
  //   }
  //   TraceMe op_trace_me([&]() {
  //     return TraceMeOp(op_name, op_type);
  //   }
  //   TraceMe trace_me_with_metadata([&value1]() {
  //     return TraceMeEncode("my_trace", {{"key1", value1}, {"key2", 42}});
  //   });
  template <typename NameGeneratorT>
  explicit TraceMe(NameGeneratorT name_generator, int level = 1) {
    DCHECK_GE(level, 1);
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(TraceMeRecorder::Active(level))) {
      new (&no_init_.name) std::string(name_generator());
      start_time_ = GetCurrentTimeNanos();
    }
#endif
  }

  // Movable.
  TraceMe(TraceMe&& other) { *this = std::move(other); }
  TraceMe& operator=(TraceMe&& other) {
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(other.start_time_ != kUntracedActivity)) {
      new (&no_init_.name) std::string(std::move(other.no_init_.name));
      other.no_init_.name.~string();
      start_time_ = std::exchange(other.start_time_, kUntracedActivity);
    }
#endif
    return *this;
  }

  ~TraceMe() { Stop(); }

  // Stop tracing the activity. Called by the destructor, but exposed to allow
  // stopping tracing before the object goes out of scope. Only has an effect
  // the first time it is called.
  void Stop() {
    // We do not need to check the trace level again here.
    // - If tracing wasn't active to start with, we have kUntracedActivity.
    // - If tracing was active and was stopped, we have
    //   TraceMeRecorder::Active().
    // - If tracing was active and was restarted at a lower level, we may
    //   spuriously record the event. This is extremely rare, and acceptable as
    //   event will be discarded when its start timestamp fall outside of the
    //   start/stop session timestamp.
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(start_time_ != kUntracedActivity)) {
      if (TF_PREDICT_TRUE(TraceMeRecorder::Active())) {
        TraceMeRecorder::Record(
            {std::move(no_init_.name), start_time_, GetCurrentTimeNanos()});
      }
      no_init_.name.~string();
      start_time_ = kUntracedActivity;
    }
#endif
  }

  // Appends new_metadata to the TraceMe name passed to the constructor.
  // metadata_generator may be a lambda or functor that returns a type that the
  // string() constructor can take, e.g., the result of TraceMeEncode.
  // metadata_generator is only evaluated when tracing is enabled.
  // metadata_generator is templated, rather than a std::function to avoid
  // allocations std::function might make even if never called.
  // Example Usage:
  //   trace_me.AppendMetadata([&value1]() {
  //     return TraceMeEncode({{"key1", value1}, {"key2", 42}});
  //   });
  template <typename MetadataGeneratorT>
  void AppendMetadata(MetadataGeneratorT metadata_generator) {
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(start_time_ != kUntracedActivity)) {
      if (TF_PREDICT_TRUE(TraceMeRecorder::Active())) {
        traceme_internal::AppendMetadata(&no_init_.name, metadata_generator());
      }
    }
#endif
  }

  // Static API, for use when scoped objects are inconvenient.

  // Record the start time of an activity.
  // Returns the activity ID, which is used to stop the activity.
  // Calls `name_generator` to get the name for activity.
  template <typename NameGeneratorT>
  static int64 ActivityStart(NameGeneratorT name_generator, int level = 1) {
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(TraceMeRecorder::Active(level))) {
      int64 activity_id = TraceMeRecorder::NewActivityId();
      TraceMeRecorder::Record(
          {name_generator(), GetCurrentTimeNanos(), -activity_id});
      return activity_id;
    }
#endif
    return kUntracedActivity;
  }

  // Record the start time of an activity.
  // Returns the activity ID, which is used to stop the activity.
  static int64 ActivityStart(absl::string_view name, int level = 1) {
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(TraceMeRecorder::Active(level))) {
      int64 activity_id = TraceMeRecorder::NewActivityId();
      TraceMeRecorder::Record(
          {std::string(name), GetCurrentTimeNanos(), -activity_id});
      return activity_id;
    }
#endif
    return kUntracedActivity;
  }

  // Same as ActivityStart above, an overload for "const std::string&"
  static int64 ActivityStart(const std::string& name, int level = 1) {
    return ActivityStart(absl::string_view(name), level);
  }

  // Same as ActivityStart above, an overload for "const char*"
  static int64 ActivityStart(const char* name, int level = 1) {
    return ActivityStart(absl::string_view(name), level);
  }

  // Record the end time of an activity started by ActivityStart().
  static void ActivityEnd(int64 activity_id) {
#if !defined(IS_MOBILE_PLATFORM)
    // We don't check the level again (see TraceMe::Stop()).
    if (TF_PREDICT_FALSE(activity_id != kUntracedActivity)) {
      if (TF_PREDICT_TRUE(TraceMeRecorder::Active())) {
        TraceMeRecorder::Record(
            {std::string(), -activity_id, GetCurrentTimeNanos()});
      }
    }
#endif
  }

  // Records the time of an instant activity.
  template <typename NameGeneratorT>
  static void InstantActivity(NameGeneratorT name_generator, int level = 1) {
#if !defined(IS_MOBILE_PLATFORM)
    if (TF_PREDICT_FALSE(TraceMeRecorder::Active(level))) {
      int64 now = GetCurrentTimeNanos();
      TraceMeRecorder::Record(
          {name_generator(), /*start_time=*/now, /*end_time=*/now});
    }
#endif
  }

  static bool Active(int level = 1) {
#if !defined(IS_MOBILE_PLATFORM)
    return TraceMeRecorder::Active(level);
#else
    return false;
#endif
  }

  static int64 NewActivityId() {
#if !defined(IS_MOBILE_PLATFORM)
    return TraceMeRecorder::NewActivityId();
#else
    return 0;
#endif
  }

 private:
  // Start time used when tracing is disabled.
  constexpr static int64 kUntracedActivity = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(TraceMe);

  // Wrap the name into a union so that we can avoid the cost of string
  // initialization when tracing is disabled.
  union NoInit {
    NoInit() {}
    ~NoInit() {}
    std::string name;
  } no_init_;

  int64 start_time_ = kUntracedActivity;
};

// Whether OpKernel::TraceString will populate additional information for
// profiler, such as tensor shapes.
inline bool TfOpDetailsEnabled() {
  return TraceMe::Active(TraceMeLevel::kVerbose);
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_TRACEME_H_
