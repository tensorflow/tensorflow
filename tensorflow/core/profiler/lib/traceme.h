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

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/internal/traceme_recorder.h"

namespace tensorflow {
namespace profiler {

// This is specifically used for instrumenting Tensorflow ops.
// Takes input as whether a TF op is expensive or not and returns the TraceMe
// level to be assigned to trace that particular op. Assigns level 2 for
// expensive ops (these are high-level details and shown by default in profiler
// UI). Assigns level 3 for cheap ops (low-level details not shown by default).
inline int GetTFTraceMeLevel(bool is_expensive) { return is_expensive ? 2 : 3; }

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
class TraceMe {
 public:
  // Constructor that traces a user-defined activity labeled with activity_name
  // in the UI. Level defines the trace priority, used for filtering TraceMe
  // events. By default, traces with TraceMe level <= 2 are recorded. Levels:
  // - Must be a positive integer.
  // - Can be a value in enum TraceMeLevel.
  // Users are welcome to use level > 3 in their code, if they wish to filter
  // out their host traces based on verbosity.
  explicit TraceMe(absl::string_view activity_name, int level = 1) {
    DCHECK_GE(level, 1);
    if (TraceMeRecorder::Active(level)) {
      new (&no_init_.name) string(activity_name);
      start_time_ = EnvTime::Default()->NowNanos();
    } else {
      start_time_ = kUntracedActivity;
    }
  }

  // string&& constructor to prevent an unnecessary string copy, e.g. when a
  // TraceMe is constructed based on the result of a StrCat operation.
  // Note: We can't take the string by value because a) it would make the
  // overloads ambiguous, and b) we want lvalue strings to use the string_view
  // constructor so we avoid copying them when tracing is disabled.
  explicit TraceMe(string &&activity_name, int level = 1) {
    DCHECK_GE(level, 1);
    if (TraceMeRecorder::Active(level)) {
      new (&no_init_.name) string(std::move(activity_name));
      start_time_ = EnvTime::Default()->NowNanos();
    } else {
      start_time_ = kUntracedActivity;
    }
  }

  // Do not allow passing strings by reference or value since the caller
  // may unintentionally maintain ownership of the activity_name.
  // Explicitly std::move the activity_name or wrap it in a string_view if
  // you really wish to maintain ownership.
  explicit TraceMe(const string &activity_name, int level = 1) = delete;

  // This overload is necessary to make TraceMe's with string literals work.
  // Otherwise, the string&& and the string_view constructor would be equally
  // good overload candidates.
  explicit TraceMe(const char *raw, int level = 1)
      : TraceMe(absl::string_view(raw), level) {}

  // This overload only generates the activity name if tracing is enabled.
  // Useful for avoiding things like string concatenation when tracing is
  // disabled. The |name_generator| may be a lambda or functor that returns a
  // type that the string() constructor can take.
  // name_generator is templated, rather than a std::function to avoid
  // allocations std::function might make even if never called.
  // Usage: profiler::TraceMe([&]{ return StrCat(prefix, ":", postfix); });
  template <typename NameGeneratorT>
  explicit TraceMe(NameGeneratorT name_generator, int level = 1) {
    DCHECK_GE(level, 1);
    if (TraceMeRecorder::Active(level)) {
      new (&no_init_.name) string(name_generator());
      start_time_ = EnvTime::Default()->NowNanos();
    } else {
      start_time_ = kUntracedActivity;
    }
  }

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
    if (start_time_ != kUntracedActivity) {
      if (TraceMeRecorder::Active()) {
        TraceMeRecorder::Record({kCompleteActivity, std::move(no_init_.name),
                                 start_time_, EnvTime::Default()->NowNanos()});
      }
      no_init_.name.~string();
      start_time_ = kUntracedActivity;
    }
  }

  ~TraceMe() { Stop(); }

  // TraceMe is not movable or copyable.
  TraceMe(const TraceMe &) = delete;
  TraceMe &operator=(const TraceMe &) = delete;

  // Static API, for use when scoped objects are inconvenient.

  // Record the start time of an activity.
  // Returns the activity ID, which is used to stop the activity.
  static uint64 ActivityStart(absl::string_view name, int level = 1) {
    return TraceMeRecorder::Active(level) ? ActivityStartImpl(name)
                                          : kUntracedActivity;
  }

  // Record the end time of an activity started by ActivityStart().
  static void ActivityEnd(uint64 activity_id) {
    // We don't check the level again (see ~TraceMe()).
    if (activity_id != kUntracedActivity) {
      if (TraceMeRecorder::Active()) {
        ActivityEndImpl(activity_id);
      }
    }
  }

 private:
  // Activity ID or start time used when tracing is disabled.
  constexpr static uint64 kUntracedActivity = 0;
  // Activity ID used as a placeholder when both start and end are present.
  constexpr static uint64 kCompleteActivity = 1;

  static uint64 ActivityStartImpl(absl::string_view activity_name);
  static void ActivityEndImpl(uint64 activity_id);

  // Wrap the name into a union so that we can avoid the cost of string
  // initialization when tracing is disabled.
  union NoInit {
    NoInit() {}
    ~NoInit() {}
    string name;
  } no_init_;

  uint64 start_time_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_LIB_TRACEME_H_
