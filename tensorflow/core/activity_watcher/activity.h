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
#ifndef TENSORFLOW_CORE_ACTIVITY_WATCHER_ACTIVITY_H_
#define TENSORFLOW_CORE_ACTIVITY_WATCHER_ACTIVITY_H_

#include <atomic>
#include <functional>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
class CoordinationServiceAgent;

namespace activity_watcher {

using ActivityId = uint64;
constexpr ActivityId kActivityNotRecorded = 0;
constexpr int kWatcherDisabled = 0;

enum ActivityCategory {
  kCollective = 0,
  kRemoteFunction = 1,
  kMisc = 2,
};

static tensorflow::string ToString(ActivityCategory category) {
  switch (category) {
    case ActivityCategory::kCollective:
      return "Collective";
    case ActivityCategory::kRemoteFunction:
      return "Remote Function";
    case ActivityCategory::kMisc:
      return "Miscellaneous";
  }
}

// An activity to be recorded.
struct Activity {
  using Attributes =
      absl::flat_hash_map<tensorflow::string, tensorflow::string>;
  // A human readable title of the activity.
  tensorflow::string title;
  // The category of the activity.
  ActivityCategory category = ActivityCategory::kMisc;
  // Key/value pairs that are attached to the activity.
  Attributes attributes;
  Activity() = default;
  Activity(tensorflow::string title, ActivityCategory category)
      : title(std::move(title)), category(category) {}
  Activity(tensorflow::string title, ActivityCategory category,
           Attributes attributes)
      : title(std::move(title)),
        category(category),
        attributes(std::move(attributes)) {}
};

// Enable activity wathcer to send own workers activities to coordination
// service and also fetch all workers' activities.
void MaybeEnableMultiWorkersWatching(CoordinationServiceAgent* agent);

namespace tfw_internal {

// Records an activity start without checking whether the watcher is enabled.
ActivityId RecordActivityStart(std::unique_ptr<Activity> activity);
// Records an activity end without checking whether the activity_id is valid.
void RecordActivityEnd(ActivityId activity_id);

TF_EXPORT extern std::atomic<int> g_watcher_level;

// Returns whether the activitity watcher is enabled.
inline bool WatcherEnabled(int level = 1) {
  return g_watcher_level.load(std::memory_order_acquire) >= level;
}

}  // namespace tfw_internal

template <typename F>
constexpr bool is_activity_generator =
    std::is_invocable_r<std::unique_ptr<Activity>, F>::value;

// Records an activity explicitly. Useful when the start and end of an activity
// happen in different threads. Generates the Activity only if activity
// watching is enabled, useful for avoiding expensive operations when activity
// watching is disabled.
// Example Usage:
//   auto aid = ActivityStart([&]() {
//     return std::make_unique<Activity>(
//         op_name, category,
//         Activity::Attributes{{"key1", value1}, {"key2", value2}});
//   }, /*level=*/2);
//   DoSomething();
//   ActivityEnd(aid);
template <
    typename ActivityGenerator,
    std::enable_if_t<is_activity_generator<ActivityGenerator>, bool> = true>
inline ActivityId ActivityStart(ActivityGenerator&& gen, int level = 1) {
  if (TF_PREDICT_FALSE(tfw_internal::WatcherEnabled(level))) {
    return tfw_internal::RecordActivityStart(
        std::forward<ActivityGenerator>(gen)());
  }
  return kActivityNotRecorded;
}

inline void ActivityEnd(ActivityId id) {
  if (TF_PREDICT_FALSE(id != kActivityNotRecorded)) {
    tfw_internal::RecordActivityEnd(id);
  }
}

// ActivityScope marks a scope as an activity and record it with a global
// ActivityRecorder.
// Example Usage:
//   {
//     ActivityScope activity_scope([&]() {
//       return std::make_unique<Activity>(
//           op_name, ActivityCategory::kMisc,
//           Activity::Attributes{{"key1", value1}, {"key2", value2}});
//     }, /*level=*/2);
//     DoSomething();
//   }
class ActivityScope {
 public:
  template <
      typename ActivityGenerator,
      std::enable_if_t<is_activity_generator<ActivityGenerator>, bool> = true>
  explicit ActivityScope(ActivityGenerator&& gen, int level = 1) {
    activity_id_ = ActivityStart(std::forward<ActivityGenerator>(gen), level);
  }
  ~ActivityScope() { ActivityEnd(activity_id_); }

 private:
  ActivityId activity_id_;
  TF_DISALLOW_COPY_AND_ASSIGN(ActivityScope);
};

}  // namespace activity_watcher
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_ACTIVITY_WATCHER_ACTIVITY_H_
