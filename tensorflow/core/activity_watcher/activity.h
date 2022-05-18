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

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace activity_watcher {

using ActivityId = uint64;

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

struct Activity {
  tensorflow::string title;
  ActivityCategory category;
  absl::flat_hash_map<tensorflow::string, tensorflow::string> attributes;
  Activity() = default;
  Activity(tensorflow::string title, ActivityCategory category)
      : title(std::move(title)), category(category) {}
};

// ActivityScope marks a scope as an activity and record it with a global
// ActivityRecorder.
class ActivityScope {
 public:
  explicit ActivityScope(std::unique_ptr<Activity> activity);
  ~ActivityScope();

 private:
  ActivityId ABSL_ATTRIBUTE_UNUSED activity_id_;
  TF_DISALLOW_COPY_AND_ASSIGN(ActivityScope);
};

// Records an activity explicitly. Useful when the start and end of an activity
// happen in different threads.
ActivityId ActivityStart(std::unique_ptr<Activity> activity);
void ActivityEnd(ActivityId id);

}  // namespace activity_watcher
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_ACTIVITY_WATCHER_ACTIVITY_H_
