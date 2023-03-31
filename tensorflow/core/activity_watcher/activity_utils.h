/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_ACTIVITY_WATCHER_ACTIVITY_UTILS_H_
#define TENSORFLOW_CORE_ACTIVITY_WATCHER_ACTIVITY_UTILS_H_

#include <memory>

#include "tensorflow/core/activity_watcher/activity.h"

namespace tensorflow {

class OpKernelContext;

namespace activity_watcher {

// A convenient way to create an activity. Writes OpKernelContext information
// and given attributes to a new activity and returns.
std::unique_ptr<Activity> ActivityFromContext(
    OpKernelContext* context, tsl::string name, ActivityCategory category,
    Activity::Attributes additional_attributes = Activity::Attributes());

}  // namespace activity_watcher
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_ACTIVITY_WATCHER_ACTIVITY_UTILS_H_
