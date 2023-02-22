/* Copyright 2022 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/activity_watcher/activity_utils.h"

#include <memory>
#include <utility>

namespace tensorflow {
namespace activity_watcher {

std::unique_ptr<Activity> ActivityFromContext(
    OpKernelContext* context, tensorflow::string name,
    ActivityCategory category, Activity::Attributes additional_attributes) {
  Activity::Attributes attributes(std::move(additional_attributes));
  if (context) {
    attributes.merge(Activity::Attributes({
        {"node_name", context->op_kernel().def().name()},
        {"step_id", absl::StrCat(context->step_id())},
        {"device", context->device()->name()},
        {"op", context->op_kernel().def().op()},
        {"iter_num", absl::StrCat(context->frame_iter().iter_id)},
    }));
  }

  return std::make_unique<Activity>(name, category, std::move(attributes));
}

}  // namespace activity_watcher
}  // namespace tensorflow
