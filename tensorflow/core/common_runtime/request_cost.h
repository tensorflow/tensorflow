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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_REQUEST_COST_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_REQUEST_COST_H_

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace tensorflow {

// RequestCost collects the costs for processing an rpc request.
// It's thread-safe. RecordCost can be called from different threads.
class RequestCost {
 public:
  // Records costs. The inputs should be pairs of cost type and cost.
  void RecordCost(
      const std::vector<std::pair<absl::string_view, absl::Duration>>& costs);

 private:
  absl::Mutex mutex_;
  // Map from cost type to cost.
  absl::flat_hash_map<std::string, absl::Duration> cost_map_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_REQUEST_COST_H_
