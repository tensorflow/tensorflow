/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace serving {

absl::StatusOr<MixedPriorityBatchingPolicy> GetMixedPriorityBatchingPolicy(
    absl::string_view attr_value) {
  if (attr_value == kLowPriorityPaddingWithMaxBatchSizeAttrValue) {
    return MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize;
  } else if (attr_value ==
             kLowPriorityPaddingWithNextAllowedBatchSizeAttrValue) {
    return MixedPriorityBatchingPolicy::
        kLowPriorityPaddingWithNextAllowedBatchSize;
  } else if (attr_value == kPriorityIsolationAttrValue) {
    return MixedPriorityBatchingPolicy::kPriorityIsolation;
  } else if (attr_value == kPriorityMergeAttrValue) {
    return MixedPriorityBatchingPolicy::kPriorityMerge;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Unknown mixed priority batching policy: %s", attr_value));
}

}  // namespace serving
}  // namespace tensorflow
