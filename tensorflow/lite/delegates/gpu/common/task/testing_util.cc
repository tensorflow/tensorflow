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

#include "tensorflow/lite/delegates/gpu/common/task/testing_util.h"

#include "absl/strings/str_cat.h"

namespace tflite {
namespace gpu {

absl::Status PointWiseNear(const std::vector<float>& ref,
                           const std::vector<float>& to_compare, float eps) {
  if (ref.size() != to_compare.size()) {
    return absl::InternalError(absl::StrCat("ref size(", ref.size(),
                                            ") != to_compare size(",
                                            to_compare.size(), ")"));
  }
  for (int i = 0; i < ref.size(); ++i) {
    const float abs_diff = fabs(ref[i] - to_compare[i]);
    if (abs_diff > eps) {
      return absl::InternalError(absl::StrCat(
          "ref[", i, "] = ", ref[i], ", to_compare[", i, "] = ", to_compare[i],
          ", abs diff = ", abs_diff, " > ", eps, " (eps)"));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace tflite
