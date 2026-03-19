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

#include "tensorflow/core/util/bad_indices_policy.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace tensorflow {

constexpr char kDefault[] = "DEFAULT";
constexpr char kErrorStr[] = "ERROR";
constexpr char kIgnoreStr[] = "IGNORE";

absl::StatusOr<BadIndicesPolicy> BadIndicesPolicyFromString(
    absl::string_view str) {
  if (str.empty()) return BadIndicesPolicy::kDefault;
  if (str == kDefault) return BadIndicesPolicy::kDefault;
  if (str == kErrorStr) return BadIndicesPolicy::kError;
  if (str == kIgnoreStr) return BadIndicesPolicy::kIgnore;
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown bad indices handling attribute: ", str));
}

}  // namespace tensorflow
