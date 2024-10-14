// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_UTILS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_UTILS_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"

namespace lrt {
namespace internal {

struct Ratio {
  using Type = int;
  Type num;
  Type denom;
  std::string ToString() const { return absl::StrCat(num, "/", denom); }
};

absl::StatusOr<Ratio> GetElementSize(LrtElementType element_type);

absl::StatusOr<size_t> GetNumElements(const LrtRankedTensorType& tensor_type);

// Get the number of bytes necessary to represent a tensor type, ignoring any
// stride information.
absl::StatusOr<size_t> GetNumPackedBytes(
    const LrtRankedTensorType& tensor_type);

}  // namespace internal
}  // namespace lrt

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_UTILS_H_
