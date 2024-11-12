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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_DETAIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_DETAIL_H_

#include <cstddef>

#include "absl/container/inlined_vector.h"
#include "absl/log/absl_check.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"

namespace litert {

// Expected size for inlined vectors for things like the input/outputs of ops or
// subgraphs.
static constexpr size_t kTensorVecSize = 8;
template <typename T>
using SmallVec = absl::InlinedVector<T, kTensorVecSize>;

// See "std::construct_at" from C++20.
template <class T, class... Args>
inline T* ConstructAt(T* p, Args&&... args) {
  return ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
}

namespace internal {

// Call function "get" and assert it returns an OK LiteRtStatus.
template <typename... Args>
void AssertOk(LiteRtStatus (*get)(Args...), Args... args) {
  auto status = get(args...);
  ABSL_CHECK_EQ(status, kLiteRtStatusOk);
}

}  // namespace internal

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_DETAIL_H_
