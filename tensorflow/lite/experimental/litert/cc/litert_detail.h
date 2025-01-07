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
#include <functional>

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

// Reduce all over zipped iters of same size.
template <typename LeftVals, typename RightVals = LeftVals>
inline bool AllZip(const LeftVals& lhs, const RightVals& rhs,
                   std::function<bool(const typename LeftVals::value_type&,
                                      const typename RightVals::value_type&)>
                       bin_pred) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto i = 0; i < lhs.size(); ++i) {
    if (!bin_pred(lhs.at(i), rhs.at(i))) {
      return false;
    }
  }
  return true;
}

// Reduce any over zipped iters of same size.
template <typename LeftVals, typename RightVals = LeftVals>
inline bool AnyZip(const LeftVals& lhs, const RightVals& rhs,
                   std::function<bool(const typename LeftVals::value_type&,
                                      const typename RightVals::value_type&)>
                       bin_pred) {
  auto neg = [&](const auto& l, const auto& r) { return !bin_pred(l, r); };
  return !(AllZip(lhs, rhs, neg));
}

namespace internal {

// Call function "get" and assert it returns value equal to given expected
// value.
template <class F, class Expected, typename... Args>
inline void AssertEq(F get, Expected expected, Args&&... args) {
  auto status = get(std::forward<Args>(args)...);
  ABSL_CHECK_EQ(status, expected);
}

// Call function "get" and assert it returns true.
template <class F, typename... Args>
inline void AssertTrue(F get, Args&&... args) {
  AssertEq(get, true, std::forward<Args>(args)...);
}

// Call function "get" and assert it returns an OK LiteRtStatus.
template <class F, typename... Args>
inline void AssertOk(F get, Args&&... args) {
  AssertEq(get, kLiteRtStatusOk, std::forward<Args>(args)...);
}

}  // namespace internal
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_DETAIL_H_
