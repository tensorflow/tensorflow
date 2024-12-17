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
#include <optional>
#include <utility>

#include "absl/log/absl_check.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"

namespace litert {

// See "std::construct_at" from C++20.
template <class T, class... Args>
T* ConstructAt(T* p, Args&&... args) {
  return ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
}

// Reduce all over zipped iters of same size.
template <typename LeftVals, typename RightVals = LeftVals>
bool AllZip(const LeftVals& lhs, const RightVals& rhs,
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
bool AnyZip(const LeftVals& lhs, const RightVals& rhs,
            std::function<bool(const typename LeftVals::value_type&,
                               const typename RightVals::value_type&)>
                bin_pred) {
  auto neg = [&](const auto& l, const auto& r) { return !bin_pred(l, r); };
  return !(AllZip(lhs, rhs, neg));
}

// Does element exist in range.
template <class It, class T>
bool Contains(It begin, It end, const T& val) {
  return std::find(begin, end, val) != end;
}

// Does element exist in range satisfying pred.
template <class It, class UPred>
bool ContainsIf(It begin, It end, UPred u_pred) {
  return std::find_if(begin, end, u_pred) != end;
}

// Get the ind of the given element if it is present.
template <class T, class It>
std::optional<size_t> FindInd(It begin, It end, T val) {
  auto it = std::find(begin, end, val);
  return (it == end) ? std::nullopt : std::make_optional(it - begin);
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
