/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BOUNDS_CHECK_H_
#define TENSORFLOW_CORE_KERNELS_BOUNDS_CHECK_H_

#include <type_traits>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// Check that 0 <= index < limit using a single comparison, assuming
// that 0 <= limit if Index is signed.  Intended for use in performance
// critical contexts where 0 <= index < limit is almost always true.
template <typename Ta, typename Tb>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC bool FastBoundsCheck(const Ta index,
                                                           const Tb limit) {
  static_assert(std::is_integral<Ta>::value && std::is_integral<Tb>::value,
                "FastBoundsCheck can only be used on integer types.");
  typedef typename std::make_unsigned<decltype(index + limit)>::type UIndex;
  return TF_PREDICT_TRUE(static_cast<UIndex>(index) <
                         static_cast<UIndex>(limit));
}

namespace internal {
// Ensure that the compiler cannot elide a copy into a local, for
// bounds checking on source tensors that might be updated asynchronously.
// This function may only be used on primitive integral types (int32, int64,
// etc).  It does not guarantee any atomicity or barriers.
template <typename T>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC const T SubtleMustCopy(const T &x) {
  static_assert(std::is_integral<T>::value,
                "SubtleMustCopy can only be used on integer types.");
  auto *to_x = reinterpret_cast<const volatile T *>(&x);
  return *to_x;
}
}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BOUNDS_CHECK_H_
