/* Copyright 2019 Google LLC. All Rights Reserved.

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

// Miscellaneous helpers internal library.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_COMMON_H_

#include <atomic>
#include <limits>
#include <type_traits>

#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/path.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#if RUY_OPT_SET & RUY_OPT_PREFETCH
#define RUY_PREFETCH(X) X
#else
#define RUY_PREFETCH(X)
#endif

#define RUY_STR(s) RUY_STR_UNEXPANDED(s)
#define RUY_STR_UNEXPANDED(s) #s

namespace ruy {

// Helper for type-erasing a pointer.
//
// Often inside Ruy, a template parameter holds type information statically, but
// we would like to have a function signature that doesn't depend on the
// template parameters, so that we can dispatch indirectly across multiple
// implementations. This helper is at the core of such type-erasure.
//
// The opposite of this operation is just `static_cast<T*>(void_ptr)`.
template <typename T>
void* ToVoidPtr(T* p) {
  return const_cast<void*>(static_cast<const void*>(p));
}

// We need this where we have multiple threads potentially writing concurrently
// to the same memory location. That is currently the case for Pack (see
// the comment in TrMulTask where Pack is called) and in tracing.
//
// This is a strict-aliasing violation. For nicer things, see C++20 atomic_ref
// and the defunct N4013. (Thanks to hboehm@).
template <typename T>
void relaxed_atomic_store(T* ptr, T value) {
  static_assert(sizeof(std::atomic<T>) == sizeof(T), "");
  std::atomic<T>* atomic = reinterpret_cast<std::atomic<T>*>(ptr);
  RUY_DCHECK(atomic->is_lock_free());
  atomic->store(value, std::memory_order_relaxed);
}

template <typename Scalar>
Scalar SymmetricZeroPoint() {
  if (std::is_floating_point<Scalar>::value) {
    return 0;
  }
  if (std::is_signed<Scalar>::value) {
    return 0;
  }
  return std::numeric_limits<Scalar>::max() / 2 + 1;
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_COMMON_H_
