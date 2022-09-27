/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_UTIL_PTR_UTIL_H_
#define TENSORFLOW_TSL_UTIL_PTR_UTIL_H_

// Utility functions for pointers.

#include <stddef.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace tsl {

namespace helper {

// Trait to select overloads and return types for MakeUnique.
template <typename T>
struct MakeUniqueResult {
  using scalar = std::unique_ptr<T>;
};
template <typename T>
struct MakeUniqueResult<T[]> {
  using array = std::unique_ptr<T[]>;
};
template <typename T, size_t N>
struct MakeUniqueResult<T[N]> {
  using invalid = void;
};

}  // namespace helper

// Transfers ownership of a raw pointer to a std::unique_ptr of deduced type.
// Example:
//   X* NewX(int, int);
//   auto x = WrapUnique(NewX(1, 2));  // 'x' is std::unique_ptr<X>.
//
// WrapUnique is useful for capturing the output of a raw pointer factory.
// However, prefer 'MakeUnique<T>(args...) over 'WrapUnique(new T(args...))'.
//   auto x = WrapUnique(new X(1, 2));  // works, but nonideal.
//   auto x = MakeUnique<X>(1, 2);  // safer, standard, avoids raw 'new'.
//
// Note: Cannot wrap pointers to array of unknown bound (i.e. U(*)[]).
template <typename T>
std::unique_ptr<T> WrapUnique(T* ptr) {
  static_assert(!std::is_array<T>::value || std::extent<T>::value != 0,
                "types T[0] or T[] are unsupported");
  return std::unique_ptr<T>(ptr);
}

template <typename T, typename... Args>
typename helper::MakeUniqueResult<T>::scalar MakeUnique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Overload for array of unknown bound.
// The allocation of arrays needs to use the array form of new,
// and cannot take element constructor arguments.
template <typename T>
typename helper::MakeUniqueResult<T>::array MakeUnique(size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

}  // namespace tsl

#endif  // TENSORFLOW_TSL_UTIL_PTR_UTIL_H_
