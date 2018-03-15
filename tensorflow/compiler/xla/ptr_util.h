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

#ifndef TENSORFLOW_COMPILER_XLA_PTR_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_PTR_UTIL_H_

// As this was moved to tensorflow/core/util, provide indirections here to
// maintain current functionality of the library.

#include <stddef.h>

#include <memory>
#include <type_traits>
#include <utility>

#include "tensorflow/core/util/ptr_util.h"

namespace xla {

template <typename T>
std::unique_ptr<T> WrapUnique(T* ptr) {
  return tensorflow::WrapUnique<T>(ptr);
}

template <typename T, typename... Args>
typename tensorflow::helper::MakeUniqueResult<T>::scalar MakeUnique(
    Args&&... args) {
  return tensorflow::MakeUnique<T, Args...>(std::forward<Args>(args)...);
}

// Overload for array of unknown bound.
// The allocation of arrays needs to use the array form of new,
// and cannot take element constructor arguments.
template <typename T>
typename tensorflow::helper::MakeUniqueResult<T>::array MakeUnique(size_t n) {
  return tensorflow::MakeUnique<T>(n);
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PTR_UTIL_H_
