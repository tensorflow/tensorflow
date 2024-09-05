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

#ifndef TENSORFLOW_TSL_LIB_GTL_INLINED_VECTOR_H_
#define TENSORFLOW_TSL_LIB_GTL_INLINED_VECTOR_H_

#include <cstddef>

#include "absl/base/macros.h"
#include "absl/container/inlined_vector.h"  // IWYU pragma: export
// TODO(kramerb): This is kept only because lots of targets transitively depend
// on it. Remove all targets' dependencies.
#include "tsl/platform/macros.h"
#include "tsl/platform/types.h"

// TODO: b/323943471 - This macro should eventually be provided by Abseil.
#ifndef ABSL_DEPRECATE_AND_INLINE
#define ABSL_DEPRECATE_AND_INLINE()
#endif

namespace tsl {
namespace gtl {

template <typename T, size_t N>
using InlinedVector ABSL_DEPRECATE_AND_INLINE() = absl::InlinedVector<T, N>;

}  // namespace gtl
}  // namespace tsl

#endif  // TENSORFLOW_TSL_LIB_GTL_INLINED_VECTOR_H_
