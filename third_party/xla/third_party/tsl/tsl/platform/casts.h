/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_CASTS_H_
#define TENSORFLOW_TSL_PLATFORM_CASTS_H_

#include "absl/base/casts.h"
#include "absl/base/macros.h"

namespace tensorflow {

template <typename To, typename From>
ABSL_DEPRECATE_AND_INLINE()
inline To down_cast(From* f) {
  return absl::down_cast<To>(f);
}

template <typename To, typename From>
ABSL_DEPRECATE_AND_INLINE()
inline To down_cast(From& f) {
  return absl::down_cast<To>(f);
}

}  // namespace tensorflow

namespace tsl {
using ::tensorflow::down_cast;
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_CASTS_H_
