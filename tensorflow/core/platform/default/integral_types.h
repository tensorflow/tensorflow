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

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_INTEGRAL_TYPES_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_INTEGRAL_TYPES_H_

#include <cstdint>

// IWYU pragma: private, include "third_party/tensorflow/core/platform/types.h"
// IWYU pragma: friend third_party/tensorflow/core/platform/types.h

namespace tensorflow {

typedef signed char int8;
typedef short int16;
typedef int int32;
[[deprecated("Use int64_t instead.")]] typedef ::std::int64_t int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef std::uint64_t uint64;

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_INTEGRAL_TYPES_H_
