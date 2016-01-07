/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_PLATFORM_DEFAULT_INTEGRAL_TYPES_H_
#define TENSORFLOW_PLATFORM_DEFAULT_INTEGRAL_TYPES_H_

// IWYU pragma: private, include "third_party/tensorflow/core/platform/types.h"
// IWYU pragma: friend third_party/tensorflow/core/platform/types.h

namespace tensorflow {

typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_DEFAULT_INTEGRAL_TYPES_H_
