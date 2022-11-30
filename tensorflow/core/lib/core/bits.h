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

#ifndef TENSORFLOW_CORE_LIB_CORE_BITS_H_
#define TENSORFLOW_CORE_LIB_CORE_BITS_H_

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tsl/lib/core/bits.h"

namespace tensorflow {

// NOLINTBEGIN(misc-unused-using-decls)

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
using ::tsl::Log2Floor;
using ::tsl::Log2Floor64;

// Return ceiling(log2(n)) for positive integer n.  Returns -1 iff n == 0.
using ::tsl::Log2Ceiling;
using ::tsl::Log2Ceiling64;

using ::tsl::NextPowerOfTwo;
using ::tsl::NextPowerOfTwo64;

// NOLINTEND(misc-unused-using-decls)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_CORE_BITS_H_
