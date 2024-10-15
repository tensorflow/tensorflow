/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_RANDOM_RANDOM_DISTRIBUTIONS_UTILS_H_
#define TENSORFLOW_CORE_LIB_RANDOM_RANDOM_DISTRIBUTIONS_UTILS_H_

#include <string.h>

#include <cstdint>

#include "xla/tsl/lib/random/random_distributions_utils.h"
#include "tensorflow/core/lib/random/philox_random.h"

namespace tensorflow {
namespace random {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::random::BoxMullerFloat;
using tsl::random::Uint32ToFloat;
using tsl::random::Uint64ToDouble;
// NOLINTEND(misc-unused-using-decls)
}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_RANDOM_RANDOM_DISTRIBUTIONS_UTILS_H_
