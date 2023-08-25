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

// An abstraction to pick from one of N elements with a specified
// weight per element.
//
// The weight for a given element can be changed in O(lg N) time
// An element can be picked in O(lg N) time.
//
// Uses O(N) bytes of memory.
//
// Alternative: distribution-sampler.h allows O(1) time picking, but no weight
// adjustment after construction.

#ifndef TENSORFLOW_CORE_LIB_RANDOM_WEIGHTED_PICKER_H_
#define TENSORFLOW_CORE_LIB_RANDOM_WEIGHTED_PICKER_H_

#include <assert.h>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/tsl/lib/random/weighted_picker.h"

namespace tensorflow {
namespace random {
using tsl::random::WeightedPicker;  // NOLINT(misc-unused-using-decls)
}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_RANDOM_WEIGHTED_PICKER_H_
