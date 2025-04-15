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

#ifndef TENSORFLOW_CORE_PLATFORM_RANDOM_H_
#define TENSORFLOW_CORE_PLATFORM_RANDOM_H_

#include "tensorflow/core/platform/types.h"
#include "tsl/platform/random.h"

namespace tensorflow {
namespace random {
using tsl::random::New64;             // NOLINT
using tsl::random::New64DefaultSeed;  // NOLINT
}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_RANDOM_H_
