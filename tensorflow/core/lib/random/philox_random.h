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

// Implement the Philox algorithm to generate random numbers in parallel.
// Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
//   http://www.thesalmons.org/john/random123/papers/random123sc11.pdf

#ifndef TENSORFLOW_CORE_LIB_RANDOM_PHILOX_RANDOM_H_
#define TENSORFLOW_CORE_LIB_RANDOM_PHILOX_RANDOM_H_

#include "tsl/lib/random/philox_random.h"

namespace tensorflow {
namespace random {
// NOLINTBEGIN(misc-unused-using-decls)
using tsl::random::Array;
using tsl::random::PhiloxRandom;
// NOLINTEND(misc-unused-using-decls)

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_RANDOM_PHILOX_RANDOM_H_
