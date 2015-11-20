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

#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/random/exact_uniform_int.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace random {

uint32 SimplePhilox::Uniform(uint32 n) {
  return ExactUniformInt<uint32>(n, [this]() { return Rand32(); });
}

uint64 SimplePhilox::Uniform64(uint64 n) {
  return ExactUniformInt<uint64>(n, [this]() { return Rand64(); });
}

uint32 SimplePhilox::Skewed(int max_log) {
  CHECK(0 <= max_log && max_log <= 32);
  const int shift = Rand32() % (max_log + 1);
  const uint32 mask = shift == 32 ? ~static_cast<uint32>(0) : (1 << shift) - 1;
  return Rand32() & mask;
}

}  // namespace random
}  // namespace tensorflow
