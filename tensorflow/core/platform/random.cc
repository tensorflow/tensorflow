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

#include "tensorflow/core/platform/random.h"

#include <random>
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace random {

uint64 New64() {
  static RandomGenerator* g = new RandomGenerator(RandomGenerator::kUrandom);
  return g->New64();
}

uint64 New64DefaultSeed() {
  static RandomGenerator* g = new RandomGenerator(RandomGenerator::kDefault);
  return g->New64();
}

RandomGenerator::RandomGenerator(SeedType seed_type) {
  switch (seed_type) {
    case kDefault:
      rng_ = std::mt19937_64();
      return;
    case kUrandom:
      std::random_device device("/dev/urandom");
      rng_ = std::mt19937_64(device());
      return;
  }
}

uint64 RandomGenerator::New64() {
  mutex_lock l(mu_);
  return rng_();
}

}  // namespace random
}  // namespace tensorflow
