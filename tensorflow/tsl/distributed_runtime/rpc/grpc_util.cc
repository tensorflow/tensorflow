/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/distributed_runtime/rpc/grpc_util.h"

#include "tensorflow/tsl/platform/random.h"

namespace tsl {

namespace {

double GenerateUniformRandomNumber() {
  return random::New64() * (1.0 / std::numeric_limits<uint64>::max());
}

double GenerateUniformRandomNumberBetween(double a, double b) {
  if (a == b) return a;
  DCHECK_LT(a, b);
  return a + GenerateUniformRandomNumber() * (b - a);
}

}  // namespace

int64_t ComputeBackoffMicroseconds(int current_retry_attempt, int64_t min_delay,
                                   int64_t max_delay) {
  DCHECK_GE(current_retry_attempt, 0);

  // This function with the constants below is calculating:
  //
  // (0.4 * min_delay) + (random[0.6,1.0] * min_delay * 1.3^retries)
  //
  // Note that there is an extra truncation that occurs and is documented in
  // comments below.
  constexpr double kBackoffBase = 1.3;
  constexpr double kBackoffRandMult = 0.4;

  // This first term does not vary with current_retry_attempt or a random
  // number. It exists to ensure the final term is >= min_delay
  const double first_term = kBackoffRandMult * min_delay;

  // This is calculating min_delay * 1.3^retries
  double uncapped_second_term = min_delay;
  while (current_retry_attempt > 0 &&
         uncapped_second_term < max_delay - first_term) {
    current_retry_attempt--;
    uncapped_second_term *= kBackoffBase;
  }
  // Note that first_term + uncapped_second_term can exceed max_delay here
  // because of the final multiply by kBackoffBase.  We fix that problem with
  // the min() below.
  double second_term = std::min(uncapped_second_term, max_delay - first_term);

  // This supplies the random jitter to ensure that retried don't cause a
  // thundering herd problem.
  second_term *=
      GenerateUniformRandomNumberBetween(1.0 - kBackoffRandMult, 1.0);

  return std::max(static_cast<int64_t>(first_term + second_term), min_delay);
}
}  // namespace tsl
