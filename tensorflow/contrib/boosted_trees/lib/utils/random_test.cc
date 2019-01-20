// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
#include "tensorflow/contrib/boosted_trees/lib/utils/random.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {
namespace {

TEST(RandomTest, Poisson) {
  random::PhiloxRandom philox(77L);
  random::SimplePhilox rng(&philox);
  for (int trial = 0; trial < 10; ++trial) {
    const int32 num_bootstrap = 10000;
    double sum = 0;
    double zeros = 0;
    double ones = 0;
    for (int i = 0; i < num_bootstrap; ++i) {
      auto n = PoissonBootstrap(&rng);
      sum += n;
      zeros += (n == 0) ? 1 : 0;
      ones += (n == 1) ? 1 : 0;
    }

    // Ensure mean is near expected value.
    const double expected_mean = 1.0;  // lambda
    const double mean_std_error = 1.0 / sqrt(num_bootstrap);
    double mean = sum / num_bootstrap;
    EXPECT_NEAR(mean, expected_mean, 3 * mean_std_error);

    // Ensure probability mass for values 0 and 1 are near expected value.
    const double expected_p = 0.368;
    const double proportion_std_error =
        sqrt(expected_p * (1 - expected_p) / num_bootstrap);
    EXPECT_NEAR(zeros / num_bootstrap, expected_p, 3 * proportion_std_error);
    EXPECT_NEAR(ones / num_bootstrap, expected_p, 3 * proportion_std_error);
  }
}

}  // namespace
}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow
