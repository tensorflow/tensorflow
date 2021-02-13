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

#include "tensorflow/core/lib/random/distribution_sampler.h"

#include <string.h>
#include <memory>
#include <vector>

#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace random {

class DistributionSamplerTest : public ::testing::Test {
 protected:
  // Returns the Chi-Squared statistic for the two distributions.
  float TestWeights(const std::vector<float>& weights, int trials_per_bin) {
    int iters = weights.size() * trials_per_bin;
    std::unique_ptr<float[]> counts(new float[weights.size()]);
    memset(counts.get(), 0, sizeof(float) * weights.size());
    DistributionSampler sampler(weights);
    PhiloxRandom philox(testing::RandomSeed(), 17);
    SimplePhilox random(&philox);
    for (int i = 0; i < iters; i++) {
      int r = sampler.Sample(&random);
      EXPECT_LT(r, weights.size());
      EXPECT_GE(r, 0);
      counts[r] += 1.0;
    }
    float chi2 = 0.0;
    for (size_t i = 0; i < weights.size(); i++) {
      counts[i] /= iters;
      float err = (counts[i] - weights[i]);
      chi2 += (err * err) / weights[i];
    }
    return chi2;
  }

  void TestDistribution(float* arr, int n) {
    std::vector<float> w;
    w.reserve(n);
    for (int i = 0; i < n; i++) {
      w.push_back(arr[i]);
    }
    float var = TestWeights(w, 1000);
    if (var < 0.001) return;
    // Maybe a statistical skew. Let's try more iterations.
    var = TestWeights(w, 100000);
    if (var < 0.001) return;
    EXPECT_TRUE(false) << "Chi2 is " << var << " in " << n * 100000
                       << "iterations";
  }
};

TEST_F(DistributionSamplerTest, KnownDistribution) {
  float kEven2[] = {0.5, 0.5};
  float kEven3[] = {0.33333333, 0.33333333, 0.33333333};
  float kEven4[] = {0.25, 0.25, 0.25, 0.25};

  float kDist1[] = {0.8, 0.15, 0.05};

  TestDistribution(kEven2, TF_ARRAYSIZE(kEven2));
  TestDistribution(kEven3, TF_ARRAYSIZE(kEven3));
  TestDistribution(kEven4, TF_ARRAYSIZE(kEven4));
  TestDistribution(kDist1, TF_ARRAYSIZE(kDist1));
}

static void BM_DistributionSampler(::testing::benchmark::State& state) {
  const int n = state.range(0);
  PhiloxRandom philox(173, 371);
  SimplePhilox rand(&philox);
  std::vector<float> weights(n, 0);
  for (int i = 0; i < n; i++) {
    weights[i] = rand.Uniform(100);
  }
  DistributionSampler picker(weights);
  int r = 0;
  for (auto s : state) {
    r |= picker.Sample(&rand);
  }
  CHECK_NE(r, kint32max);
}

BENCHMARK(BM_DistributionSampler)->Arg(10)->Arg(100)->Arg(1000);

}  // namespace random
}  // namespace tensorflow
