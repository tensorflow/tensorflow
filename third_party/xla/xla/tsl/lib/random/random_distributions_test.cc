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

#include "xla/tsl/lib/random/random_distributions.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "xla/tsl/lib/math/math_util.h"
#include "xla/tsl/lib/random/philox_random.h"
#include "xla/tsl/lib/random/philox_random_test_utils.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/random.h"

namespace tsl {
namespace random {
namespace {

// The largest z-value we want to tolerate. Since the z-test approximates a
// unit normal distribution, it should almost definitely never exceed 6.
static constexpr float kZLimit = 6.0;

// As bfloat16 has much less precision, the largest z-value will should be
// larger than float32.
static constexpr float kZLimitBfloat16 = 20.0;

// A utility function to fill the given array with samples from the given
// distribution, using the single adapter of the underlying generator
template <class Distribution>
void FillRandomsWithSingles(PhiloxRandom gen,
                            typename Distribution::ResultElementType* p,
                            int64_t size) {
  int granularity = Distribution::kResultElementCount;

  CHECK(size % granularity == 0)
      << " size: " << size << " granularity: " << granularity;

  SingleSampleAdapter<PhiloxRandom> single_samples(&gen);

  Distribution dist;
  for (int i = 0; i < size; i += granularity) {
    auto sample = dist(&single_samples);
    std::copy(&sample[0], &sample[0] + granularity, &p[i]);
  }
}

// Check the given array of samples matches the given theoretical moment
// function at different orders. The test is considered passing if the z-tests
// of all statistical moments are all below z_limit.
// typename T in the template argument could be either float or double.
// Arguments:
//   samples: an array of samples to be tested for their statistical properties;
//   theoretical_moments: a functor that can calculate arbitrary order of
//       of the given distribution;
//   max_moments: the largest moments of the uniform distribution to be tested;
//   stride: the distance between samples to check for statistical properties
//       0 means the n-th moment of each sample
//       any other strides tests for spatial correlation between samples;
//   z_limit: the maximum z-test we would consider the test to pass;
template <typename T>
bool CheckSamplesMoments(const std::vector<T>& samples,
                         const std::function<double(int)>& theoretical_moments,
                         int max_moments, int stride, T z_limit) {
  const T* const samples_data = &samples[0];
  const int samples_size = samples.size();
  std::vector<double> moments(max_moments + 1);
  double* const moments_data = &moments[0];
  std::vector<int> moments_sample_count(max_moments + 1);
  int* const moments_sample_count_data = &moments_sample_count[0];

  for (int k = 0; k < samples_size; ++k) {
    double moment = 1.;
    for (int i = 0; i <= max_moments; ++i) {
      int index = k + i * stride;
      if (index >= samples_size) {
        break;
      }
      // moments[i] store the i-th order measured moments.
      // bypass std::vector::operator[] because they are too slow in the debug
      // mode, given the large number of samples.
      moments_data[i] += moment;
      ++moments_sample_count_data[i];
      moment *= static_cast<double>(samples_data[index]);
    }
  }

  // normalize the moments
  for (int i = 0; i <= max_moments; ++i) {
    moments[i] /= moments_sample_count[i];
  }

  bool status = true;

  for (int i = 1; i <= max_moments; ++i) {
    // Calculate the theoretical mean and variance
    const double moments_i_mean =
        (stride == 0) ? theoretical_moments(i)
                      : MathUtil::IPow(theoretical_moments(1), i);
    const double moments_i_squared =
        (stride == 0) ? theoretical_moments(2 * i)
                      : MathUtil::IPow(theoretical_moments(2), i);
    const double moments_i_var =
        moments_i_squared - moments_i_mean * moments_i_mean;

    // assume every operation has a small numerical error.
    static const double kNumericalError = 1e-6;
    // it takes i multiplications to calculate one i-th moment.
    const double error_per_moment = i * kNumericalError;
    const double total_variance =
        moments_i_var / moments_sample_count[i] + error_per_moment;
    // z_test is approximately a unit normal distribution.
    const double z_test =
        fabs((moments[i] - moments_i_mean) / sqrt(total_variance));

    if (z_test > static_cast<double>(z_limit)) {
      LOG(ERROR) << "failing z_test:"
                 << " moment: " << i << " stride: " << stride
                 << " z_test: " << z_test << " z_limit: " << z_limit
                 << " measured moments: " << moments[i]
                 << " theoretical mean of the moments: " << moments_i_mean
                 << " theoretical var of the moments: " << moments_i_var
                 << " sample count: " << moments_sample_count[i];
      status = false;
    }
  }

  return status;
}

// This tests checks that the generated samples match the theoretical moments
// of the uniform distribution.
template <typename T>
void UniformMomentsTest(int count, int max_moments,
                        const std::vector<int>& strides, T z_limit) {
  auto uniform_moments = [](int n) -> double { return 1. / (n + 1); };

  std::vector<T> v1(count);
  uint64 seed = GetTestSeed();
  PhiloxRandom gen(seed);
  FillRandoms<UniformDistribution<PhiloxRandom, T> >(gen, &v1[0], v1.size());
  for (int stride : strides) {
    bool status =
        CheckSamplesMoments(v1, uniform_moments, max_moments, stride, z_limit);
    ASSERT_TRUE(status) << " UniformMomentsTest failing. seed: " << seed;
  }
}

// This test checks that the generated samples match the theoretical moments
// of the unit normal distribution.
template <typename T>
void NormalMomentsTest(int count, int max_moments,
                       const std::vector<int>& strides, T z_limit) {
  auto normal_moments = [](int n) -> double {
    if (n % 2 == 1) {
      // For an odd order, the moment of a unit normal distribution is zero.
      return 0.;
    } else {
      // For an even order, the moment of a unit normal distribution is.
      // (n-1)!!
      double v = 1.;
      for (int i = n - 1; i >= 1; i -= 2) {
        v *= i;
      }
      return v;
    }
  };

  std::vector<T> v1(count);
  uint64 seed = GetTestSeed();
  PhiloxRandom gen(seed);
  FillRandoms<NormalDistribution<PhiloxRandom, T> >(gen, &v1[0], v1.size());

  for (int stride : strides) {
    bool status =
        CheckSamplesMoments(v1, normal_moments, max_moments, stride, z_limit);
    ASSERT_TRUE(status) << " NormalMomentsTest failing. seed: " << seed;
  }
}

// A functor to calculate the moments for the truncated normal distribution.
// For any odd order, the moment is zero. But for any other n, it can be proven
// that the following recursive relationship for the moments of the truncated
// standard normal:
//   m(n) = (n - 1) * m(n - 2) - 2 * v ^ (n - 1) * f(v) / (2 * Phi(v) - 1)
//   where v is the cut-off value, f(v) is the p.d.f of the standard
//     normal, and Phi(v) is the c.d.f of the standard normal.
class TruncatedNormalMoments {
 public:
  double operator()(int n) {
    if (n == 0) {
      return 1;
    }
    if (n % 2 == 1) {
      // For an odd order, the moment is always zero
      return 0.;
    }

    // Memoization and check the cached results.
    auto iter = cached_results_.find(n);
    if (iter != cached_results_.end()) {
      return iter->second;
    }

    // The real computation of the moment.
    double bias = 2.0 * MathUtil::IPow(kV, n - 1) * kFV / (2.0 * kPhiV - 1.0);
    double moment_n_minus_2 = (*this)(n - 2);
    double moment_n = (n - 1) * moment_n_minus_2 - bias;

    cached_results_[n] = moment_n;
    return moment_n;
  }

 private:
  const double kV = 2.0;
  // f(v), where f is the p.d.f of the normal distribution and v=2.
  const double kFV = 1.0 / sqrt(2.0 * M_PI) * exp(-kV * kV / 2.0);
  // The numerical evaluation of Phi(v), where v is the truncate value.
  // v = 2 in the current implementation.
  const double kPhiV = 0.977249868051821;
  std::unordered_map<int, double> cached_results_;
};

// This test checks that the generated samples matche the theoretical moments
// of the truncated normal distribution.
template <typename T>
void RandomParametersMomentsTest(int count, int max_moments,
                                 const std::vector<int>& strides, T z_limit) {
  std::vector<T> v1(count);
  uint64 seed = GetTestSeed();
  PhiloxRandom gen(seed);
  FillRandomsWithSingles<
      TruncatedNormalDistribution<SingleSampleAdapter<PhiloxRandom>, T> >(
      gen, &v1[0], v1.size());

  for (int stride : strides) {
    bool status = CheckSamplesMoments(v1, TruncatedNormalMoments(), max_moments,
                                      stride, z_limit);
    ASSERT_TRUE(status) << " NormalMomentsTest failing. seed: " << seed;
  }
}

TEST(PhiloxRandomTest, UniformBfloat16MomentsTest) {
  const std::vector<int> strides = {0, 1, 4, 17};
  UniformMomentsTest<bfloat16>(1 << 20, 40, strides, bfloat16(kZLimitBfloat16));
}

TEST(PhiloxRandomTest, NormalBfloat16MomentsTest) {
  const std::vector<int> strides = {0, 1, 4, 17};
  NormalMomentsTest<bfloat16>(8 << 20, 25, strides, bfloat16(kZLimitBfloat16));
}

TEST(PhiloxRandomTest, RandomParametersBfloat16MomentsTest) {
  const std::vector<int> strides = {0, 1, 4, 17};
  RandomParametersMomentsTest<bfloat16>(1 << 20, 40, strides,
                                        bfloat16(kZLimitBfloat16));
}

TEST(PhiloxRandomTest, UniformFloatMomentsTest) {
  const std::vector<int> strides = {0, 1, 4, 17};
  UniformMomentsTest<float>(1 << 20, 40, strides, kZLimit);
}

TEST(PhiloxRandomTest, NormalFloatMomentsTest) {
  const std::vector<int> strides = {0, 1, 4, 17};
  NormalMomentsTest<float>(8 << 20, 25, strides, kZLimit);
}

TEST(PhiloxRandomTest, RandomParametersFloatMomentsTest) {
  const std::vector<int> strides = {0, 1, 4, 17};
  RandomParametersMomentsTest<float>(1 << 20, 40, strides, kZLimit);
}

TEST(PhiloxRandomTest, UniformDoubleMomentsTest) {
  const std::vector<int> strides = {0, 1, 4, 17};
  UniformMomentsTest<double>(1 << 20, 40, strides, kZLimit);
}

TEST(PhiloxRandomTest, NormalDoubleMomentsTest) {
  const std::vector<int> strides = {0, 1, 4, 17};
  NormalMomentsTest<double>(8 << 20, 25, strides, kZLimit);
}

TEST(PhiloxRandomTest, RandomParametersDoubleMomentsTest) {
  const std::vector<int> strides = {0, 1, 4, 17};
  RandomParametersMomentsTest<double>(1 << 20, 40, strides, kZLimit);
}

class MockGenerator {
 public:
  explicit MockGenerator(uint64 seed) : counter_(seed) {}
  using ResultType = std::vector<uint32>;
  using ResultElementType = uint32;
  static constexpr int kResultElementCount = 1;
  ResultType operator()() {
    ResultType result;
    result.push_back(counter_++);
    return result;
  }

 private:
  uint32 counter_;
};

template <typename T>
void SingleSampleAdapterSkipTest() {
  std::vector<uint64> skips(10);
  std::vector<uint64> skip_afters(10);
  std::iota(skips.begin(), skips.end(), 0);
  std::iota(skip_afters.begin(), skip_afters.end(), 0);
  uint64 total_samples = 100;
  uint64 seed = GetTestSeed();

  for (uint64 skip : skips) {
    for (uint64 skip_after : skip_afters) {
      // Baseline rngs.
      T parent_gen(seed);
      SingleSampleAdapter<T> gen(&parent_gen);

      // Rng on which Skip() is performed.
      T parent_gen_to_skip(seed);
      SingleSampleAdapter<T> gen_to_skip(&parent_gen_to_skip);

      // Skip over `skip_after` samples from both `gen` and `gen_to_skip`.
      int cur = 0;
      for (; cur < skip_after; cur++) {
        gen();
        gen_to_skip();
      }

      // Skip over `skip_` samples from `gen` iteratively.
      for (; cur < skip_after + skip; cur++) {
        gen();
      }

      // Skip over `skip_` samples from `gen_to_skip` by calling `Skip()`.
      gen_to_skip.Skip(skip);

      // Assert that they produce same outputs afterwards.
      for (; cur < total_samples; cur++) {
        ASSERT_EQ(gen(), gen_to_skip());
      }
    }
  }
}

TEST(SingleSampleAdapterTest, PhiloxRandomSkip) {
  SingleSampleAdapterSkipTest<PhiloxRandom>();
}

TEST(SingleSampleAdapterTest, MockGeneratorSkip) {
  SingleSampleAdapterSkipTest<MockGenerator>();
}

}  // namespace
}  // namespace random
}  // namespace tsl
