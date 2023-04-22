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

#include "tensorflow/core/lib/random/philox_random.h"

#include <math.h>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/lib/random/philox_random_test_utils.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace random {
namespace {

// A trivial distribution that just returns the PhiloxRandom as a distribution
class TrivialPhiloxDistribution {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = PhiloxRandom::kResultElementCount;
  typedef PhiloxRandom::ResultType ResultType;
  typedef PhiloxRandom::ResultElementType ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(PhiloxRandom* gen) { return (*gen)(); }
};

// This test checks that skipping certain number of samples, is equivalent to
// generate the same number of samples without skipping.
TEST(PhiloxRandomTest, SkipMatchTest) {
  constexpr int count = 1024;
  constexpr int skip_count = 2048;

  uint64 test_seed = GetTestSeed();
  std::vector<uint32> v1(count);
  {
    PhiloxRandom gen(test_seed);
    gen.Skip(skip_count / 4);
    FillRandoms<TrivialPhiloxDistribution>(gen, &v1[0], v1.size());
  }

  std::vector<uint32> v2(count + skip_count);
  {
    PhiloxRandom gen(test_seed);
    FillRandoms<TrivialPhiloxDistribution>(gen, &v2[0], v2.size());
  }

  for (int i = 0; i < count; ++i) {
    ASSERT_EQ(v1[i], v2[i + skip_count]);
  }
}

}  // namespace
}  // namespace random
}  // namespace tensorflow
