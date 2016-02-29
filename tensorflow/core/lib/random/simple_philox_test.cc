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

#include <set>
#include <string>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace random {
namespace {

TEST(SimplePhiloxTest, FloatTest) {
  PhiloxRandom philox(7, 7);
  SimplePhilox gen(&philox);
  static const int kIters = 1000000;
  for (int i = 0; i < kIters; ++i) {
    float f = gen.RandFloat();
    EXPECT_LE(0.0f, f);
    EXPECT_GT(1.0f, f);
  }
  for (int i = 0; i < kIters; ++i) {
    double d = gen.RandDouble();
    EXPECT_LE(0.0, d);
    EXPECT_GT(1.0, d);
  }
}

static void DifferenceTest(const char *names, SimplePhilox *gen1,
                           SimplePhilox *gen2) {
  static const int kIters = 100;
  bool different = false;
  for (int i = 0; i < kIters; ++i) {
    if (gen1->Rand32() != gen2->Rand32()) {
      different = true;
      break;
    }
  }
  CHECK(different) << "different seeds but same output!";
}

TEST(SimplePhiloxTest, DifferenceTest) {
  PhiloxRandom philox1(1, 1), philox2(17, 17);
  SimplePhilox gen1(&philox1), gen2(&philox2);

  DifferenceTest("SimplePhilox: different seeds", &gen1, &gen2);
}

TEST(SimplePhiloxTest, DifferenceTestCloseSeeds) {
  PhiloxRandom philox1(1, 1), philox2(2, 1);
  SimplePhilox gen1(&philox1), gen2(&philox2);

  DifferenceTest("SimplePhilox: close seeds", &gen1, &gen2);
}

TEST(SimplePhiloxTest, Regression_CloseSeedsAreDifferent) {
  const int kCount = 1000;

  // Two seeds differ only by the last bit.
  PhiloxRandom philox1(0, 1), philox2(1, 1);
  SimplePhilox gen1(&philox1), gen2(&philox2);

  std::set<uint32> first;
  std::set<uint32> all;
  for (int i = 0; i < kCount; ++i) {
    uint32 v = gen1.Rand32();
    first.insert(v);
    all.insert(v);
    all.insert(gen2.Rand32());
  }

  // Broken array initialization implementation (before 2009-08-18) using the
  // above seeds return <1000, 1007>, generating output that is >99% similar.
  // The fix returns <1000, 2000> for completely disjoint sets.
  EXPECT_EQ(kCount, first.size());
  EXPECT_EQ(2 * kCount, all.size());
}

TEST(SimplePhiloxTest, TestUniform) {
  PhiloxRandom philox(17, 17);
  SimplePhilox gen(&philox);

  uint32 range = 3 * (1L << 29);
  uint32 threshold = 1L << 30;

  size_t count = 0;
  static const int kTrials = 100000;
  for (int i = 0; i < kTrials; ++i) {
    uint32 rnd = gen.Uniform(range);
    if (rnd < threshold) {
      ++count;
    }
  }

  EXPECT_LT(fabs((threshold + 0.0) / range - (count + 0.0) / kTrials), 0.005);
}

TEST(SimplePhiloxTest, TestUniform64) {
  PhiloxRandom philox(17, 17);
  SimplePhilox gen(&philox);

  uint64 range = 3 * (1LL << 59);
  uint64 threshold = 1LL << 60;

  size_t count = 0;
  static const int kTrials = 100000;
  for (int i = 0; i < kTrials; ++i) {
    uint64 rnd = gen.Uniform64(range);
    if (rnd < threshold) {
      ++count;
    }
  }

  EXPECT_LT(fabs((threshold + 0.0) / range - (count + 0.0) / kTrials), 0.005);
}

}  // namespace
}  // namespace random
}  // namespace tensorflow
