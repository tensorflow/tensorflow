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

#include "xla/tsl/lib/random/weighted_picker.h"

#include <string.h>

#include <vector>

#include "xla/tsl/lib/random/simple_philox.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"
#include "tsl/platform/types.h"

namespace tsl {
namespace random {

static void TestPicker(SimplePhilox* rnd, int size);
static void CheckUniform(SimplePhilox* rnd, WeightedPicker* picker, int trials);
static void CheckSkewed(SimplePhilox* rnd, WeightedPicker* picker, int trials);
static void TestPickAt(int items, const int32* weights);

TEST(WeightedPicker, Simple) {
  PhiloxRandom philox(testing::RandomSeed(), 17);
  SimplePhilox rnd(&philox);

  {
    VLOG(0) << "======= Zero-length picker";
    WeightedPicker picker(0);
    EXPECT_EQ(picker.Pick(&rnd), -1);
  }

  {
    VLOG(0) << "======= Singleton picker";
    WeightedPicker picker(1);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    EXPECT_EQ(picker.Pick(&rnd), 0);
  }

  {
    VLOG(0) << "======= Grown picker";
    WeightedPicker picker(0);
    for (int i = 0; i < 10; i++) {
      picker.Append(1);
    }
    CheckUniform(&rnd, &picker, 100000);
  }

  {
    VLOG(0) << "======= Grown picker with zero weights";
    WeightedPicker picker(1);
    picker.Resize(10);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    EXPECT_EQ(picker.Pick(&rnd), 0);
  }

  {
    VLOG(0) << "======= Shrink picker and check weights";
    WeightedPicker picker(1);
    picker.Resize(10);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    EXPECT_EQ(picker.Pick(&rnd), 0);
    for (int i = 0; i < 10; i++) {
      picker.set_weight(i, i);
    }
    EXPECT_EQ(picker.total_weight(), 45);
    picker.Resize(5);
    EXPECT_EQ(picker.total_weight(), 10);
    picker.Resize(2);
    EXPECT_EQ(picker.total_weight(), 1);
    picker.Resize(1);
    EXPECT_EQ(picker.total_weight(), 0);
  }
}

TEST(WeightedPicker, BigWeights) {
  PhiloxRandom philox(testing::RandomSeed() + 1, 17);
  SimplePhilox rnd(&philox);
  VLOG(0) << "======= Check uniform with big weights";
  WeightedPicker picker(2);
  picker.SetAllWeights(2147483646L / 3);  // (2^31 - 2) / 3
  CheckUniform(&rnd, &picker, 100000);
}

TEST(WeightedPicker, Deterministic) {
  VLOG(0) << "======= Testing deterministic pick";
  static const int32 weights[] = {1, 0, 200, 5, 42};
  TestPickAt(TF_ARRAYSIZE(weights), weights);
}

TEST(WeightedPicker, Randomized) {
  PhiloxRandom philox(testing::RandomSeed() + 10, 17);
  SimplePhilox rnd(&philox);
  TestPicker(&rnd, 1);
  TestPicker(&rnd, 2);
  TestPicker(&rnd, 3);
  TestPicker(&rnd, 4);
  TestPicker(&rnd, 7);
  TestPicker(&rnd, 8);
  TestPicker(&rnd, 9);
  TestPicker(&rnd, 10);
  TestPicker(&rnd, 100);
}

static void TestPicker(SimplePhilox* rnd, int size) {
  VLOG(0) << "======= Testing size " << size;

  // Check that empty picker returns -1
  {
    WeightedPicker picker(size);
    picker.SetAllWeights(0);
    for (int i = 0; i < 100; i++) EXPECT_EQ(picker.Pick(rnd), -1);
  }

  // Create zero weights array
  std::vector<int32> weights(size);
  for (int elem = 0; elem < size; elem++) {
    weights[elem] = 0;
  }

  // Check that singleton picker always returns the same element
  for (int elem = 0; elem < size; elem++) {
    WeightedPicker picker(size);
    picker.SetAllWeights(0);
    picker.set_weight(elem, elem + 1);
    for (int i = 0; i < 100; i++) EXPECT_EQ(picker.Pick(rnd), elem);
    weights[elem] = 10;
    picker.SetWeightsFromArray(size, &weights[0]);
    for (int i = 0; i < 100; i++) EXPECT_EQ(picker.Pick(rnd), elem);
    weights[elem] = 0;
  }

  // Check that uniform picker generates elements roughly uniformly
  {
    WeightedPicker picker(size);
    CheckUniform(rnd, &picker, 100000);
  }

  // Check uniform picker that was grown piecemeal
  if (size / 3 > 0) {
    WeightedPicker picker(size / 3);
    while (picker.num_elements() != size) {
      picker.Append(1);
    }
    CheckUniform(rnd, &picker, 100000);
  }

  // Check that skewed distribution works
  if (size <= 10) {
    // When picker grows one element at a time
    WeightedPicker picker(size);
    int32_t weight = 1;
    for (int elem = 0; elem < size; elem++) {
      picker.set_weight(elem, weight);
      weights[elem] = weight;
      weight *= 2;
    }
    CheckSkewed(rnd, &picker, 1000000);

    // When picker is created from an array
    WeightedPicker array_picker(0);
    array_picker.SetWeightsFromArray(size, &weights[0]);
    CheckSkewed(rnd, &array_picker, 1000000);
  }
}

static void CheckUniform(SimplePhilox* rnd, WeightedPicker* picker,
                         int trials) {
  const int size = picker->num_elements();
  int* count = new int[size];
  memset(count, 0, sizeof(count[0]) * size);
  for (int i = 0; i < size * trials; i++) {
    const int elem = picker->Pick(rnd);
    EXPECT_GE(elem, 0);
    EXPECT_LT(elem, size);
    count[elem]++;
  }
  const int expected_min = int(0.9 * trials);
  const int expected_max = int(1.1 * trials);
  for (int i = 0; i < size; i++) {
    EXPECT_GE(count[i], expected_min);
    EXPECT_LE(count[i], expected_max);
  }
  delete[] count;
}

static void CheckSkewed(SimplePhilox* rnd, WeightedPicker* picker, int trials) {
  const int size = picker->num_elements();
  int* count = new int[size];
  memset(count, 0, sizeof(count[0]) * size);
  for (int i = 0; i < size * trials; i++) {
    const int elem = picker->Pick(rnd);
    EXPECT_GE(elem, 0);
    EXPECT_LT(elem, size);
    count[elem]++;
  }

  for (int i = 0; i < size - 1; i++) {
    LOG(INFO) << i << ": " << count[i];
    const float ratio = float(count[i + 1]) / float(count[i]);
    EXPECT_GE(ratio, 1.6f);
    EXPECT_LE(ratio, 2.4f);
  }
  delete[] count;
}

static void TestPickAt(int items, const int32* weights) {
  WeightedPicker picker(items);
  picker.SetWeightsFromArray(items, weights);
  int weight_index = 0;
  for (int i = 0; i < items; ++i) {
    for (int j = 0; j < weights[i]; ++j) {
      int pick = picker.PickAt(weight_index);
      EXPECT_EQ(pick, i);
      ++weight_index;
    }
  }
  EXPECT_EQ(weight_index, picker.total_weight());
}

static void BM_Create(::testing::benchmark::State& state) {
  int arg = state.range(0);
  for (auto s : state) {
    WeightedPicker p(arg);
  }
}
BENCHMARK(BM_Create)->Range(1, 1024);

static void BM_CreateAndSetWeights(::testing::benchmark::State& state) {
  int arg = state.range(0);
  std::vector<int32> weights(arg);
  for (int i = 0; i < arg; i++) {
    weights[i] = i * 10;
  }
  for (auto s : state) {
    WeightedPicker p(arg);
    p.SetWeightsFromArray(arg, &weights[0]);
  }
}
BENCHMARK(BM_CreateAndSetWeights)->Range(1, 1024);

static void BM_Pick(::testing::benchmark::State& state) {
  int arg = state.range(0);
  PhiloxRandom philox(301, 17);
  SimplePhilox rnd(&philox);
  WeightedPicker p(arg);
  int result = 0;
  for (auto s : state) {
    result += p.Pick(&rnd);
  }
  VLOG(4) << result;  // Dummy use
}
BENCHMARK(BM_Pick)->Range(1, 1024);

}  // namespace random
}  // namespace tsl
