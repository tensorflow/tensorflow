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
#include "tensorflow/lite/testing/generate_testspec.h"

#include <random>

#include <gtest/gtest.h>

namespace tflite {
namespace testing {
namespace {

TEST(GenerateRandomTensor, FloatValue) {
  std::mt19937 random_engine;
  auto random_func = [&]() {
    return std::uniform_real_distribution<float>(-0.5, 0.5)(random_engine);
  };

  std::set<float> values;
  float sum_x_square = 0.0f;
  float sum_x = 0.0f;
  for (int i = 0; i < 100; i++) {
    const auto& data = GenerateRandomTensor<float>({1, 3, 4}, random_func);
    for (float value : data) {
      values.insert(value);
      sum_x_square += value * value;
      sum_x += value;
    }
  }

  // Eech round, generated tensor has different values.
  EXPECT_GT(values.size(), 200);
  int num = 1 * 3 * 4 * 100;
  float stddev = sum_x_square / num - (sum_x / num) * (sum_x / num);

  // Stddev is greater than 1/2 stddev of uniform distribution: (B-A)^2 / 12
  float minstddev = 1.0f / 12 / 2;
  EXPECT_GT(stddev, minstddev);
}

}  // namespace
}  // namespace testing
}  // namespace tflite
