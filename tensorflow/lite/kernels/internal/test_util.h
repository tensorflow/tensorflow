/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TEST_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TEST_UTIL_H_

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

// Computes output and padding dimensions.
bool ComputeConvSizes(const RuntimeShape& input_shape, int output_depth,
                      int filter_width, int filter_height, int stride,
                      int dilation_width_factor, int dilation_height_factor,
                      PaddingType padding_type, RuntimeShape* output_shape,
                      int* pad_width, int* pad_height);

// Returns a mt19937 random engine.
std::mt19937& RandomEngine();

// Returns a random integer uniformly distributed between |min| and |max|.
int UniformRandomInt(int min, int max);

// Returns a random float uniformly distributed between |min| and |max|.
float UniformRandomFloat(float min, float max);

// Returns a random element in |v|.
template <typename T>
const T& RandomElement(const std::vector<T>& v) {
  return v[UniformRandomInt(0, v.size() - 1)];
}

// Returns a random exponentially distributed integer.
int ExponentialRandomPositiveInt(float percentile, int percentile_val,
                                 int max_val);

// Returns a random exponentially distributed float.
float ExponentialRandomPositiveFloat(float percentile, float percentile_val,
                                     float max_val);

// Fills a vector with random floats between |min| and |max|.
void FillRandom(std::vector<float>* vec, float min, float max);

// Fills a vector with random numbers between |min| and |max|.
template <typename T>
void FillRandom(std::vector<T>* vec, T min, T max) {
  std::uniform_int_distribution<T> dist(min, max);
  auto gen = std::bind(dist, RandomEngine());
  std::generate(std::begin(*vec), std::end(*vec), gen);
}

// Fills a vector with random numbers.
template <typename T>
void FillRandom(std::vector<T>* vec) {
  FillRandom(vec, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
}

template <typename T>
void FillRandom(typename std::vector<T>::iterator begin_it,
                typename std::vector<T>::iterator end_it, T min, T max) {
  std::uniform_int_distribution<T> dist(min, max);
  auto gen = std::bind(dist, RandomEngine());
  std::generate(begin_it, end_it, gen);
}

// Fill with a "skyscraper" pattern, in which there is a central section (across
// the depth) with higher values than the surround.
template <typename T>
void FillRandomSkyscraper(std::vector<T>* vec, int depth,
                          double middle_proportion, uint8 middle_min,
                          uint8 sides_max) {
  for (auto base_it = std::begin(*vec); base_it != std::end(*vec);
       base_it += depth) {
    auto left_it = base_it + std::ceil(0.5 * depth * (1.0 - middle_proportion));
    auto right_it =
        base_it + std::ceil(0.5 * depth * (1.0 + middle_proportion));
    FillRandom(base_it, left_it, std::numeric_limits<T>::min(), sides_max);
    FillRandom(left_it, right_it, middle_min, std::numeric_limits<T>::max());
    FillRandom(right_it, base_it + depth, std::numeric_limits<T>::min(),
               sides_max);
  }
}

}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TEST_UTIL_H_
