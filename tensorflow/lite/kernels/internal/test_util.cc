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
#include "tensorflow/lite/kernels/internal/test_util.h"

#include <cmath>
#include <iterator>

namespace tflite {

// this is a copied from an internal function in propagate_fixed_sizes.cc
bool ComputeConvSizes(const RuntimeShape& input_shape, int output_depth,
                      int filter_width, int filter_height, int stride,
                      int dilation_width_factor, int dilation_height_factor,
                      PaddingType padding_type, RuntimeShape* output_shape,
                      int* pad_width, int* pad_height) {
  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int batch = input_shape.Dims(0);

  int dilated_filter_width = dilation_width_factor * (filter_width - 1) + 1;
  int dilated_filter_height = dilation_height_factor * (filter_height - 1) + 1;

  int output_height = 0;
  int output_width = 0;
  if (padding_type == PaddingType::kValid) {
    // Official TF is
    // ceil((input_height - (dilated_filter_height - 1)) / stride),
    // implemented as
    // floor(
    //   (input_height - (dilated_filter_height - 1) + (stride - 1)) / stride).
    output_height = (input_height + stride - dilated_filter_height) / stride;
    output_width = (input_width + stride - dilated_filter_width) / stride;
  } else if (padding_type == PaddingType::kSame) {
    output_height = (input_height + stride - 1) / stride;
    output_width = (input_width + stride - 1) / stride;
  } else {
    return false;
  }

  if (output_width <= 0 || output_height <= 0) {
    return false;
  }

  *pad_height = std::max(
      0, ((output_height - 1) * stride + dilated_filter_height - input_height) /
             2);
  *pad_width = std::max(
      0,
      ((output_width - 1) * stride + dilated_filter_width - input_width) / 2);

  output_shape->BuildFrom({batch, output_height, output_width, output_depth});
  return true;
}

std::mt19937& RandomEngine() {
  static std::mt19937 engine;
  return engine;
}

int UniformRandomInt(int min, int max) {
  std::uniform_int_distribution<int> dist(min, max);
  return dist(RandomEngine());
}

float UniformRandomFloat(float min, float max) {
  std::uniform_real_distribution<float> dist(min, max);
  return dist(RandomEngine());
}

int ExponentialRandomPositiveInt(float percentile, int percentile_val,
                                 int max_val) {
  const float lambda =
      -std::log(1.f - percentile) / static_cast<float>(percentile_val);
  std::exponential_distribution<float> dist(lambda);
  float val;
  do {
    val = dist(RandomEngine());
  } while (!val || !std::isfinite(val) || val > max_val);
  return static_cast<int>(std::ceil(val));
}

float ExponentialRandomPositiveFloat(float percentile, float percentile_val,
                                     float max_val) {
  const float lambda =
      -std::log(1.f - percentile) / static_cast<float>(percentile_val);
  std::exponential_distribution<float> dist(lambda);
  float val;
  do {
    val = dist(RandomEngine());
  } while (!std::isfinite(val) || val > max_val);
  return val;
}

void FillRandom(std::vector<float>* vec, float min, float max) {
  std::uniform_real_distribution<float> dist(min, max);
  // TODO(b/154540105): use std::ref to avoid copying the random engine.
  auto gen = std::bind(dist, RandomEngine());
  std::generate(std::begin(*vec), std::end(*vec), gen);
}

}  // namespace tflite
