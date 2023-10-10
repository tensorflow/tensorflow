/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBJPEG_DECODER_TEST_HELPER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBJPEG_DECODER_TEST_HELPER_H_

#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

// Checks if the values are almost equal and their absolute difference doesn't
// exceed tolerance.
// Params: int tolerance. arg is int.
MATCHER_P(AreAlmostEqualWithTolerance, tolerance, "") {
  int a = static_cast<int>(testing::get<0>(arg));
  int b = static_cast<int>(testing::get<1>(arg));
  int diff = std::abs(b - a);
  if (result_listener) {
    std::ostringstream os;
    os << "difference between " << a << " and " << b << " is " << diff
       << " which is greater than " << tolerance << ". ";
    *result_listener << os.str();
  }
  return diff <= tolerance;
}

// Matcher for matching the 13x13 yellow-white chessboard pattern in an image.
// Parameters: int tolerance. arg is vector<uint8_t> image.
// Checks that relative difference between pixel values is within tolerance.
MATCHER_P(HasChessboardPatternWithTolerance, tolerance, "") {
  const std::vector<uint8_t> image = static_cast<std::vector<uint8_t>>(arg);
  const std::vector<uint8_t> kYellow = {253, 242, 0};
  const std::vector<uint8_t> kWhite = {255, 255, 255};
  const int kHeightRect = 23;
  const int kWidthRect = 19;
  const int kChannels = 3;
  // Rectangles on the chessboard have fixed height and width.
  // Check color at centre pixel for every rectangle.
  // There are 13x13 rectangles in all.
  const int row_stride = kChannels * 250;
  int rect = 0;

  for (int i = 0; i < 13; i++)
    for (int j = 0; j < 13; j++) {
      int row = i * kHeightRect + (kHeightRect / 2);
      int col = j * kWidthRect + (kWidthRect / 2);
      int pixel = row * row_stride + col * kChannels;
      std::vector<uint8_t> decoded_color = {image[pixel], image[pixel + 1],
                                            image[pixel + 2]};
      if (!testing::ExplainMatchResult(
              testing::Pointwise(AreAlmostEqualWithTolerance(tolerance),
                                 (rect & 1) ? kWhite : kYellow),
              decoded_color, result_listener)) {
        *result_listener << "Pixel values at row#" << row << ", col#" << col
                         << " don't match.";

        return false;
      }
      rect++;
    }
  return true;
}

// Matcher for matching the
// SMPTE(https://en.wikipedia.org/wiki/SMPTE_color_bars) rainbow color bars
// pattern in an image. Parameters: int tolerance. arg is vector<uint8_t> image.
// Verifies that relative difference between pixel values is within tolerance.
MATCHER_P(HasRainbowPatternWithTolerance, tolerance, "") {
  const std::vector<uint8_t> image = static_cast<std::vector<uint8_t>>(arg);

  const int kWidth = 250;
  const int kRow = 150;
  const int kChannels = 3;
  const int kBandSize = 35;
  std::vector<std::vector<uint8_t>> colors = {
      {192, 192, 192}, {192, 192, 0}, {0, 193, 192}, {0, 192, 0},
      {192, 0, 192},   {193, 0, 0},   {0, 0, 193}};
  // Match pixel colors at 7 locations in a fixed row.
  for (int i = 0; i < 7; ++i) {
    int col = (i * kBandSize + kBandSize / 2);
    int pixel = kRow * kWidth * kChannels + col * kChannels;
    std::vector<uint8_t> decoded_color = {image[pixel], image[pixel + 1],
                                          image[pixel + 2]};
    if (!testing::ExplainMatchResult(
            testing::Pointwise(AreAlmostEqualWithTolerance(tolerance),
                               colors[i]),
            decoded_color, result_listener)) {
      *result_listener << "Pixel values at row#" << kRow << ", col#" << col
                       << " don't match.";

      return false;
    }
  }
  return true;
}

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBJPEG_DECODER_TEST_HELPER_H_
