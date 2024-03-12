/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/stablehlo_reduce_window_test_util.h"

#include <functional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite::reduce_window::reference {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TEST(ReferenceTest, DilateWorks) {
  reference::Tensor<int> input = reference::Tensor<int>::iota(/*shape=*/{3, 3});
  reference::Tensor<int> output =
      reference::Dilate(input, /*dilations=*/{2, 3}, /*padding_value=*/-1);

  EXPECT_THAT(output.data, ElementsAreArray({
                               // clang-format off
                                1, -1, -1,  2, -1, -1,  3,
                               -1, -1, -1, -1, -1, -1, -1,
                                4, -1, -1,  5, -1, -1,  6,
                               -1, -1, -1, -1, -1, -1, -1,
                                7, -1, -1,  8, -1, -1,  9
                               // clang-format on
                           }));
}

TEST(ReferenceTest, PadWorks) {
  reference::Tensor<int> input = reference::Tensor<int>::iota(/*shape=*/{3, 3});
  reference::Tensor<int> output =
      reference::Pad(input, /*padding=*/{1, 2, 3, 4}, /*padding_value=*/-1);

  EXPECT_THAT(output.data,
              ElementsAreArray({
                  // clang-format off
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                  -1, -1, -1,  1,  2,  3, -1, -1, -1, -1,
                  -1, -1, -1,  4,  5,  6, -1, -1, -1, -1,
                  -1, -1, -1,  7,  8,  9, -1, -1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                  // clang-format on
              }));
}

TEST(ReferenceTest, PadIgnoresNegativeValues) {
  reference::Tensor<int> input = reference::Tensor<int>::iota(/*shape=*/{3, 3});
  reference::Tensor<int> output =
      reference::Pad(input, /*padding=*/{-1, -1, -1, -1}, /*padding_value=*/-1);

  EXPECT_THAT(output.data, ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST(ReferenceTest, CropWorks) {
  reference::Tensor<int> input =
      reference::Tensor<int>::iota(/*shape=*/{6, 10});
  reference::Tensor<int> output =
      reference::Crop(input, /*cropping=*/{-4, -1, -2, -3});

  EXPECT_THAT(output.data, ElementsAreArray({43, 44, 45, 46, 47}));
}

TEST(ReferenceTest, CropIgnoresPositiveValues) {
  reference::Tensor<int> input = reference::Tensor<int>::iota(/*shape=*/{3, 3});
  reference::Tensor<int> output =
      reference::Crop(input, /*cropping=*/{0, 0, 0, 0});

  EXPECT_THAT(output.data, ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST(ReferenceTest, WindowCopyWorks) {
  reference::Tensor<int> input = reference::Tensor<int>::iota(/*shape=*/{6, 4});
  EXPECT_THAT(reference::WindowCopy(input, /*window_dimensions=*/{2, 2},
                                    /*window_dilations=*/{2, 2},
                                    /*window_offset=*/{2, 1})
                  .data,
              ElementsAreArray({10, 12, 18, 20}));
}

TEST(ReferenceTest, RandomJaxReference0) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, -1, 0, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(19, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {0, 0, 0, 0, 0, 0, 0, 0, 4,   6,   8,   10,  12,  14,  16,  18,
           0, 0, 0, 0, 0, 0, 0, 0, 24,  26,  28,  30,  32,  34,  36,  38,
           0, 0, 0, 0, 0, 0, 0, 0, 44,  46,  48,  50,  52,  54,  56,  58,
           0, 0, 0, 0, 0, 0, 0, 0, 64,  66,  68,  70,  72,  74,  76,  78,
           0, 0, 0, 0, 0, 0, 0, 0, 84,  86,  88,  90,  92,  94,  96,  98,
           0, 0, 0, 0, 0, 0, 0, 0, 104, 106, 108, 110, 112, 114, 116, 118,
           0, 0, 0, 0, 0, 0, 0, 0, 124, 126, 128, 130, 132, 134, 136, 138,
           0, 0, 0, 0, 0, 0, 0, 0, 144, 146, 148, 150, 152, 154, 156, 158,
           0, 0, 0, 0, 0, 0, 0, 0, 164, 166, 168, 170, 172, 174, 176, 178,
           0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(ReferenceTest, RandomJaxReference1) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, -1, 1, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(6, 18));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0,
                        0, 0,   0, 0,   0, 3,   0, 5,   0, 7,   0, 9,   0, 11,
                        0, 13,  0, 15,  0, 17,  0, 19,  0, 43,  0, 45,  0, 47,
                        0, 49,  0, 51,  0, 53,  0, 55,  0, 57,  0, 59,  0, 83,
                        0, 85,  0, 87,  0, 89,  0, 91,  0, 93,  0, 95,  0, 97,
                        0, 99,  0, 123, 0, 125, 0, 127, 0, 129, 0, 131, 0, 133,
                        0, 135, 0, 137, 0, 139, 0, 163, 0, 165, 0, 167, 0, 169,
                        0, 171, 0, 173, 0, 175, 0, 177, 0, 179}));
}

TEST(ReferenceTest, RandomJaxReference2) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{2, -2, -2, 2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(8, 4));

  EXPECT_THAT(res.data,
              ElementsAreArray({5,  7,  9,  9,  15, 17, 19, 19, 25, 27, 29,
                                29, 35, 37, 39, 39, 45, 47, 49, 49, 55, 57,
                                59, 59, 65, 67, 69, 69, 75, 77, 79, 79}));
}

TEST(ReferenceTest, RandomJaxReference3) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{0, 1, -1, 1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(6, 19));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, 2,           -2147483647, 3,           -2147483647,
           4,           -2147483647, 5,           -2147483647, 6,
           -2147483647, 7,           -2147483647, 8,           -2147483647,
           9,           -2147483647, 10,          -2147483647, -2147483647,
           22,          -2147483647, 23,          -2147483647, 24,
           -2147483647, 25,          -2147483647, 26,          -2147483647,
           27,          -2147483647, 28,          -2147483647, 29,
           -2147483647, 30,          -2147483647, -2147483647, 42,
           -2147483647, 43,          -2147483647, 44,          -2147483647,
           45,          -2147483647, 46,          -2147483647, 47,
           -2147483647, 48,          -2147483647, 49,          -2147483647,
           50,          -2147483647, -2147483647, 62,          -2147483647,
           63,          -2147483647, 64,          -2147483647, 65,
           -2147483647, 66,          -2147483647, 67,          -2147483647,
           68,          -2147483647, 69,          -2147483647, 70,
           -2147483647, -2147483647, 82,          -2147483647, 83,
           -2147483647, 84,          -2147483647, 85,          -2147483647,
           86,          -2147483647, 87,          -2147483647, 88,
           -2147483647, 89,          -2147483647, 90,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference4) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-2, -2, -1, -2},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(3, 3));

  EXPECT_THAT(res.data,
              ElementsAreArray({46, 50, 54, 86, 90, 94, 126, 130, 134}));
}

TEST(ReferenceTest, RandomJaxReference5) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{1, 2, 1, 1},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(11, 6));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,    12,   14,   16,   18,   20,   1,    44,   96,   156,  224,
           300,  1,    384,  476,  576,  684,  800,  1,    924,  1056, 1196,
           1344, 1500, 1,    1664, 1836, 2016, 2204, 2400, 1,    2604, 2816,
           3036, 3264, 3500, 1,    3744, 3996, 4256, 4524, 4800, 1,    5084,
           5376, 5676, 5984, 6300, 1,    6624, 6956, 7296, 7644, 8000, 1,
           82,   84,   86,   88,   90,   1,    92,   94,   96,   98,   100}));
}

TEST(ReferenceTest, RandomJaxReference6) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{2, -1, 0, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 17));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,           -2147483647, 2,           -2147483647, 3,
           -2147483647, 4,           -2147483647, 5,           -2147483647,
           6,           -2147483647, 7,           -2147483647, 8,
           -2147483647, 9,           11,          -2147483647, 12,
           -2147483647, 13,          -2147483647, 14,          -2147483647,
           15,          -2147483647, 16,          -2147483647, 17,
           -2147483647, 18,          -2147483647, 19,          21,
           -2147483647, 22,          -2147483647, 23,          -2147483647,
           24,          -2147483647, 25,          -2147483647, 26,
           -2147483647, 27,          -2147483647, 28,          -2147483647,
           29,          31,          -2147483647, 32,          -2147483647,
           33,          -2147483647, 34,          -2147483647, 35,
           -2147483647, 36,          -2147483647, 37,          -2147483647,
           38,          -2147483647, 39,          41,          -2147483647,
           42,          -2147483647, 43,          -2147483647, 44,
           -2147483647, 45,          -2147483647, 46,          -2147483647,
           47,          -2147483647, 48,          -2147483647, 49,
           51,          -2147483647, 52,          -2147483647, 53,
           -2147483647, 54,          -2147483647, 55,          -2147483647,
           56,          -2147483647, 57,          -2147483647, 58,
           -2147483647, 59,          61,          -2147483647, 62,
           -2147483647, 63,          -2147483647, 64,          -2147483647,
           65,          -2147483647, 66,          -2147483647, 67,
           -2147483647, 68,          -2147483647, 69,          71,
           -2147483647, 72,          -2147483647, 73,          -2147483647,
           74,          -2147483647, 75,          -2147483647, 76,
           -2147483647, 77,          -2147483647, 78,          -2147483647,
           79,          81,          -2147483647, 82,          -2147483647,
           83,          -2147483647, 84,          -2147483647, 85,
           -2147483647, 86,          -2147483647, 87,          -2147483647,
           88,          -2147483647, 89}));
}

TEST(ReferenceTest, RandomJaxReference7) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-2, -2, 1, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(3, 11));

  EXPECT_THAT(res.data,
              ElementsAreArray({0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                0, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                0, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70}));
}

TEST(ReferenceTest, RandomJaxReference8) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{2, 1, -2, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(13, 3));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, 4,           6,           8,           14,
           16,          18,          24,          26,          28,
           34,          36,          38,          44,          46,
           48,          54,          56,          58,          64,
           66,          68,          74,          76,          78,
           84,          86,          88,          94,          96,
           98,          -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference9) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-1, 2, -2, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 7));

  EXPECT_THAT(res.data, ElementsAreArray(
                            {32, 33, 34, 35, 36, 37, 38, 42, 43, 44, 45, 46, 47,
                             48, 52, 53, 54, 55, 56, 57, 58, 62, 63, 64, 65, 66,
                             67, 68, 72, 73, 74, 75, 76, 77, 78, 82, 83, 84, 85,
                             86, 87, 88, 92, 93, 94, 95, 96, 97, 98, 82, 83, 84,
                             85, 86, 87, 88, 92, 93, 94, 95, 96, 97, 98}));
}

TEST(ReferenceTest, RandomJaxReference10) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{0, -1, 0, 2},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(17, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
           23, 24, 25, 26, 27, 28, 29, 30, 21, 22, 23, 24, 25, 26, 27, 28,
           29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 31, 32, 33, 34,
           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
           41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
           57, 58, 59, 60, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
           63, 64, 65, 66, 67, 68, 69, 70, 61, 62, 63, 64, 65, 66, 67, 68,
           69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 71, 72, 73, 74,
           75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
           81, 82, 83, 84, 85, 86, 87, 88, 89, 90}));
}

TEST(ReferenceTest, RandomJaxReference11) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, 0, 2, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(4, 6));

  EXPECT_THAT(res.data,
              ElementsAreArray({0,   22,  26, 30,  34,  38,  0,   62,
                                66,  70,  74, 78,  0,   102, 106, 110,
                                114, 118, 0,  142, 146, 150, 154, 158}));
}

TEST(ReferenceTest, RandomJaxReference12) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, -2, 1, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference13) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{1, 2, 1, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(13, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, 2,           4,           6,           8,
           -2147483647, 12,          14,          16,          18,
           -2147483647, 22,          24,          26,          28,
           -2147483647, 32,          34,          36,          38,
           -2147483647, 42,          44,          46,          48,
           -2147483647, 52,          54,          56,          58,
           -2147483647, 62,          64,          66,          68,
           -2147483647, 72,          74,          76,          78,
           -2147483647, 82,          84,          86,          88,
           -2147483647, 92,          94,          96,          98,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference14) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, 2, 1, -1},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(11, 9));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(ReferenceTest, RandomJaxReference15) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-2, -2, 1, 2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(3, 11));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 2147483646,
                        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 2147483646,
                        61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference16) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{0, 0, 0, 0},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(5, 19));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {1,          2147483646, 2,          2147483646, 3,
                   2147483646, 4,          2147483646, 5,          2147483646,
                   6,          2147483646, 7,          2147483646, 8,
                   2147483646, 9,          2147483646, 10,         21,
                   2147483646, 22,         2147483646, 23,         2147483646,
                   24,         2147483646, 25,         2147483646, 26,
                   2147483646, 27,         2147483646, 28,         2147483646,
                   29,         2147483646, 30,         41,         2147483646,
                   42,         2147483646, 43,         2147483646, 44,
                   2147483646, 45,         2147483646, 46,         2147483646,
                   47,         2147483646, 48,         2147483646, 49,
                   2147483646, 50,         61,         2147483646, 62,
                   2147483646, 63,         2147483646, 64,         2147483646,
                   65,         2147483646, 66,         2147483646, 67,
                   2147483646, 68,         2147483646, 69,         2147483646,
                   70,         81,         2147483646, 82,         2147483646,
                   83,         2147483646, 84,         2147483646, 85,
                   2147483646, 86,         2147483646, 87,         2147483646,
                   88,         2147483646, 89,         2147483646, 90}));
}

TEST(ReferenceTest, RandomJaxReference17) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, -1, 2, 1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 20));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           1,          2147483646, 1,          2147483646, 2,
           2147483646, 3,          2147483646, 4,          2147483646,
           5,          2147483646, 6,          2147483646, 7,
           2147483646, 8,          2147483646, 9,          2147483646,
           1,          2147483646, 1,          2147483646, 2,
           2147483646, 3,          2147483646, 4,          2147483646,
           5,          2147483646, 6,          2147483646, 7,
           2147483646, 8,          2147483646, 9,          2147483646,
           11,         2147483646, 11,         2147483646, 12,
           2147483646, 13,         2147483646, 14,         2147483646,
           15,         2147483646, 16,         2147483646, 17,
           2147483646, 18,         2147483646, 19,         2147483646,
           21,         2147483646, 21,         2147483646, 22,
           2147483646, 23,         2147483646, 24,         2147483646,
           25,         2147483646, 26,         2147483646, 27,
           2147483646, 28,         2147483646, 29,         2147483646,
           31,         2147483646, 31,         2147483646, 32,
           2147483646, 33,         2147483646, 34,         2147483646,
           35,         2147483646, 36,         2147483646, 37,
           2147483646, 38,         2147483646, 39,         2147483646,
           41,         2147483646, 41,         2147483646, 42,
           2147483646, 43,         2147483646, 44,         2147483646,
           45,         2147483646, 46,         2147483646, 47,
           2147483646, 48,         2147483646, 49,         2147483646,
           51,         2147483646, 51,         2147483646, 52,
           2147483646, 53,         2147483646, 54,         2147483646,
           55,         2147483646, 56,         2147483646, 57,
           2147483646, 58,         2147483646, 59,         2147483646,
           61,         2147483646, 61,         2147483646, 62,
           2147483646, 63,         2147483646, 64,         2147483646,
           65,         2147483646, 66,         2147483646, 67,
           2147483646, 68,         2147483646, 69,         2147483646,
           71,         2147483646, 71,         2147483646, 72,
           2147483646, 73,         2147483646, 74,         2147483646,
           75,         2147483646, 76,         2147483646, 77,
           2147483646, 78,         2147483646, 79,         2147483646}));
}

TEST(ReferenceTest, RandomJaxReference18) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{1, -2, -1, 0},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(9, 18));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1, 1,  1, 1,  1, 1,  1, 1,  1, 1,  1, 1,  1, 1,  1, 1,  1, 1,
           1, 2,  1, 3,  1, 4,  1, 5,  1, 6,  1, 7,  1, 8,  1, 9,  1, 10,
           1, 12, 1, 13, 1, 14, 1, 15, 1, 16, 1, 17, 1, 18, 1, 19, 1, 20,
           1, 22, 1, 23, 1, 24, 1, 25, 1, 26, 1, 27, 1, 28, 1, 29, 1, 30,
           1, 32, 1, 33, 1, 34, 1, 35, 1, 36, 1, 37, 1, 38, 1, 39, 1, 40,
           1, 42, 1, 43, 1, 44, 1, 45, 1, 46, 1, 47, 1, 48, 1, 49, 1, 50,
           1, 52, 1, 53, 1, 54, 1, 55, 1, 56, 1, 57, 1, 58, 1, 59, 1, 60,
           1, 62, 1, 63, 1, 64, 1, 65, 1, 66, 1, 67, 1, 68, 1, 69, 1, 70,
           1, 72, 1, 73, 1, 74, 1, 75, 1, 76, 1, 77, 1, 78, 1, 79, 1, 80}));
}

TEST(ReferenceTest, RandomJaxReference19) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{1, 0, 0, -1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 9));

  EXPECT_THAT(res.data, ElementsAreArray(
                            {1,  2,  3,  4,  5,  6,  7,  8,  9,  1,  2,  3,  4,
                             5,  6,  7,  8,  9,  11, 12, 13, 14, 15, 16, 17, 18,
                             19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33,
                             34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47,
                             48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62,
                             63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76,
                             77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89}));
}

TEST(ReferenceTest, RandomJaxReference20) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, 2, 1, -1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(11, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,  2,          4,          6,          8,          11,        12,
           14, 16,         18,         21,         22,         24,        26,
           28, 31,         32,         34,         36,         38,        41,
           42, 44,         46,         48,         51,         52,        54,
           56, 58,         61,         62,         64,         66,        68,
           71, 72,         74,         76,         78,         81,        82,
           84, 86,         88,         91,         92,         94,        96,
           98, 2147483646, 2147483646, 2147483646, 2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference21) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{1, 0, 1, -1},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(5, 9));

  EXPECT_THAT(res.data,
              ElementsAreArray({1,   2,   3,   4,   5,   6,   7,   8,   9,
                                32,  34,  36,  38,  40,  42,  44,  46,  48,
                                72,  74,  76,  78,  80,  82,  84,  86,  88,
                                112, 114, 116, 118, 120, 122, 124, 126, 128,
                                152, 154, 156, 158, 160, 162, 164, 166, 168}));
}

TEST(ReferenceTest, RandomJaxReference22) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-2, 2, -2, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 7));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {23,          24,          25,          26,          27,
           28,          29,          33,          34,          35,
           36,          37,          38,          39,          43,
           44,          45,          46,          47,          48,
           49,          53,          54,          55,          56,
           57,          58,          59,          63,          64,
           65,          66,          67,          68,          69,
           73,          74,          75,          76,          77,
           78,          79,          83,          84,          85,
           86,          87,          88,          89,          93,
           94,          95,          96,          97,          98,
           99,          -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference23) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{2, -2, 2, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 11));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   3,   5,   7,   9,
           11,  13,  15,  17,  19,  0,   11,  23,  25,  27,  29,  31,  33,  35,
           37,  39,  0,   21,  43,  45,  47,  49,  51,  53,  55,  57,  59,  0,
           31,  63,  65,  67,  69,  71,  73,  75,  77,  79,  0,   41,  83,  85,
           87,  89,  91,  93,  95,  97,  99,  0,   51,  103, 105, 107, 109, 111,
           113, 115, 117, 119, 0,   61,  123, 125, 127, 129, 131, 133, 135, 137,
           139, 0,   71,  143, 145, 147, 149, 151, 153, 155, 157, 159}));
}

TEST(ReferenceTest, RandomJaxReference24) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{2, 2, -2, -2},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(11, 6));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({3,   4,   5,   6,   7,   8,   16,  18,  20,  22,  24,
                        26,  36,  38,  40,  42,  44,  46,  56,  58,  60,  62,
                        64,  66,  76,  78,  80,  82,  84,  86,  96,  98,  100,
                        102, 104, 106, 116, 118, 120, 122, 124, 126, 136, 138,
                        140, 142, 144, 146, 156, 158, 160, 162, 164, 166, 176,
                        178, 180, 182, 184, 186, 93,  94,  95,  96,  97,  98}));
}

TEST(ReferenceTest, RandomJaxReference25) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{2, -1, 2, 2},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 14));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 1,
                        1, 1, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 1, 1,
                        1, 1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 1,
                        1, 1, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1, 1,
                        1, 1, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 1, 1,
                        1, 1, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 1, 1,
                        1, 1, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 1, 1,
                        1, 1, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 1, 1,
                        1, 1, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 1,
                        1, 1, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 1, 1}));
}

TEST(ReferenceTest, RandomJaxReference26) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{-1, 1, -1, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(17, 7));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference27) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{1, -2, 2, -2},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(8, 5));

  EXPECT_THAT(res.data,
              ElementsAreArray({0, 1,   3,   5,   7,   0, 12,  16,  20,  24,
                                0, 32,  36,  40,  44,  0, 52,  56,  60,  64,
                                0, 72,  76,  80,  84,  0, 92,  96,  100, 104,
                                0, 112, 116, 120, 124, 0, 132, 136, 140, 144}));
}

TEST(ReferenceTest, RandomJaxReference28) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-2, -2, 0, -2},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(6, 8));

  EXPECT_THAT(res.data, ElementsAreArray(
                            {21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34,
                             35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48,
                             51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64,
                             65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78}));
}

TEST(ReferenceTest, RandomJaxReference29) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-1, -1, 2, 0},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(4, 21));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2147483646, 2147483646, 11,         2147483646, 12,
                   2147483646, 13,         2147483646, 14,         2147483646,
                   15,         2147483646, 16,         2147483646, 17,
                   2147483646, 18,         2147483646, 19,         2147483646,
                   20,         2147483646, 2147483646, 31,         2147483646,
                   32,         2147483646, 33,         2147483646, 34,
                   2147483646, 35,         2147483646, 36,         2147483646,
                   37,         2147483646, 38,         2147483646, 39,
                   2147483646, 40,         2147483646, 2147483646, 51,
                   2147483646, 52,         2147483646, 53,         2147483646,
                   54,         2147483646, 55,         2147483646, 56,
                   2147483646, 57,         2147483646, 58,         2147483646,
                   59,         2147483646, 60,         2147483646, 2147483646,
                   71,         2147483646, 72,         2147483646, 73,
                   2147483646, 74,         2147483646, 75,         2147483646,
                   76,         2147483646, 77,         2147483646, 78,
                   2147483646, 79,         2147483646, 80}));
}

TEST(ReferenceTest, RandomJaxReference30) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-1, 1, -2, -1},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 4));

  EXPECT_THAT(res.data,
              ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(ReferenceTest, RandomJaxReference31) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-2, 1, -1, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(5, 16));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, 22,          -2147483647, 23,          -2147483647,
           24,          -2147483647, 25,          -2147483647, 26,
           -2147483647, 27,          -2147483647, 28,          -2147483647,
           29,          -2147483647, 42,          -2147483647, 43,
           -2147483647, 44,          -2147483647, 45,          -2147483647,
           46,          -2147483647, 47,          -2147483647, 48,
           -2147483647, 49,          -2147483647, 62,          -2147483647,
           63,          -2147483647, 64,          -2147483647, 65,
           -2147483647, 66,          -2147483647, 67,          -2147483647,
           68,          -2147483647, 69,          -2147483647, 82,
           -2147483647, 83,          -2147483647, 84,          -2147483647,
           85,          -2147483647, 86,          -2147483647, 87,
           -2147483647, 88,          -2147483647, 89,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference32) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-1, 2, -1, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(9, 5));

  EXPECT_THAT(res.data,
              ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(ReferenceTest, RandomJaxReference33) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{1, -1, 2, 1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(17, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           11,          12,          13,          14,          15,
           16,          17,          18,          19,          20,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           21,          22,          23,          24,          25,
           26,          27,          28,          29,          30,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           31,          32,          33,          34,          35,
           36,          37,          38,          39,          40,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           41,          42,          43,          44,          45,
           46,          47,          48,          49,          50,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           51,          52,          53,          54,          55,
           56,          57,          58,          59,          60,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           61,          62,          63,          64,          65,
           66,          67,          68,          69,          70,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           71,          72,          73,          74,          75,
           76,          77,          78,          79,          80,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           81,          82,          83,          84,          85,
           86,          87,          88,          89,          90,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference34) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, 2, 2, 2},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(12, 12));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,   2,   4,   6,   8,   10,  12,  14,  16,  18,  9,   10,  11,  12,
           24,  26,  28,  30,  32,  34,  36,  38,  19,  20,  21,  22,  44,  46,
           48,  50,  52,  54,  56,  58,  29,  30,  31,  32,  64,  66,  68,  70,
           72,  74,  76,  78,  39,  40,  41,  42,  84,  86,  88,  90,  92,  94,
           96,  98,  49,  50,  51,  52,  104, 106, 108, 110, 112, 114, 116, 118,
           59,  60,  61,  62,  124, 126, 128, 130, 132, 134, 136, 138, 69,  70,
           71,  72,  144, 146, 148, 150, 152, 154, 156, 158, 79,  80,  81,  82,
           164, 166, 168, 170, 172, 174, 176, 178, 89,  90,  91,  92,  184, 186,
           188, 190, 192, 194, 196, 198, 99,  100, 0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0}));
}

TEST(ReferenceTest, RandomJaxReference35) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{1, 2, 1, -1},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(6, 9));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({11,  12,  13,  14,  15,  16,  17,  18,  19,  42,  44,
                        46,  48,  50,  52,  54,  56,  58,  82,  84,  86,  88,
                        90,  92,  94,  96,  98,  122, 124, 126, 128, 130, 132,
                        134, 136, 138, 162, 164, 166, 168, 170, 172, 174, 176,
                        178, 91,  92,  93,  94,  95,  96,  97,  98,  99}));
}

TEST(ReferenceTest, RandomJaxReference36) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{2, 2, 2, 1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(11, 22));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, 1,           -2147483647, 2,
           -2147483647, 3,           -2147483647, 4,           -2147483647,
           5,           -2147483647, 6,           -2147483647, 7,
           -2147483647, 8,           -2147483647, 9,           -2147483647,
           10,          -2147483647, -2147483647, -2147483647, 11,
           -2147483647, 12,          -2147483647, 13,          -2147483647,
           14,          -2147483647, 15,          -2147483647, 16,
           -2147483647, 17,          -2147483647, 18,          -2147483647,
           19,          -2147483647, 20,          -2147483647, -2147483647,
           -2147483647, 21,          -2147483647, 22,          -2147483647,
           23,          -2147483647, 24,          -2147483647, 25,
           -2147483647, 26,          -2147483647, 27,          -2147483647,
           28,          -2147483647, 29,          -2147483647, 30,
           -2147483647, -2147483647, -2147483647, 31,          -2147483647,
           32,          -2147483647, 33,          -2147483647, 34,
           -2147483647, 35,          -2147483647, 36,          -2147483647,
           37,          -2147483647, 38,          -2147483647, 39,
           -2147483647, 40,          -2147483647, -2147483647, -2147483647,
           41,          -2147483647, 42,          -2147483647, 43,
           -2147483647, 44,          -2147483647, 45,          -2147483647,
           46,          -2147483647, 47,          -2147483647, 48,
           -2147483647, 49,          -2147483647, 50,          -2147483647,
           -2147483647, -2147483647, 51,          -2147483647, 52,
           -2147483647, 53,          -2147483647, 54,          -2147483647,
           55,          -2147483647, 56,          -2147483647, 57,
           -2147483647, 58,          -2147483647, 59,          -2147483647,
           60,          -2147483647, -2147483647, -2147483647, 61,
           -2147483647, 62,          -2147483647, 63,          -2147483647,
           64,          -2147483647, 65,          -2147483647, 66,
           -2147483647, 67,          -2147483647, 68,          -2147483647,
           69,          -2147483647, 70,          -2147483647, -2147483647,
           -2147483647, 71,          -2147483647, 72,          -2147483647,
           73,          -2147483647, 74,          -2147483647, 75,
           -2147483647, 76,          -2147483647, 77,          -2147483647,
           78,          -2147483647, 79,          -2147483647, 80,
           -2147483647, -2147483647, -2147483647, 81,          -2147483647,
           82,          -2147483647, 83,          -2147483647, 84,
           -2147483647, 85,          -2147483647, 86,          -2147483647,
           87,          -2147483647, 88,          -2147483647, 89,
           -2147483647, 90,          -2147483647, -2147483647, -2147483647,
           91,          -2147483647, 92,          -2147483647, 93,
           -2147483647, 94,          -2147483647, 95,          -2147483647,
           96,          -2147483647, 97,          -2147483647, 98,
           -2147483647, 99,          -2147483647, 100,         -2147483647,
           -2147483647, -2147483647, 91,          -2147483647, 92,
           -2147483647, 93,          -2147483647, 94,          -2147483647,
           95,          -2147483647, 96,          -2147483647, 97,
           -2147483647, 98,          -2147483647, 99,          -2147483647,
           100,         -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference37) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, 2, 1, 2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(18, 6));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {12,          14,          16,          18,          20,
           20,          22,          24,          26,          28,
           30,          30,          22,          24,          26,
           28,          30,          30,          32,          34,
           36,          38,          40,          40,          32,
           34,          36,          38,          40,          40,
           42,          44,          46,          48,          50,
           50,          42,          44,          46,          48,
           50,          50,          52,          54,          56,
           58,          60,          60,          52,          54,
           56,          58,          60,          60,          62,
           64,          66,          68,          70,          70,
           62,          64,          66,          68,          70,
           70,          72,          74,          76,          78,
           80,          80,          72,          74,          76,
           78,          80,          80,          82,          84,
           86,          88,          90,          90,          82,
           84,          86,          88,          90,          90,
           92,          94,          96,          98,          100,
           100,         92,          94,          96,          98,
           100,         100,         -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference38) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, -2, 1, 1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(17, 11));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,           2,           3,           4,           5,
           6,           7,           8,           9,           10,
           10,          -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, 11,          12,          13,
           14,          15,          16,          17,          18,
           19,          20,          20,          -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, 21,
           22,          23,          24,          25,          26,
           27,          28,          29,          30,          30,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, 31,          32,          33,          34,
           35,          36,          37,          38,          39,
           40,          40,          -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, 41,          42,
           43,          44,          45,          46,          47,
           48,          49,          50,          50,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           51,          52,          53,          54,          55,
           56,          57,          58,          59,          60,
           60,          -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, 61,          62,          63,
           64,          65,          66,          67,          68,
           69,          70,          70,          -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, 71,
           72,          73,          74,          75,          76,
           77,          78,          79,          80,          80,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, 81,          82,          83,          84,
           85,          86,          87,          88,          89,
           90,          90}));
}

TEST(ReferenceTest, RandomJaxReference39) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-1, -1, -2, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(15, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {0,   0,   0,   0,   0,   0,   0,   0, 36, 38, 40, 42, 44,  46,  48,
           50,  0,   0,   0,   0,   0,   0,   0, 0,  56, 58, 60, 62,  64,  66,
           68,  70,  0,   0,   0,   0,   0,   0, 0,  0,  76, 78, 80,  82,  84,
           86,  88,  90,  0,   0,   0,   0,   0, 0,  0,  0,  96, 98,  100, 102,
           104, 106, 108, 110, 0,   0,   0,   0, 0,  0,  0,  0,  116, 118, 120,
           122, 124, 126, 128, 130, 0,   0,   0, 0,  0,  0,  0,  0,   136, 138,
           140, 142, 144, 146, 148, 150, 0,   0, 0,  0,  0,  0,  0,   0,   156,
           158, 160, 162, 164, 166, 168, 170, 0, 0,  0,  0,  0,  0,   0,   0}));
}

TEST(ReferenceTest, RandomJaxReference40) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{2, -1, -2, 2},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(19, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1,  1,  1,  1,  1,  3,  5,  7,  9,  1,  3,  5,  7,  9,
                        1,  13, 15, 17, 19, 1,  13, 15, 17, 19, 1,  23, 25, 27,
                        29, 1,  23, 25, 27, 29, 1,  33, 35, 37, 39, 1,  33, 35,
                        37, 39, 1,  43, 45, 47, 49, 1,  43, 45, 47, 49, 1,  53,
                        55, 57, 59, 1,  53, 55, 57, 59, 1,  63, 65, 67, 69, 1,
                        63, 65, 67, 69, 1,  73, 75, 77, 79, 1,  73, 75, 77, 79,
                        1,  83, 85, 87, 89, 1,  83, 85, 87, 89, 1}));
}

TEST(ReferenceTest, RandomJaxReference41) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{-1, 2, -2, 0},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(18, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, 23,          24,
           25,          26,          27,          28,          29,
           30,          -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, 33,
           34,          35,          36,          37,          38,
           39,          40,          -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           43,          44,          45,          46,          47,
           48,          49,          50,          -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, 53,          54,          55,          56,
           57,          58,          59,          60,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, 63,          64,          65,
           66,          67,          68,          69,          70,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, 73,          74,
           75,          76,          77,          78,          79,
           80,          -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, 83,
           84,          85,          86,          87,          88,
           89,          90,          -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           93,          94,          95,          96,          97,
           98,          99,          100,         -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, 93,          94,          95,          96,
           97,          98,          99,          100}));
}

TEST(ReferenceTest, RandomJaxReference42) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, -1, -1, 1},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(15, 9));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {156,  182,  210,  240,  272,  306,  342,  380,  20,   506,  552,
           600,  650,  702,  756,  812,  870,  30,   506,  552,  600,  650,
           702,  756,  812,  870,  30,   1056, 1122, 1190, 1260, 1332, 1406,
           1482, 1560, 40,   1056, 1122, 1190, 1260, 1332, 1406, 1482, 1560,
           40,   1806, 1892, 1980, 2070, 2162, 2256, 2352, 2450, 50,   1806,
           1892, 1980, 2070, 2162, 2256, 2352, 2450, 50,   2756, 2862, 2970,
           3080, 3192, 3306, 3422, 3540, 60,   2756, 2862, 2970, 3080, 3192,
           3306, 3422, 3540, 60,   3906, 4032, 4160, 4290, 4422, 4556, 4692,
           4830, 70,   3906, 4032, 4160, 4290, 4422, 4556, 4692, 4830, 70,
           5256, 5402, 5550, 5700, 5852, 6006, 6162, 6320, 80,   5256, 5402,
           5550, 5700, 5852, 6006, 6162, 6320, 80,   6806, 6972, 7140, 7310,
           7482, 7656, 7832, 8010, 90,   6806, 6972, 7140, 7310, 7482, 7656,
           7832, 8010, 90}));
}

TEST(ReferenceTest, RandomJaxReference43) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{1, 0, -2, 1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(19, 18));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {2,  -2147483647, 3,  -2147483647, 4,   -2147483647, 5,  -2147483647,
           6,  -2147483647, 7,  -2147483647, 8,   -2147483647, 9,  -2147483647,
           10, -2147483647, 2,  -2147483647, 3,   -2147483647, 4,  -2147483647,
           5,  -2147483647, 6,  -2147483647, 7,   -2147483647, 8,  -2147483647,
           9,  -2147483647, 10, -2147483647, 12,  -2147483647, 13, -2147483647,
           14, -2147483647, 15, -2147483647, 16,  -2147483647, 17, -2147483647,
           18, -2147483647, 19, -2147483647, 20,  -2147483647, 12, -2147483647,
           13, -2147483647, 14, -2147483647, 15,  -2147483647, 16, -2147483647,
           17, -2147483647, 18, -2147483647, 19,  -2147483647, 20, -2147483647,
           22, -2147483647, 23, -2147483647, 24,  -2147483647, 25, -2147483647,
           26, -2147483647, 27, -2147483647, 28,  -2147483647, 29, -2147483647,
           30, -2147483647, 22, -2147483647, 23,  -2147483647, 24, -2147483647,
           25, -2147483647, 26, -2147483647, 27,  -2147483647, 28, -2147483647,
           29, -2147483647, 30, -2147483647, 32,  -2147483647, 33, -2147483647,
           34, -2147483647, 35, -2147483647, 36,  -2147483647, 37, -2147483647,
           38, -2147483647, 39, -2147483647, 40,  -2147483647, 32, -2147483647,
           33, -2147483647, 34, -2147483647, 35,  -2147483647, 36, -2147483647,
           37, -2147483647, 38, -2147483647, 39,  -2147483647, 40, -2147483647,
           42, -2147483647, 43, -2147483647, 44,  -2147483647, 45, -2147483647,
           46, -2147483647, 47, -2147483647, 48,  -2147483647, 49, -2147483647,
           50, -2147483647, 42, -2147483647, 43,  -2147483647, 44, -2147483647,
           45, -2147483647, 46, -2147483647, 47,  -2147483647, 48, -2147483647,
           49, -2147483647, 50, -2147483647, 52,  -2147483647, 53, -2147483647,
           54, -2147483647, 55, -2147483647, 56,  -2147483647, 57, -2147483647,
           58, -2147483647, 59, -2147483647, 60,  -2147483647, 52, -2147483647,
           53, -2147483647, 54, -2147483647, 55,  -2147483647, 56, -2147483647,
           57, -2147483647, 58, -2147483647, 59,  -2147483647, 60, -2147483647,
           62, -2147483647, 63, -2147483647, 64,  -2147483647, 65, -2147483647,
           66, -2147483647, 67, -2147483647, 68,  -2147483647, 69, -2147483647,
           70, -2147483647, 62, -2147483647, 63,  -2147483647, 64, -2147483647,
           65, -2147483647, 66, -2147483647, 67,  -2147483647, 68, -2147483647,
           69, -2147483647, 70, -2147483647, 72,  -2147483647, 73, -2147483647,
           74, -2147483647, 75, -2147483647, 76,  -2147483647, 77, -2147483647,
           78, -2147483647, 79, -2147483647, 80,  -2147483647, 72, -2147483647,
           73, -2147483647, 74, -2147483647, 75,  -2147483647, 76, -2147483647,
           77, -2147483647, 78, -2147483647, 79,  -2147483647, 80, -2147483647,
           82, -2147483647, 83, -2147483647, 84,  -2147483647, 85, -2147483647,
           86, -2147483647, 87, -2147483647, 88,  -2147483647, 89, -2147483647,
           90, -2147483647, 82, -2147483647, 83,  -2147483647, 84, -2147483647,
           85, -2147483647, 86, -2147483647, 87,  -2147483647, 88, -2147483647,
           89, -2147483647, 90, -2147483647, 92,  -2147483647, 93, -2147483647,
           94, -2147483647, 95, -2147483647, 96,  -2147483647, 97, -2147483647,
           98, -2147483647, 99, -2147483647, 100, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference44) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, -2, 2, -1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(17, 11));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, 1,           2,           3,
           4,           5,           6,           7,           8,
           9,           -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, 11,
           12,          13,          14,          15,          16,
           17,          18,          19,          -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, 21,          22,          23,          24,
           25,          26,          27,          28,          29,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, 31,          32,
           33,          34,          35,          36,          37,
           38,          39,          -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           41,          42,          43,          44,          45,
           46,          47,          48,          49,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, 51,          52,          53,
           54,          55,          56,          57,          58,
           59,          -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, 61,
           62,          63,          64,          65,          66,
           67,          68,          69,          -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, 71,          72,          73,          74,
           75,          76,          77,          78,          79,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, 81,          82,
           83,          84,          85,          86,          87,
           88,          89}));
}

TEST(ReferenceTest, RandomJaxReference45) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, -1, -2, -2},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(18, 6));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({3,  4,  5,  6,  7,  8,  1,  1,  1,  1,  1,  1,  13, 14,
                        15, 16, 17, 18, 1,  1,  1,  1,  1,  1,  23, 24, 25, 26,
                        27, 28, 1,  1,  1,  1,  1,  1,  33, 34, 35, 36, 37, 38,
                        1,  1,  1,  1,  1,  1,  43, 44, 45, 46, 47, 48, 1,  1,
                        1,  1,  1,  1,  53, 54, 55, 56, 57, 58, 1,  1,  1,  1,
                        1,  1,  63, 64, 65, 66, 67, 68, 1,  1,  1,  1,  1,  1,
                        73, 74, 75, 76, 77, 78, 1,  1,  1,  1,  1,  1,  83, 84,
                        85, 86, 87, 88, 1,  1,  1,  1,  1,  1}));
}

TEST(ReferenceTest, RandomJaxReference46) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{-1, 2, 0, -1},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 17));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19,
           21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29,
           31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39,
           41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49, 49,
           51, 52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59,
           61, 62, 62, 63, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69,
           71, 72, 72, 73, 73, 74, 74, 75, 75, 76, 76, 77, 77, 78, 78, 79, 79,
           81, 82, 82, 83, 83, 84, 84, 85, 85, 86, 86, 87, 87, 88, 88, 89, 89,
           91, 92, 92, 93, 93, 94, 94, 95, 95, 96, 96, 97, 97, 98, 98, 99, 99,
           1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1}));
}

TEST(ReferenceTest, RandomJaxReference47) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, -1, 0, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(18, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 0,  0,  0,  0,  0,  0,  0,
           0,  0,  0,  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0,  0,  0,  0,
           0,  0,  0,  0,  0,  0,  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  31, 32, 33, 34, 35, 36, 37, 38,
           39, 40, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  41, 42, 43, 44, 45,
           46, 47, 48, 49, 50, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  51, 52,
           53, 54, 55, 56, 57, 58, 59, 60, 0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 0,  0,  0,  0,  0,  0,
           0,  0,  0,  0,  71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 0,  0,  0,
           0,  0,  0,  0,  0,  0,  0,  81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
           0,  0,  0,  0,  0,  0,  0,  0,  0,  0}));
}

TEST(ReferenceTest, RandomJaxReference48) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, -1, 1, 2},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(16, 6));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({11, 156,  210,  272,  342,  20, 1, 1, 1, 1, 1, 1,
                        21, 506,  600,  702,  812,  30, 1, 1, 1, 1, 1, 1,
                        31, 1056, 1190, 1332, 1482, 40, 1, 1, 1, 1, 1, 1,
                        41, 1806, 1980, 2162, 2352, 50, 1, 1, 1, 1, 1, 1,
                        51, 2756, 2970, 3192, 3422, 60, 1, 1, 1, 1, 1, 1,
                        61, 3906, 4160, 4422, 4692, 70, 1, 1, 1, 1, 1, 1,
                        71, 5256, 5550, 5852, 6162, 80, 1, 1, 1, 1, 1, 1,
                        81, 6806, 7140, 7482, 7832, 90, 1, 1, 1, 1, 1, 1}));
}

TEST(ReferenceTest, RandomJaxReference49) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{0, 1, -2, 0},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 17));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2,          2147483646, 3,          2147483646, 4,
                   2147483646, 5,          2147483646, 6,          2147483646,
                   7,          2147483646, 8,          2147483646, 9,
                   2147483646, 10,         12,         2147483646, 13,
                   2147483646, 14,         2147483646, 15,         2147483646,
                   16,         2147483646, 17,         2147483646, 18,
                   2147483646, 19,         2147483646, 20,         22,
                   2147483646, 23,         2147483646, 24,         2147483646,
                   25,         2147483646, 26,         2147483646, 27,
                   2147483646, 28,         2147483646, 29,         2147483646,
                   30,         32,         2147483646, 33,         2147483646,
                   34,         2147483646, 35,         2147483646, 36,
                   2147483646, 37,         2147483646, 38,         2147483646,
                   39,         2147483646, 40,         42,         2147483646,
                   43,         2147483646, 44,         2147483646, 45,
                   2147483646, 46,         2147483646, 47,         2147483646,
                   48,         2147483646, 49,         2147483646, 50,
                   52,         2147483646, 53,         2147483646, 54,
                   2147483646, 55,         2147483646, 56,         2147483646,
                   57,         2147483646, 58,         2147483646, 59,
                   2147483646, 60,         62,         2147483646, 63,
                   2147483646, 64,         2147483646, 65,         2147483646,
                   66,         2147483646, 67,         2147483646, 68,
                   2147483646, 69,         2147483646, 70,         72,
                   2147483646, 73,         2147483646, 74,         2147483646,
                   75,         2147483646, 76,         2147483646, 77,
                   2147483646, 78,         2147483646, 79,         2147483646,
                   80,         82,         2147483646, 83,         2147483646,
                   84,         2147483646, 85,         2147483646, 86,
                   2147483646, 87,         2147483646, 88,         2147483646,
                   89,         2147483646, 90,         92,         2147483646,
                   93,         2147483646, 94,         2147483646, 95,
                   2147483646, 96,         2147483646, 97,         2147483646,
                   98,         2147483646, 99,         2147483646, 100}));
}

TEST(ReferenceTest, RandomJaxReference50) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{-1, -1, 1, 0},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(16, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference51) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{0, 2, -2, -1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(19, 7));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2,          3,          4,          5,          6,
                   7,          8,          2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 12,
                   13,         14,         15,         16,         17,
                   18,         2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 22,         23,
                   24,         25,         26,         27,         28,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 32,         33,         34,
                   35,         36,         37,         38,         2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 42,         43,         44,         45,
                   46,         47,         48,         2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   52,         53,         54,         55,         56,
                   57,         58,         2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 62,
                   63,         64,         65,         66,         67,
                   68,         2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 72,         73,
                   74,         75,         76,         77,         78,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 82,         83,         84,
                   85,         86,         87,         88,         2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 92,         93,         94,         95,
                   96,         97,         98}));
}

TEST(ReferenceTest, RandomJaxReference52) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-2, 0, 1, 2},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(8, 11));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 0,  31,  32, 33, 34,
                   35, 36, 37, 38, 39, 40, 0,  41, 42, 43, 44, 45,  46, 47, 48,
                   49, 50, 0,  51, 52, 53, 54, 55, 56, 57, 58, 59,  60, 0,  61,
                   62, 63, 64, 65, 66, 67, 68, 69, 70, 0,  71, 72,  73, 74, 75,
                   76, 77, 78, 79, 80, 0,  81, 82, 83, 84, 85, 86,  87, 88, 89,
                   90, 0,  91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 0}));
}

TEST(ReferenceTest, RandomJaxReference53) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{2, 1, 0, 2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(11, 10));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   1,          2,          3,          4,          5,
                   6,          7,          8,          9,          10,
                   11,         12,         13,         14,         15,
                   16,         17,         18,         19,         20,
                   21,         22,         23,         24,         25,
                   26,         27,         28,         29,         30,
                   31,         32,         33,         34,         35,
                   36,         37,         38,         39,         40,
                   41,         42,         43,         44,         45,
                   46,         47,         48,         49,         50,
                   51,         52,         53,         54,         55,
                   56,         57,         58,         59,         60,
                   61,         62,         63,         64,         65,
                   66,         67,         68,         69,         70,
                   71,         72,         73,         74,         75,
                   76,         77,         78,         79,         80,
                   81,         82,         83,         84,         85,
                   86,         87,         88,         89,         90,
                   91,         92,         93,         94,         95,
                   96,         97,         98,         99,         100}));
}

TEST(ReferenceTest, RandomJaxReference54) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, 0, 0, 2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 12));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  -2147483647, -2147483647,
           21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  -2147483647, -2147483647,
           31, 32, 33, 34, 35, 36, 37, 38, 39, 40,  -2147483647, -2147483647,
           41, 42, 43, 44, 45, 46, 47, 48, 49, 50,  -2147483647, -2147483647,
           51, 52, 53, 54, 55, 56, 57, 58, 59, 60,  -2147483647, -2147483647,
           61, 62, 63, 64, 65, 66, 67, 68, 69, 70,  -2147483647, -2147483647,
           71, 72, 73, 74, 75, 76, 77, 78, 79, 80,  -2147483647, -2147483647,
           81, 82, 83, 84, 85, 86, 87, 88, 89, 90,  -2147483647, -2147483647,
           91, 92, 93, 94, 95, 96, 97, 98, 99, 100, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference55) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{2, 1, -2, 2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(20, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {3,           5,           7,           9,           -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           13,          15,          17,          19,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           23,          25,          27,          29,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           33,          35,          37,          39,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           43,          45,          47,          49,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           53,          55,          57,          59,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           63,          65,          67,          69,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           73,          75,          77,          79,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           83,          85,          87,          89,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           93,          95,          97,          99,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference56) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, 0, 0, 1},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(18, 11));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  1,  11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 1,  11, 12, 13, 14, 15,  16, 17, 18, 19, 20, 1,  21,
           22, 23, 24, 25, 26, 27, 28, 29, 30, 1,   21, 22, 23, 24, 25, 26, 27,
           28, 29, 30, 1,  31, 32, 33, 34, 35, 36,  37, 38, 39, 40, 1,  31, 32,
           33, 34, 35, 36, 37, 38, 39, 40, 1,  41,  42, 43, 44, 45, 46, 47, 48,
           49, 50, 1,  41, 42, 43, 44, 45, 46, 47,  48, 49, 50, 1,  51, 52, 53,
           54, 55, 56, 57, 58, 59, 60, 1,  51, 52,  53, 54, 55, 56, 57, 58, 59,
           60, 1,  61, 62, 63, 64, 65, 66, 67, 68,  69, 70, 1,  61, 62, 63, 64,
           65, 66, 67, 68, 69, 70, 1,  71, 72, 73,  74, 75, 76, 77, 78, 79, 80,
           1,  71, 72, 73, 74, 75, 76, 77, 78, 79,  80, 1,  81, 82, 83, 84, 85,
           86, 87, 88, 89, 90, 1,  81, 82, 83, 84,  85, 86, 87, 88, 89, 90, 1,
           91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 1}));
}

TEST(ReferenceTest, RandomJaxReference57) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{0, 0, -2, 2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 9));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({3,  4,  5,  6,  7,  8,  9,  10, 10, 13, 14,  15, 16,
                        17, 18, 19, 20, 20, 23, 24, 25, 26, 27, 28,  29, 30,
                        30, 33, 34, 35, 36, 37, 38, 39, 40, 40, 43,  44, 45,
                        46, 47, 48, 49, 50, 50, 53, 54, 55, 56, 57,  58, 59,
                        60, 60, 63, 64, 65, 66, 67, 68, 69, 70, 70,  73, 74,
                        75, 76, 77, 78, 79, 80, 80, 83, 84, 85, 86,  87, 88,
                        89, 90, 90, 93, 94, 95, 96, 97, 98, 99, 100, 100}));
}

TEST(ReferenceTest, RandomJaxReference58) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-1, 2, 1, -2},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(11, 9));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(ReferenceTest, RandomJaxReference59) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{2, -2, 2, 2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(18, 11));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 1,          1,          2,          3,
                   4,          5,          6,          7,          8,
                   9,          10,         1,          1,          2,
                   3,          4,          5,          6,          7,
                   8,          9,          10,         11,         11,
                   12,         13,         14,         15,         16,
                   17,         18,         19,         20,         11,
                   11,         12,         13,         14,         15,
                   16,         17,         18,         19,         20,
                   21,         21,         22,         23,         24,
                   25,         26,         27,         28,         29,
                   30,         21,         21,         22,         23,
                   24,         25,         26,         27,         28,
                   29,         30,         31,         31,         32,
                   33,         34,         35,         36,         37,
                   38,         39,         40,         31,         31,
                   32,         33,         34,         35,         36,
                   37,         38,         39,         40,         41,
                   41,         42,         43,         44,         45,
                   46,         47,         48,         49,         50,
                   41,         41,         42,         43,         44,
                   45,         46,         47,         48,         49,
                   50,         51,         51,         52,         53,
                   54,         55,         56,         57,         58,
                   59,         60,         51,         51,         52,
                   53,         54,         55,         56,         57,
                   58,         59,         60,         61,         61,
                   62,         63,         64,         65,         66,
                   67,         68,         69,         70,         61,
                   61,         62,         63,         64,         65,
                   66,         67,         68,         69,         70,
                   71,         71,         72,         73,         74,
                   75,         76,         77,         78,         79,
                   80,         71,         71,         72,         73,
                   74,         75,         76,         77,         78,
                   79,         80,         81,         81,         82,
                   83,         84,         85,         86,         87,
                   88,         89,         90}));
}

TEST(ReferenceTest, RandomJaxReference60) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, 2, -1, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(6, 4));

  EXPECT_THAT(res.data,
              ElementsAreArray({5,   9,   13,  17,  45,  49,  53,  57,
                                85,  89,  93,  97,  125, 129, 133, 137,
                                165, 169, 173, 177, 0,   0,   0,   0}));
}

TEST(ReferenceTest, RandomJaxReference61) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{0, -1, 2, -1},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(17, 20));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {0,  0, 1,  0, 2,  0, 3,  0, 4,  0, 5,  0, 6,  0, 7,  0, 8,  0,
           9,  0, 0,  0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0, 17, 0,
           18, 0, 19, 0, 0,  0, 11, 0, 12, 0, 13, 0, 14, 0, 15, 0, 16, 0,
           17, 0, 18, 0, 19, 0, 0,  0, 21, 0, 22, 0, 23, 0, 24, 0, 25, 0,
           26, 0, 27, 0, 28, 0, 29, 0, 0,  0, 21, 0, 22, 0, 23, 0, 24, 0,
           25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 0,  0, 31, 0, 32, 0, 33, 0,
           34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 0,  0, 31, 0, 32, 0,
           33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39, 0, 0,  0, 41, 0,
           42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47, 0, 48, 0, 49, 0, 0,  0,
           41, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47, 0, 48, 0, 49, 0,
           0,  0, 51, 0, 52, 0, 53, 0, 54, 0, 55, 0, 56, 0, 57, 0, 58, 0,
           59, 0, 0,  0, 51, 0, 52, 0, 53, 0, 54, 0, 55, 0, 56, 0, 57, 0,
           58, 0, 59, 0, 0,  0, 61, 0, 62, 0, 63, 0, 64, 0, 65, 0, 66, 0,
           67, 0, 68, 0, 69, 0, 0,  0, 61, 0, 62, 0, 63, 0, 64, 0, 65, 0,
           66, 0, 67, 0, 68, 0, 69, 0, 0,  0, 71, 0, 72, 0, 73, 0, 74, 0,
           75, 0, 76, 0, 77, 0, 78, 0, 79, 0, 0,  0, 71, 0, 72, 0, 73, 0,
           74, 0, 75, 0, 76, 0, 77, 0, 78, 0, 79, 0, 0,  0, 81, 0, 82, 0,
           83, 0, 84, 0, 85, 0, 86, 0, 87, 0, 88, 0, 89, 0, 0,  0, 81, 0,
           82, 0, 83, 0, 84, 0, 85, 0, 86, 0, 87, 0, 88, 0, 89, 0}));
}

TEST(ReferenceTest, RandomJaxReference62) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-2, -1, 2, 0},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(3, 12));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
           -2147483647, -2147483647, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
           -2147483647, -2147483647, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90}));
}

TEST(ReferenceTest, RandomJaxReference63) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-1, 0, 2, -2},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(16, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1, 1, 1,    1,    1,    1,    1,    1,    1,    1,
                        1, 1, 231,  264,  299,  336,  375,  416,  459,  504,
                        1, 1, 1,    1,    1,    1,    1,    1,    1,    1,
                        1, 1, 651,  704,  759,  816,  875,  936,  999,  1064,
                        1, 1, 1,    1,    1,    1,    1,    1,    1,    1,
                        1, 1, 1271, 1344, 1419, 1496, 1575, 1656, 1739, 1824,
                        1, 1, 1,    1,    1,    1,    1,    1,    1,    1,
                        1, 1, 2091, 2184, 2279, 2376, 2475, 2576, 2679, 2784,
                        1, 1, 1,    1,    1,    1,    1,    1,    1,    1,
                        1, 1, 3111, 3224, 3339, 3456, 3575, 3696, 3819, 3944,
                        1, 1, 1,    1,    1,    1,    1,    1,    1,    1,
                        1, 1, 4331, 4464, 4599, 4736, 4875, 5016, 5159, 5304,
                        1, 1, 1,    1,    1,    1,    1,    1,    1,    1,
                        1, 1, 5751, 5904, 6059, 6216, 6375, 6536, 6699, 6864,
                        1, 1, 1,    1,    1,    1,    1,    1,    1,    1,
                        1, 1, 7371, 7544, 7719, 7896, 8075, 8256, 8439, 8624}));
}

TEST(ReferenceTest, RandomJaxReference64) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, 2, 0, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(11, 3));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {3,  5,  7,  13,          15,          17,         23, 25, 27,
                   33, 35, 37, 43,          45,          47,         53, 55, 57,
                   63, 65, 67, 73,          75,          77,         83, 85, 87,
                   93, 95, 97, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference65) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-1, 0, 2, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(4, 11));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({0, 32,  34,  36,  38,  40,  42,  44,  46,  48,  50,
                        0, 72,  74,  76,  78,  80,  82,  84,  86,  88,  90,
                        0, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130,
                        0, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170}));
}

TEST(ReferenceTest, RandomJaxReference66) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{0, 0, -1, -1},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(5, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({14,  16,  18,  20,  22,  24,  26,  28,  54,  56,
                        58,  60,  62,  64,  66,  68,  94,  96,  98,  100,
                        102, 104, 106, 108, 134, 136, 138, 140, 142, 144,
                        146, 148, 174, 176, 178, 180, 182, 184, 186, 188}));
}

TEST(ReferenceTest, RandomJaxReference67) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{1, 0, 2, 2},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(6, 13));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {0, 0,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0, 11, 23,  25,  27,  29,  31,  33,  35,  37,  39,  20,  0,
           0, 31, 63,  65,  67,  69,  71,  73,  75,  77,  79,  40,  0,
           0, 51, 103, 105, 107, 109, 111, 113, 115, 117, 119, 60,  0,
           0, 71, 143, 145, 147, 149, 151, 153, 155, 157, 159, 80,  0,
           0, 91, 183, 185, 187, 189, 191, 193, 195, 197, 199, 100, 0}));
}

TEST(ReferenceTest, RandomJaxReference68) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, 2, 1, -2},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(13, 9));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(ReferenceTest, RandomJaxReference69) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, 1, -2, -1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(8, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({25, 26, 27, 28, 29, 35, 36, 37, 38, 39, 45, 46, 47, 48,
                        49, 55, 56, 57, 58, 59, 65, 66, 67, 68, 69, 75, 76, 77,
                        78, 79, 85, 86, 87, 88, 89, 95, 96, 97, 98, 99}));
}

TEST(ReferenceTest, RandomJaxReference70) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-1, -2, 0, 2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(4, 6));

  EXPECT_THAT(res.data, ElementsAreArray({11, 13, 15, 17, 19, 2147483646,
                                          31, 33, 35, 37, 39, 2147483646,
                                          51, 53, 55, 57, 59, 2147483646,
                                          71, 73, 75, 77, 79, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference71) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, 2, -2, 2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(21, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {3,           4,           5,           6,           7,
           8,           9,           10,          -2147483647, -2147483647,
           3,           4,           5,           6,           7,
           8,           9,           10,          -2147483647, -2147483647,
           13,          14,          15,          16,          17,
           18,          19,          20,          -2147483647, -2147483647,
           13,          14,          15,          16,          17,
           18,          19,          20,          -2147483647, -2147483647,
           23,          24,          25,          26,          27,
           28,          29,          30,          -2147483647, -2147483647,
           23,          24,          25,          26,          27,
           28,          29,          30,          -2147483647, -2147483647,
           33,          34,          35,          36,          37,
           38,          39,          40,          -2147483647, -2147483647,
           33,          34,          35,          36,          37,
           38,          39,          40,          -2147483647, -2147483647,
           43,          44,          45,          46,          47,
           48,          49,          50,          -2147483647, -2147483647,
           43,          44,          45,          46,          47,
           48,          49,          50,          -2147483647, -2147483647,
           53,          54,          55,          56,          57,
           58,          59,          60,          -2147483647, -2147483647,
           53,          54,          55,          56,          57,
           58,          59,          60,          -2147483647, -2147483647,
           63,          64,          65,          66,          67,
           68,          69,          70,          -2147483647, -2147483647,
           63,          64,          65,          66,          67,
           68,          69,          70,          -2147483647, -2147483647,
           73,          74,          75,          76,          77,
           78,          79,          80,          -2147483647, -2147483647,
           73,          74,          75,          76,          77,
           78,          79,          80,          -2147483647, -2147483647,
           83,          84,          85,          86,          87,
           88,          89,          90,          -2147483647, -2147483647,
           83,          84,          85,          86,          87,
           88,          89,          90,          -2147483647, -2147483647,
           93,          94,          95,          96,          97,
           98,          99,          100,         -2147483647, -2147483647,
           93,          94,          95,          96,          97,
           98,          99,          100,         -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference72) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, -1, 2, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(5, 5));

  EXPECT_THAT(res.data,
              ElementsAreArray({1,   4,   8,  12,  16,  21,  44, 48,  52,
                                56,  41,  84, 88,  92,  96,  61, 124, 128,
                                132, 136, 81, 164, 168, 172, 176}));
}

TEST(ReferenceTest, RandomJaxReference73) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, 0, 0, 0},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({3,  4,  5,  6,  7,  8,  9,  10, 13, 14, 15, 16, 17, 18,
                        19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36,
                        37, 38, 39, 40, 43, 44, 45, 46, 47, 48, 49, 50, 53, 54,
                        55, 56, 57, 58, 59, 60, 63, 64, 65, 66, 67, 68, 69, 70,
                        73, 74, 75, 76, 77, 78, 79, 80, 83, 84, 85, 86, 87, 88,
                        89, 90, 93, 94, 95, 96, 97, 98, 99, 100}));
}

TEST(ReferenceTest, RandomJaxReference74) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, -2, -2, -1},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(7, 5));

  EXPECT_THAT(res.data,
              ElementsAreArray({36,  40,  44,  48,  52,  76,  80,  84,  88,
                                92,  116, 120, 124, 128, 132, 156, 160, 164,
                                168, 172, 196, 200, 204, 208, 212, 236, 240,
                                244, 248, 252, 276, 280, 284, 288, 292}));
}

TEST(ReferenceTest, RandomJaxReference75) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, 1, -2, 1},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(9, 5));

  EXPECT_THAT(res.data,
              ElementsAreArray({16,  20,  24,  28,  0,   36,  40,  44,  48,
                                0,   56,  60,  64,  68,  0,   76,  80,  84,
                                88,  0,   96,  100, 104, 108, 0,   116, 120,
                                124, 128, 0,   136, 140, 144, 148, 0,   156,
                                160, 164, 168, 0,   176, 180, 184, 188, 0}));
}

TEST(ReferenceTest, RandomJaxReference76) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, -1, -1, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(6, 18));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,
                        0, 0,  0, 2,  0, 3,  0, 4,  0, 5,  0, 6,  0, 7,  0, 8,
                        0, 9,  0, 10, 0, 22, 0, 23, 0, 24, 0, 25, 0, 26, 0, 27,
                        0, 28, 0, 29, 0, 30, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46,
                        0, 47, 0, 48, 0, 49, 0, 50, 0, 62, 0, 63, 0, 64, 0, 65,
                        0, 66, 0, 67, 0, 68, 0, 69, 0, 70, 0, 82, 0, 83, 0, 84,
                        0, 85, 0, 86, 0, 87, 0, 88, 0, 89, 0, 90}));
}

TEST(ReferenceTest, RandomJaxReference77) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-1, 2, -1, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference78) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, 1, 2, -1},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(18, 6));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({0,  11, 13, 15, 17, 19, 0,  0,  0,  0,  0,  0,  0,  21,
                        23, 25, 27, 29, 0,  0,  0,  0,  0,  0,  0,  31, 33, 35,
                        37, 39, 0,  0,  0,  0,  0,  0,  0,  41, 43, 45, 47, 49,
                        0,  0,  0,  0,  0,  0,  0,  51, 53, 55, 57, 59, 0,  0,
                        0,  0,  0,  0,  0,  61, 63, 65, 67, 69, 0,  0,  0,  0,
                        0,  0,  0,  71, 73, 75, 77, 79, 0,  0,  0,  0,  0,  0,
                        0,  81, 83, 85, 87, 89, 0,  0,  0,  0,  0,  0,  0,  91,
                        93, 95, 97, 99, 0,  0,  0,  0,  0,  0}));
}

TEST(ReferenceTest, RandomJaxReference79) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-1, -1, -2, 1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(8, 9));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27,
                   28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44,
                   45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                   62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77,
                   78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90}));
}

TEST(ReferenceTest, RandomJaxReference80) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, 2, 1, -1},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1, 24,   56,   96,   144,  1, 264,  336,  416,  504,
                        1, 704,  816,  936,  1064, 1, 1344, 1496, 1656, 1824,
                        1, 2184, 2376, 2576, 2784, 1, 3224, 3456, 3696, 3944,
                        1, 4464, 4736, 5016, 5304, 1, 5904, 6216, 6536, 6864,
                        1, 7544, 7896, 8256, 8624, 1, 92,   94,   96,   98}));
}

TEST(ReferenceTest, RandomJaxReference81) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, -1, 0, 2},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(5, 6));

  EXPECT_THAT(res.data,
              ElementsAreArray({1,  3,  5,  7,  9,  1,  21, 23, 25, 27,
                                29, 1,  41, 43, 45, 47, 49, 1,  61, 63,
                                65, 67, 69, 1,  81, 83, 85, 87, 89, 1}));
}

TEST(ReferenceTest, RandomJaxReference82) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, 2, 0, 2},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(11, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {429,      2925,     8925,     20349,    171,      69069,    112125,
           172125,   252909,   551,      494109,   664125,   874125,   1129869,
           1131,     1803549,  2234925,  2738925,  3323229,  1911,     4765389,
           5640525,  6630525,  7744989,  2891,     10387629, 11936925, 13652925,
           15547149, 4071,     19918269, 22420125, 25150125, 28121709, 5451,
           34845309, 38626125, 42706125, 47100669, 7031,     56896749, 62330925,
           68144925, 74356029, 8811,     8463,     8835,     9215,     9603,
           99,       1,        1,        1,        1,        1}));
}

TEST(ReferenceTest, RandomJaxReference83) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, -1, -2, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, 2,           3,
           4,           5,           6,           7,           8,
           9,           12,          13,          14,          15,
           16,          17,          18,          19,          22,
           23,          24,          25,          26,          27,
           28,          29,          32,          33,          34,
           35,          36,          37,          38,          39,
           42,          43,          44,          45,          46,
           47,          48,          49,          52,          53,
           54,          55,          56,          57,          58,
           59,          62,          63,          64,          65,
           66,          67,          68,          69,          72,
           73,          74,          75,          76,          77,
           78,          79,          82,          83,          84,
           85,          86,          87,          88,          89}));
}

TEST(ReferenceTest, RandomJaxReference84) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{2, -2, -2, 2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(19, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2,          3,          4,          5,          6,
           7,          8,          9,          10,         2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           12,         13,         14,         15,         16,
           17,         18,         19,         20,         2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           22,         23,         24,         25,         26,
           27,         28,         29,         30,         2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           32,         33,         34,         35,         36,
           37,         38,         39,         40,         2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           42,         43,         44,         45,         46,
           47,         48,         49,         50,         2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           52,         53,         54,         55,         56,
           57,         58,         59,         60,         2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           62,         63,         64,         65,         66,
           67,         68,         69,         70,         2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           72,         73,         74,         75,         76,
           77,         78,         79,         80,         2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           82,         83,         84,         85,         86,
           87,         88,         89,         90,         2147483646}));
}

TEST(ReferenceTest, RandomJaxReference85) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, 2, -2, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(11, 2));

  EXPECT_THAT(res.data, ElementsAreArray(
                            {-2147483647, -2147483647, -2147483647, -2147483647,
                             -2147483647, -2147483647, -2147483647, -2147483647,
                             -2147483647, -2147483647, -2147483647, -2147483647,
                             -2147483647, -2147483647, -2147483647, -2147483647,
                             -2147483647, -2147483647, -2147483647, -2147483647,
                             -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference86) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, -1, 2, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(8, 5));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {-2147483647, 12, 14, 16, 18, -2147483647, 22, 24, 26, 28,
                   -2147483647, 32, 34, 36, 38, -2147483647, 42, 44, 46, 48,
                   -2147483647, 52, 54, 56, 58, -2147483647, 62, 64, 66, 68,
                   -2147483647, 72, 74, 76, 78, -2147483647, 82, 84, 86, 88}));
}

TEST(ReferenceTest, RandomJaxReference87) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-2, 0, 2, -1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(8, 10));

  EXPECT_THAT(res.data, ElementsAreArray(
                            {-2147483647, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                             -2147483647, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                             -2147483647, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                             -2147483647, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                             -2147483647, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                             -2147483647, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                             -2147483647, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                             -2147483647, 91, 92, 93, 94, 95, 96, 97, 98, 99}));
}

TEST(ReferenceTest, RandomJaxReference88) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-2, 1, 2, 0},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(4, 11));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({-2147483647, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                        -2147483647, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                        -2147483647, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                        -2147483647, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90}));
}

TEST(ReferenceTest, RandomJaxReference89) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, -2, 2, 2},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(9, 14));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(ReferenceTest, RandomJaxReference90) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-2, 0, 1, 1},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(4, 11));

  EXPECT_THAT(res.data,
              ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(ReferenceTest, RandomJaxReference91) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-2, -2, 1, 2},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(5, 6));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({704,  574464,   763776,   995904,   1276800,  1200,
                        1344, 2010624,  2477376,  3020544,  3648000,  2000,
                        2184, 5189184,  6120576,  7171584,  8352000,  3000,
                        3224, 11142144, 12773376, 14577024, 16564800, 4200,
                        4464, 21141504, 23755776, 26604864, 29702400, 5600}));
}

TEST(ReferenceTest, RandomJaxReference92) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{0, 0, 0, 2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 10));

  EXPECT_THAT(res.data, ElementsAreArray(
                            {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                             14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                             27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                             40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                             53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                             66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
                             79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90}));
}

TEST(ReferenceTest, RandomJaxReference93) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{0, -1, 0, -2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 17));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {1,          2147483646, 2,          2147483646, 3,
                   2147483646, 4,          2147483646, 5,          2147483646,
                   6,          2147483646, 7,          2147483646, 8,
                   2147483646, 9,          11,         2147483646, 12,
                   2147483646, 13,         2147483646, 14,         2147483646,
                   15,         2147483646, 16,         2147483646, 17,
                   2147483646, 18,         2147483646, 19,         21,
                   2147483646, 22,         2147483646, 23,         2147483646,
                   24,         2147483646, 25,         2147483646, 26,
                   2147483646, 27,         2147483646, 28,         2147483646,
                   29,         31,         2147483646, 32,         2147483646,
                   33,         2147483646, 34,         2147483646, 35,
                   2147483646, 36,         2147483646, 37,         2147483646,
                   38,         2147483646, 39,         41,         2147483646,
                   42,         2147483646, 43,         2147483646, 44,
                   2147483646, 45,         2147483646, 46,         2147483646,
                   47,         2147483646, 48,         2147483646, 49,
                   51,         2147483646, 52,         2147483646, 53,
                   2147483646, 54,         2147483646, 55,         2147483646,
                   56,         2147483646, 57,         2147483646, 58,
                   2147483646, 59,         61,         2147483646, 62,
                   2147483646, 63,         2147483646, 64,         2147483646,
                   65,         2147483646, 66,         2147483646, 67,
                   2147483646, 68,         2147483646, 69,         71,
                   2147483646, 72,         2147483646, 73,         2147483646,
                   74,         2147483646, 75,         2147483646, 76,
                   2147483646, 77,         2147483646, 78,         2147483646,
                   79,         81,         2147483646, 82,         2147483646,
                   83,         2147483646, 84,         2147483646, 85,
                   2147483646, 86,         2147483646, 87,         2147483646,
                   88,         2147483646, 89}));
}

TEST(ReferenceTest, RandomJaxReference94) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-2, 0, -1, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(8, 3));

  EXPECT_THAT(res.data, ElementsAreArray({23, 25, 27, 33, 35, 37, 43, 45,
                                          47, 53, 55, 57, 63, 65, 67, 73,
                                          75, 77, 83, 85, 87, 93, 95, 97}));
}

TEST(ReferenceTest, RandomJaxReference95) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{0, 0, 2, 2},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 23));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,  1,  1,  1,  2,  1,  3,   1,  4,  1,  5,  1,  6,  1,  7,  1,  8,
           1,  9,  1,  10, 1,  1,  1,   1,  11, 1,  12, 1,  13, 1,  14, 1,  15,
           1,  16, 1,  17, 1,  18, 1,   19, 1,  20, 1,  1,  1,  1,  21, 1,  22,
           1,  23, 1,  24, 1,  25, 1,   26, 1,  27, 1,  28, 1,  29, 1,  30, 1,
           1,  1,  1,  31, 1,  32, 1,   33, 1,  34, 1,  35, 1,  36, 1,  37, 1,
           38, 1,  39, 1,  40, 1,  1,   1,  1,  41, 1,  42, 1,  43, 1,  44, 1,
           45, 1,  46, 1,  47, 1,  48,  1,  49, 1,  50, 1,  1,  1,  1,  51, 1,
           52, 1,  53, 1,  54, 1,  55,  1,  56, 1,  57, 1,  58, 1,  59, 1,  60,
           1,  1,  1,  1,  61, 1,  62,  1,  63, 1,  64, 1,  65, 1,  66, 1,  67,
           1,  68, 1,  69, 1,  70, 1,   1,  1,  1,  71, 1,  72, 1,  73, 1,  74,
           1,  75, 1,  76, 1,  77, 1,   78, 1,  79, 1,  80, 1,  1,  1,  1,  81,
           1,  82, 1,  83, 1,  84, 1,   85, 1,  86, 1,  87, 1,  88, 1,  89, 1,
           90, 1,  1,  1,  1,  91, 1,   92, 1,  93, 1,  94, 1,  95, 1,  96, 1,
           97, 1,  98, 1,  99, 1,  100, 1,  1}));
}

TEST(ReferenceTest, RandomJaxReference96) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{2, -1, -1, 2},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   5,   7,   9,
           11,  13,  15,  17,  19,  10,  0,   30,  34,  38,  42,  46,  50,
           54,  58,  30,  0,   70,  74,  78,  82,  86,  90,  94,  98,  50,
           0,   110, 114, 118, 122, 126, 130, 134, 138, 70,  0,   150, 154,
           158, 162, 166, 170, 174, 178, 90,  0,   190, 194, 198, 202, 206,
           210, 214, 218, 110, 0,   230, 234, 238, 242, 246, 250, 254, 258,
           130, 0,   270, 274, 278, 282, 286, 290, 294, 298, 150, 0,   310,
           314, 318, 322, 326, 330, 334, 338, 170, 0}));
}

TEST(ReferenceTest, RandomJaxReference97) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{2, 2, -1, 1},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(12, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({5,   9,   13,  17,  10,  25,  29,  33,  37,  20,
                        50,  58,  66,  74,  40,  90,  98,  106, 114, 60,
                        130, 138, 146, 154, 80,  170, 178, 186, 194, 100,
                        210, 218, 226, 234, 120, 250, 258, 266, 274, 140,
                        290, 298, 306, 314, 160, 330, 338, 346, 354, 180,
                        165, 169, 173, 177, 90,  185, 189, 193, 197, 100}));
}

TEST(ReferenceTest, RandomJaxReference98) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{2, -2, -1, 0},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(18, 17));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2,          2,          3,
                   3,          4,          4,          5,          5,
                   6,          6,          7,          7,          8,
                   8,          9,          9,          10,         2,
                   2,          3,          3,          4,          4,
                   5,          5,          6,          6,          7,
                   7,          8,          8,          9,          9,
                   10,         12,         12,         13,         13,
                   14,         14,         15,         15,         16,
                   16,         17,         17,         18,         18,
                   19,         19,         20,         12,         12,
                   13,         13,         14,         14,         15,
                   15,         16,         16,         17,         17,
                   18,         18,         19,         19,         20,
                   22,         22,         23,         23,         24,
                   24,         25,         25,         26,         26,
                   27,         27,         28,         28,         29,
                   29,         30,         22,         22,         23,
                   23,         24,         24,         25,         25,
                   26,         26,         27,         27,         28,
                   28,         29,         29,         30,         32,
                   32,         33,         33,         34,         34,
                   35,         35,         36,         36,         37,
                   37,         38,         38,         39,         39,
                   40,         32,         32,         33,         33,
                   34,         34,         35,         35,         36,
                   36,         37,         37,         38,         38,
                   39,         39,         40,         42,         42,
                   43,         43,         44,         44,         45,
                   45,         46,         46,         47,         47,
                   48,         48,         49,         49,         50,
                   42,         42,         43,         43,         44,
                   44,         45,         45,         46,         46,
                   47,         47,         48,         48,         49,
                   49,         50,         52,         52,         53,
                   53,         54,         54,         55,         55,
                   56,         56,         57,         57,         58,
                   58,         59,         59,         60,         52,
                   52,         53,         53,         54,         54,
                   55,         55,         56,         56,         57,
                   57,         58,         58,         59,         59,
                   60,         62,         62,         63,         63,
                   64,         64,         65,         65,         66,
                   66,         67,         67,         68,         68,
                   69,         69,         70,         62,         62,
                   63,         63,         64,         64,         65,
                   65,         66,         66,         67,         67,
                   68,         68,         69,         69,         70,
                   72,         72,         73,         73,         74,
                   74,         75,         75,         76,         76,
                   77,         77,         78,         78,         79,
                   79,         80,         72,         72,         73,
                   73,         74,         74,         75,         75,
                   76,         76,         77,         77,         78,
                   78,         79,         79,         80,         82,
                   82,         83,         83,         84,         84,
                   85,         85,         86,         86,         87,
                   87,         88,         88,         89,         89,
                   90}));
}

TEST(ReferenceTest, RandomJaxReference99) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-1, -1, -2, 1},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(4, 9));

  EXPECT_THAT(res.data, ElementsAreArray({12, 13, 14, 15, 16, 17, 18, 19, 20,
                                          32, 33, 34, 35, 36, 37, 38, 39, 40,
                                          52, 53, 54, 55, 56, 57, 58, 59, 60,
                                          72, 73, 74, 75, 76, 77, 78, 79, 80}));
}

TEST(ReferenceTest, RandomJaxReference100) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{0, 1, 1, 1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 20));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,   7,  7,  8,  8,  9,
           9,  10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15,  15, 16, 16, 17, 17,
           18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23,  24, 24, 25, 25, 26,
           26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32,  32, 33, 33, 34, 34,
           35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40,  41, 41, 42, 42, 43,
           43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 48, 49,  49, 50, 50, 51, 51,
           52, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 57,  58, 58, 59, 59, 60,
           60, 61, 61, 62, 62, 63, 63, 64, 64, 65, 65, 66,  66, 67, 67, 68, 68,
           69, 69, 70, 70, 71, 71, 72, 72, 73, 73, 74, 74,  75, 75, 76, 76, 77,
           77, 78, 78, 79, 79, 80, 80, 81, 81, 82, 82, 83,  83, 84, 84, 85, 85,
           86, 86, 87, 87, 88, 88, 89, 89, 90, 90, 91, 91,  92, 92, 93, 93, 94,
           94, 95, 95, 96, 96, 97, 97, 98, 98, 99, 99, 100, 100}));
}

TEST(ReferenceTest, RandomJaxReference101) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, 2, 2, 0},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(17, 12));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1, 1, 231,  264,  299,  336,  375,  416,  459,  504,  551,  600,
           1, 1, 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1, 1, 651,  704,  759,  816,  875,  936,  999,  1064, 1131, 1200,
           1, 1, 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1, 1, 1271, 1344, 1419, 1496, 1575, 1656, 1739, 1824, 1911, 2000,
           1, 1, 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1, 1, 2091, 2184, 2279, 2376, 2475, 2576, 2679, 2784, 2891, 3000,
           1, 1, 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1, 1, 3111, 3224, 3339, 3456, 3575, 3696, 3819, 3944, 4071, 4200,
           1, 1, 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1, 1, 4331, 4464, 4599, 4736, 4875, 5016, 5159, 5304, 5451, 5600,
           1, 1, 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1, 1, 5751, 5904, 6059, 6216, 6375, 6536, 6699, 6864, 7031, 7200,
           1, 1, 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1, 1, 7371, 7544, 7719, 7896, 8075, 8256, 8439, 8624, 8811, 9000,
           1, 1, 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1, 1, 91,   92,   93,   94,   95,   96,   97,   98,   99,   100}));
}

TEST(ReferenceTest, RandomJaxReference102) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{1, 1, -2, 1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(11, 16));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference103) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, 1, 1, -1},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(11, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {2,    3,    8,    15,   24,   35,   48,   63,   12,   143,  168,
           195,  224,  255,  288,  323,  22,   483,  528,  575,  624,  675,
           728,  783,  32,   1023, 1088, 1155, 1224, 1295, 1368, 1443, 42,
           1763, 1848, 1935, 2024, 2115, 2208, 2303, 52,   2703, 2808, 2915,
           3024, 3135, 3248, 3363, 62,   3843, 3968, 4095, 4224, 4355, 4488,
           4623, 72,   5183, 5328, 5475, 5624, 5775, 5928, 6083, 82,   6723,
           6888, 7055, 7224, 7395, 7568, 7743, 92,   8463, 8648, 8835, 9024,
           9215, 9408, 9603, 1,    1,    1,    1,    1,    1,    1,    1}));
}

TEST(ReferenceTest, RandomJaxReference104) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{2, -1, 1, -1},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(18, 9));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,   3,   5,   7,   9,   11,  13,  15,  17,  0,   0,   0,   0,   0,
           0,   0,   0,   0,   12,  26,  30,  34,  38,  42,  46,  50,  54,  0,
           0,   0,   0,   0,   0,   0,   0,   0,   32,  66,  70,  74,  78,  82,
           86,  90,  94,  0,   0,   0,   0,   0,   0,   0,   0,   0,   52,  106,
           110, 114, 118, 122, 126, 130, 134, 0,   0,   0,   0,   0,   0,   0,
           0,   0,   72,  146, 150, 154, 158, 162, 166, 170, 174, 0,   0,   0,
           0,   0,   0,   0,   0,   0,   92,  186, 190, 194, 198, 202, 206, 210,
           214, 0,   0,   0,   0,   0,   0,   0,   0,   0,   112, 226, 230, 234,
           238, 242, 246, 250, 254, 0,   0,   0,   0,   0,   0,   0,   0,   0,
           132, 266, 270, 274, 278, 282, 286, 290, 294, 0,   0,   0,   0,   0,
           0,   0,   0,   0,   152, 306, 310, 314, 318, 322, 326, 330, 334, 0,
           0,   0,   0,   0,   0,   0,   0,   0}));
}

TEST(ReferenceTest, RandomJaxReference105) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-1, 2, 1, -1},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(18, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           231,  264,  299,  336,  375,  416,  459,  504,  551,  1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,    1,    651,  704,
           759,  816,  875,  936,  999,  1064, 1131, 1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1271, 1344, 1419, 1496,
           1575, 1656, 1739, 1824, 1911, 1,    1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    2091, 2184, 2279, 2376, 2475, 2576,
           2679, 2784, 2891, 1,    1,    1,    1,    1,    1,    1,    1,
           1,    1,    1,    3111, 3224, 3339, 3456, 3575, 3696, 3819, 3944,
           4071, 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
           1,    4331, 4464, 4599, 4736, 4875, 5016, 5159, 5304, 5451, 1,
           1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    5751,
           5904, 6059, 6216, 6375, 6536, 6699, 6864, 7031, 1,    1,    1,
           1,    1,    1,    1,    1,    1,    1,    1,    7371, 7544, 7719,
           7896, 8075, 8256, 8439, 8624, 8811, 1,    1,    1,    1,    1,
           1,    1,    1,    1,    1,    1,    91,   92,   93,   94,   95,
           96,   97,   98,   99}));
}

TEST(ReferenceTest, RandomJaxReference106) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, 2, -2, 2},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(7, 18));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
           1,  1,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,
           9,  10, 10, 1,  22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28,
           28, 29, 29, 30, 30, 1,  42, 43, 43, 44, 44, 45, 45, 46, 46, 47,
           47, 48, 48, 49, 49, 50, 50, 1,  62, 63, 63, 64, 64, 65, 65, 66,
           66, 67, 67, 68, 68, 69, 69, 70, 70, 1,  82, 83, 83, 84, 84, 85,
           85, 86, 86, 87, 87, 88, 88, 89, 89, 90, 90, 1,  1,  1,  1,  1,
           1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1}));
}

TEST(ReferenceTest, RandomJaxReference107) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{0, 1, 2, 0},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 11));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({2147483646, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                        2147483646, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        2147483646, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                        2147483646, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                        2147483646, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                        2147483646, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                        2147483646, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                        2147483646, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                        2147483646, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                        2147483646, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100}));
}

TEST(ReferenceTest, RandomJaxReference108) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, -1, 2, -1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(11, 20));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, 1,           -2147483647, 2,
           -2147483647, 3,           -2147483647, 4,           -2147483647,
           5,           -2147483647, 6,           -2147483647, 7,
           -2147483647, 8,           -2147483647, 9,           -2147483647,
           -2147483647, -2147483647, 11,          -2147483647, 12,
           -2147483647, 13,          -2147483647, 14,          -2147483647,
           15,          -2147483647, 16,          -2147483647, 17,
           -2147483647, 18,          -2147483647, 19,          -2147483647,
           -2147483647, -2147483647, 21,          -2147483647, 22,
           -2147483647, 23,          -2147483647, 24,          -2147483647,
           25,          -2147483647, 26,          -2147483647, 27,
           -2147483647, 28,          -2147483647, 29,          -2147483647,
           -2147483647, -2147483647, 31,          -2147483647, 32,
           -2147483647, 33,          -2147483647, 34,          -2147483647,
           35,          -2147483647, 36,          -2147483647, 37,
           -2147483647, 38,          -2147483647, 39,          -2147483647,
           -2147483647, -2147483647, 41,          -2147483647, 42,
           -2147483647, 43,          -2147483647, 44,          -2147483647,
           45,          -2147483647, 46,          -2147483647, 47,
           -2147483647, 48,          -2147483647, 49,          -2147483647,
           -2147483647, -2147483647, 51,          -2147483647, 52,
           -2147483647, 53,          -2147483647, 54,          -2147483647,
           55,          -2147483647, 56,          -2147483647, 57,
           -2147483647, 58,          -2147483647, 59,          -2147483647,
           -2147483647, -2147483647, 61,          -2147483647, 62,
           -2147483647, 63,          -2147483647, 64,          -2147483647,
           65,          -2147483647, 66,          -2147483647, 67,
           -2147483647, 68,          -2147483647, 69,          -2147483647,
           -2147483647, -2147483647, 71,          -2147483647, 72,
           -2147483647, 73,          -2147483647, 74,          -2147483647,
           75,          -2147483647, 76,          -2147483647, 77,
           -2147483647, 78,          -2147483647, 79,          -2147483647,
           -2147483647, -2147483647, 81,          -2147483647, 82,
           -2147483647, 83,          -2147483647, 84,          -2147483647,
           85,          -2147483647, 86,          -2147483647, 87,
           -2147483647, 88,          -2147483647, 89,          -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference109) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, -2, 0, 0},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(8, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27,
                        29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55,
                        57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79}));
}

TEST(ReferenceTest, RandomJaxReference110) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{-1, -1, 2, 0},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(17, 20));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
           1,  1,  1,  1,  11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17,
           17, 18, 18, 19, 19, 20, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
           1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  21, 21, 22, 22, 23, 23, 24,
           24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 1,  1,  1,  1,  1,
           1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  31,
           31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39,
           40, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
           1,  1,  1,  1,  1,  41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46,
           47, 47, 48, 48, 49, 49, 50, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
           1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  51, 51, 52, 52, 53, 53,
           54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 1,  1,  1,  1,
           1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
           61, 61, 62, 62, 63, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69,
           69, 70, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
           1,  1,  1,  1,  1,  1,  71, 71, 72, 72, 73, 73, 74, 74, 75, 75, 76,
           76, 77, 77, 78, 78, 79, 79, 80, 1,  1,  1,  1,  1,  1,  1,  1,  1,
           1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  81, 81, 82, 82, 83,
           83, 84, 84, 85, 85, 86, 86, 87, 87, 88, 88, 89, 89, 90, 1,  1,  1,
           1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1}));
}

TEST(ReferenceTest, RandomJaxReference111) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-1, 0, 2, -1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(8, 11));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {2147483646, 2147483646, 11, 12, 13, 14, 15, 16, 17, 18, 19,
           2147483646, 2147483646, 21, 22, 23, 24, 25, 26, 27, 28, 29,
           2147483646, 2147483646, 31, 32, 33, 34, 35, 36, 37, 38, 39,
           2147483646, 2147483646, 41, 42, 43, 44, 45, 46, 47, 48, 49,
           2147483646, 2147483646, 51, 52, 53, 54, 55, 56, 57, 58, 59,
           2147483646, 2147483646, 61, 62, 63, 64, 65, 66, 67, 68, 69,
           2147483646, 2147483646, 71, 72, 73, 74, 75, 76, 77, 78, 79,
           2147483646, 2147483646, 81, 82, 83, 84, 85, 86, 87, 88, 89}));
}

TEST(ReferenceTest, RandomJaxReference112) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, 1, 1, 2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(20, 13));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {2147483646, 1,          2,          3,          4,
           5,          6,          7,          8,          9,
           10,         2147483646, 2147483646, 2147483646, 1,
           2,          3,          4,          5,          6,
           7,          8,          9,          10,         2147483646,
           2147483646, 2147483646, 11,         12,         13,
           14,         15,         16,         17,         18,
           19,         20,         2147483646, 2147483646, 2147483646,
           11,         12,         13,         14,         15,
           16,         17,         18,         19,         20,
           2147483646, 2147483646, 2147483646, 21,         22,
           23,         24,         25,         26,         27,
           28,         29,         30,         2147483646, 2147483646,
           2147483646, 21,         22,         23,         24,
           25,         26,         27,         28,         29,
           30,         2147483646, 2147483646, 2147483646, 31,
           32,         33,         34,         35,         36,
           37,         38,         39,         40,         2147483646,
           2147483646, 2147483646, 31,         32,         33,
           34,         35,         36,         37,         38,
           39,         40,         2147483646, 2147483646, 2147483646,
           41,         42,         43,         44,         45,
           46,         47,         48,         49,         50,
           2147483646, 2147483646, 2147483646, 41,         42,
           43,         44,         45,         46,         47,
           48,         49,         50,         2147483646, 2147483646,
           2147483646, 51,         52,         53,         54,
           55,         56,         57,         58,         59,
           60,         2147483646, 2147483646, 2147483646, 51,
           52,         53,         54,         55,         56,
           57,         58,         59,         60,         2147483646,
           2147483646, 2147483646, 61,         62,         63,
           64,         65,         66,         67,         68,
           69,         70,         2147483646, 2147483646, 2147483646,
           61,         62,         63,         64,         65,
           66,         67,         68,         69,         70,
           2147483646, 2147483646, 2147483646, 71,         72,
           73,         74,         75,         76,         77,
           78,         79,         80,         2147483646, 2147483646,
           2147483646, 71,         72,         73,         74,
           75,         76,         77,         78,         79,
           80,         2147483646, 2147483646, 2147483646, 81,
           82,         83,         84,         85,         86,
           87,         88,         89,         90,         2147483646,
           2147483646, 2147483646, 81,         82,         83,
           84,         85,         86,         87,         88,
           89,         90,         2147483646, 2147483646, 2147483646,
           91,         92,         93,         94,         95,
           96,         97,         98,         99,         100,
           2147483646, 2147483646, 2147483646, 91,         92,
           93,         94,         95,         96,         97,
           98,         99,         100,        2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference113) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{2, -2, 1, 0},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(5, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           1,           2,           3,           4,           5,
           6,           7,           8,           9,           10,
           21,          22,          23,          24,          25,
           26,          27,          28,          29,          30,
           41,          42,          43,          44,          45,
           46,          47,          48,          49,          50,
           61,          62,          63,          64,          65,
           66,          67,          68,          69,          70}));
}

TEST(ReferenceTest, RandomJaxReference114) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, -1, 1, 1},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(8, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {12,  24,  26,  28,  30,  32,  34,  36,  38,  19,  22,  44,  46,  48,
           50,  52,  54,  56,  58,  29,  32,  64,  66,  68,  70,  72,  74,  76,
           78,  39,  42,  84,  86,  88,  90,  92,  94,  96,  98,  49,  52,  104,
           106, 108, 110, 112, 114, 116, 118, 59,  62,  124, 126, 128, 130, 132,
           134, 136, 138, 69,  72,  144, 146, 148, 150, 152, 154, 156, 158, 79,
           82,  164, 166, 168, 170, 172, 174, 176, 178, 89}));
}

TEST(ReferenceTest, RandomJaxReference115) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, -2, -2, 1},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(7, 4));

  EXPECT_THAT(res.data, ElementsAreArray({74,  82,  90,  98,  114, 122, 130,
                                          138, 154, 162, 170, 178, 194, 202,
                                          210, 218, 234, 242, 250, 258, 274,
                                          282, 290, 298, 314, 322, 330, 338}));
}

TEST(ReferenceTest, RandomJaxReference116) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{0, -2, 1, 1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(16, 21));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, 1,           -2147483647, 2,           -2147483647,
           3,           -2147483647, 4,           -2147483647, 5,
           -2147483647, 6,           -2147483647, 7,           -2147483647,
           8,           -2147483647, 9,           -2147483647, 10,
           -2147483647, -2147483647, 11,          -2147483647, 12,
           -2147483647, 13,          -2147483647, 14,          -2147483647,
           15,          -2147483647, 16,          -2147483647, 17,
           -2147483647, 18,          -2147483647, 19,          -2147483647,
           20,          -2147483647, -2147483647, 11,          -2147483647,
           12,          -2147483647, 13,          -2147483647, 14,
           -2147483647, 15,          -2147483647, 16,          -2147483647,
           17,          -2147483647, 18,          -2147483647, 19,
           -2147483647, 20,          -2147483647, -2147483647, 21,
           -2147483647, 22,          -2147483647, 23,          -2147483647,
           24,          -2147483647, 25,          -2147483647, 26,
           -2147483647, 27,          -2147483647, 28,          -2147483647,
           29,          -2147483647, 30,          -2147483647, -2147483647,
           21,          -2147483647, 22,          -2147483647, 23,
           -2147483647, 24,          -2147483647, 25,          -2147483647,
           26,          -2147483647, 27,          -2147483647, 28,
           -2147483647, 29,          -2147483647, 30,          -2147483647,
           -2147483647, 31,          -2147483647, 32,          -2147483647,
           33,          -2147483647, 34,          -2147483647, 35,
           -2147483647, 36,          -2147483647, 37,          -2147483647,
           38,          -2147483647, 39,          -2147483647, 40,
           -2147483647, -2147483647, 31,          -2147483647, 32,
           -2147483647, 33,          -2147483647, 34,          -2147483647,
           35,          -2147483647, 36,          -2147483647, 37,
           -2147483647, 38,          -2147483647, 39,          -2147483647,
           40,          -2147483647, -2147483647, 41,          -2147483647,
           42,          -2147483647, 43,          -2147483647, 44,
           -2147483647, 45,          -2147483647, 46,          -2147483647,
           47,          -2147483647, 48,          -2147483647, 49,
           -2147483647, 50,          -2147483647, -2147483647, 41,
           -2147483647, 42,          -2147483647, 43,          -2147483647,
           44,          -2147483647, 45,          -2147483647, 46,
           -2147483647, 47,          -2147483647, 48,          -2147483647,
           49,          -2147483647, 50,          -2147483647, -2147483647,
           51,          -2147483647, 52,          -2147483647, 53,
           -2147483647, 54,          -2147483647, 55,          -2147483647,
           56,          -2147483647, 57,          -2147483647, 58,
           -2147483647, 59,          -2147483647, 60,          -2147483647,
           -2147483647, 51,          -2147483647, 52,          -2147483647,
           53,          -2147483647, 54,          -2147483647, 55,
           -2147483647, 56,          -2147483647, 57,          -2147483647,
           58,          -2147483647, 59,          -2147483647, 60,
           -2147483647, -2147483647, 61,          -2147483647, 62,
           -2147483647, 63,          -2147483647, 64,          -2147483647,
           65,          -2147483647, 66,          -2147483647, 67,
           -2147483647, 68,          -2147483647, 69,          -2147483647,
           70,          -2147483647, -2147483647, 61,          -2147483647,
           62,          -2147483647, 63,          -2147483647, 64,
           -2147483647, 65,          -2147483647, 66,          -2147483647,
           67,          -2147483647, 68,          -2147483647, 69,
           -2147483647, 70,          -2147483647, -2147483647, 71,
           -2147483647, 72,          -2147483647, 73,          -2147483647,
           74,          -2147483647, 75,          -2147483647, 76,
           -2147483647, 77,          -2147483647, 78,          -2147483647,
           79,          -2147483647, 80,          -2147483647, -2147483647,
           71,          -2147483647, 72,          -2147483647, 73,
           -2147483647, 74,          -2147483647, 75,          -2147483647,
           76,          -2147483647, 77,          -2147483647, 78,
           -2147483647, 79,          -2147483647, 80,          -2147483647,
           -2147483647, 81,          -2147483647, 82,          -2147483647,
           83,          -2147483647, 84,          -2147483647, 85,
           -2147483647, 86,          -2147483647, 87,          -2147483647,
           88,          -2147483647, 89,          -2147483647, 90,
           -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference117) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, -2, -1, 0},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(8, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {156,  182,  210,  240,  272,  306,  342,  380,  506,  552,  600,
           650,  702,  756,  812,  870,  1056, 1122, 1190, 1260, 1332, 1406,
           1482, 1560, 1806, 1892, 1980, 2070, 2162, 2256, 2352, 2450, 2756,
           2862, 2970, 3080, 3192, 3306, 3422, 3540, 3906, 4032, 4160, 4290,
           4422, 4556, 4692, 4830, 5256, 5402, 5550, 5700, 5852, 6006, 6162,
           6320, 6806, 6972, 7140, 7310, 7482, 7656, 7832, 8010}));
}

TEST(ReferenceTest, RandomJaxReference118) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{-2, -1, -2, 1},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(8, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({25,  27,  29,  31,  33,  35,  37,  39,  45,  47,  49,
                        51,  53,  55,  57,  59,  65,  67,  69,  71,  73,  75,
                        77,  79,  85,  87,  89,  91,  93,  95,  97,  99,  105,
                        107, 109, 111, 113, 115, 117, 119, 125, 127, 129, 131,
                        133, 135, 137, 139, 145, 147, 149, 151, 153, 155, 157,
                        159, 165, 167, 169, 171, 173, 175, 177, 179}));
}

TEST(ReferenceTest, RandomJaxReference119) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-2, 0, 1, 2},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(6, 22));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1, 861,  1, 924,  1, 989,  1, 1056, 1, 1125, 1, 1196,
                        1, 1269, 1, 1344, 1, 1421, 1, 1500, 1, 1,    1, 1581,
                        1, 1664, 1, 1749, 1, 1836, 1, 1925, 1, 2016, 1, 2109,
                        1, 2204, 1, 2301, 1, 2400, 1, 1,    1, 2501, 1, 2604,
                        1, 2709, 1, 2816, 1, 2925, 1, 3036, 1, 3149, 1, 3264,
                        1, 3381, 1, 3500, 1, 1,    1, 3621, 1, 3744, 1, 3869,
                        1, 3996, 1, 4125, 1, 4256, 1, 4389, 1, 4524, 1, 4661,
                        1, 4800, 1, 1,    1, 4941, 1, 5084, 1, 5229, 1, 5376,
                        1, 5525, 1, 5676, 1, 5829, 1, 5984, 1, 6141, 1, 6300,
                        1, 1,    1, 6461, 1, 6624, 1, 6789, 1, 6956, 1, 7125,
                        1, 7296, 1, 7469, 1, 7644, 1, 7821, 1, 8000, 1, 1}));
}

TEST(ReferenceTest, RandomJaxReference120) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{-2, 1, 2, 0},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 21));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2147483646, 2147483646, 11,         2147483646, 12,
                   2147483646, 13,         2147483646, 14,         2147483646,
                   15,         2147483646, 16,         2147483646, 17,
                   2147483646, 18,         2147483646, 19,         2147483646,
                   20,         2147483646, 2147483646, 21,         2147483646,
                   22,         2147483646, 23,         2147483646, 24,
                   2147483646, 25,         2147483646, 26,         2147483646,
                   27,         2147483646, 28,         2147483646, 29,
                   2147483646, 30,         2147483646, 2147483646, 31,
                   2147483646, 32,         2147483646, 33,         2147483646,
                   34,         2147483646, 35,         2147483646, 36,
                   2147483646, 37,         2147483646, 38,         2147483646,
                   39,         2147483646, 40,         2147483646, 2147483646,
                   41,         2147483646, 42,         2147483646, 43,
                   2147483646, 44,         2147483646, 45,         2147483646,
                   46,         2147483646, 47,         2147483646, 48,
                   2147483646, 49,         2147483646, 50,         2147483646,
                   2147483646, 51,         2147483646, 52,         2147483646,
                   53,         2147483646, 54,         2147483646, 55,
                   2147483646, 56,         2147483646, 57,         2147483646,
                   58,         2147483646, 59,         2147483646, 60,
                   2147483646, 2147483646, 61,         2147483646, 62,
                   2147483646, 63,         2147483646, 64,         2147483646,
                   65,         2147483646, 66,         2147483646, 67,
                   2147483646, 68,         2147483646, 69,         2147483646,
                   70,         2147483646, 2147483646, 71,         2147483646,
                   72,         2147483646, 73,         2147483646, 74,
                   2147483646, 75,         2147483646, 76,         2147483646,
                   77,         2147483646, 78,         2147483646, 79,
                   2147483646, 80,         2147483646, 2147483646, 81,
                   2147483646, 82,         2147483646, 83,         2147483646,
                   84,         2147483646, 85,         2147483646, 86,
                   2147483646, 87,         2147483646, 88,         2147483646,
                   89,         2147483646, 90,         2147483646, 2147483646,
                   91,         2147483646, 92,         2147483646, 93,
                   2147483646, 94,         2147483646, 95,         2147483646,
                   96,         2147483646, 97,         2147483646, 98,
                   2147483646, 99,         2147483646, 100}));
}

TEST(ReferenceTest, RandomJaxReference121) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{0, 2, -1, 1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(20, 9));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference122) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{0, 2, -1, 1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(5, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference123) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-2, -2, 0, 2},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(6, 6));

  EXPECT_THAT(res.data,
              ElementsAreArray({43,  47,  51,  55,  59,  0,   63,  67,  71,
                                75,  79,  0,   83,  87,  91,  95,  99,  0,
                                103, 107, 111, 115, 119, 0,   123, 127, 131,
                                135, 139, 0,   143, 147, 151, 155, 159, 0}));
}

TEST(ReferenceTest, RandomJaxReference124) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, 2, -2, 0},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 4));

  EXPECT_THAT(res.data,
              ElementsAreArray({69,   125,  189,  261,  429,  525,  629,  741,
                                989,  1125, 1269, 1421, 1749, 1925, 2109, 2301,
                                2709, 2925, 3149, 3381, 3869, 4125, 4389, 4661,
                                5229, 5525, 5829, 6141, 6789, 7125, 7469, 7821,
                                83,   85,   87,   89,   93,   95,   97,   99}));
}

TEST(ReferenceTest, RandomJaxReference125) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{-1, -1, 2, 1},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(9, 21));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(ReferenceTest, RandomJaxReference126) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, 1, 0, 0},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(20, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,           3,           5,           7,           9,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           11,          13,          15,          17,          19,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           21,          23,          25,          27,          29,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           31,          33,          35,          37,          39,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           41,          43,          45,          47,          49,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           51,          53,          55,          57,          59,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           61,          63,          65,          67,          69,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           71,          73,          75,          77,          79,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           81,          83,          85,          87,          89,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           91,          93,          95,          97,          99,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference127) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, -2, 0, -2},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(16, 4));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {0, 0, 0, 0, 12,  16,  20,  24,  0, 0, 0, 0, 32,  36,  40,  44,
           0, 0, 0, 0, 52,  56,  60,  64,  0, 0, 0, 0, 72,  76,  80,  84,
           0, 0, 0, 0, 92,  96,  100, 104, 0, 0, 0, 0, 112, 116, 120, 124,
           0, 0, 0, 0, 132, 136, 140, 144, 0, 0, 0, 0, 152, 156, 160, 164}));
}

TEST(ReferenceTest, RandomJaxReference128) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-1, -2, 0, -2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(7, 3));

  EXPECT_THAT(res.data,
              ElementsAreArray({11, 13, 15, 21, 23, 25, 31, 33, 35, 41, 43,
                                45, 51, 53, 55, 61, 63, 65, 71, 73, 75}));
}

TEST(ReferenceTest, RandomJaxReference129) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{1, 2, -1, 2},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(12, 9));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(ReferenceTest, RandomJaxReference130) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-1, 1, 1, 1},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(19, 6));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,  1,  1,  1,  1,  1,   1,  12, 14, 16, 18, 20, 1,  1,  1,  1,  1,
           1,  1,  22, 24, 26, 28,  30, 1,  1,  1,  1,  1,  1,  1,  32, 34, 36,
           38, 40, 1,  1,  1,  1,   1,  1,  1,  42, 44, 46, 48, 50, 1,  1,  1,
           1,  1,  1,  1,  52, 54,  56, 58, 60, 1,  1,  1,  1,  1,  1,  1,  62,
           64, 66, 68, 70, 1,  1,   1,  1,  1,  1,  1,  72, 74, 76, 78, 80, 1,
           1,  1,  1,  1,  1,  1,   82, 84, 86, 88, 90, 1,  1,  1,  1,  1,  1,
           1,  92, 94, 96, 98, 100, 1,  1,  1,  1,  1,  1}));
}

TEST(ReferenceTest, RandomJaxReference131) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{1, -1, -2, -1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 16));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({2,  -2147483647, 3,  -2147483647, 4,  -2147483647,
                        5,  -2147483647, 6,  -2147483647, 7,  -2147483647,
                        8,  -2147483647, 9,  -2147483647, 12, -2147483647,
                        13, -2147483647, 14, -2147483647, 15, -2147483647,
                        16, -2147483647, 17, -2147483647, 18, -2147483647,
                        19, -2147483647, 22, -2147483647, 23, -2147483647,
                        24, -2147483647, 25, -2147483647, 26, -2147483647,
                        27, -2147483647, 28, -2147483647, 29, -2147483647,
                        32, -2147483647, 33, -2147483647, 34, -2147483647,
                        35, -2147483647, 36, -2147483647, 37, -2147483647,
                        38, -2147483647, 39, -2147483647, 42, -2147483647,
                        43, -2147483647, 44, -2147483647, 45, -2147483647,
                        46, -2147483647, 47, -2147483647, 48, -2147483647,
                        49, -2147483647, 52, -2147483647, 53, -2147483647,
                        54, -2147483647, 55, -2147483647, 56, -2147483647,
                        57, -2147483647, 58, -2147483647, 59, -2147483647,
                        62, -2147483647, 63, -2147483647, 64, -2147483647,
                        65, -2147483647, 66, -2147483647, 67, -2147483647,
                        68, -2147483647, 69, -2147483647, 72, -2147483647,
                        73, -2147483647, 74, -2147483647, 75, -2147483647,
                        76, -2147483647, 77, -2147483647, 78, -2147483647,
                        79, -2147483647, 82, -2147483647, 83, -2147483647,
                        84, -2147483647, 85, -2147483647, 86, -2147483647,
                        87, -2147483647, 88, -2147483647, 89, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference132) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, 2, 2, -1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 10));

  EXPECT_THAT(res.data, ElementsAreArray(
                            {-2147483647, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             -2147483647, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                             -2147483647, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                             -2147483647, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                             -2147483647, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                             -2147483647, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                             -2147483647, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                             -2147483647, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                             -2147483647, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                             -2147483647, 91, 92, 93, 94, 95, 96, 97, 98, 99}));
}

TEST(ReferenceTest, RandomJaxReference133) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, 2, 1, -1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 4));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({2,  2,  4,  6,  12, 12, 14, 16, 22, 22, 24, 26, 32, 32,
                        34, 36, 42, 42, 44, 46, 52, 52, 54, 56, 62, 62, 64, 66,
                        72, 72, 74, 76, 82, 82, 84, 86, 92, 92, 94, 96}));
}

TEST(ReferenceTest, RandomJaxReference134) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-2, 2, 2, 1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(5, 22));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, 31,          -2147483647, 32,
           -2147483647, 33,          -2147483647, 34,          -2147483647,
           35,          -2147483647, 36,          -2147483647, 37,
           -2147483647, 38,          -2147483647, 39,          -2147483647,
           40,          -2147483647, -2147483647, -2147483647, 51,
           -2147483647, 52,          -2147483647, 53,          -2147483647,
           54,          -2147483647, 55,          -2147483647, 56,
           -2147483647, 57,          -2147483647, 58,          -2147483647,
           59,          -2147483647, 60,          -2147483647, -2147483647,
           -2147483647, 71,          -2147483647, 72,          -2147483647,
           73,          -2147483647, 74,          -2147483647, 75,
           -2147483647, 76,          -2147483647, 77,          -2147483647,
           78,          -2147483647, 79,          -2147483647, 80,
           -2147483647, -2147483647, -2147483647, 91,          -2147483647,
           92,          -2147483647, 93,          -2147483647, 94,
           -2147483647, 95,          -2147483647, 96,          -2147483647,
           97,          -2147483647, 98,          -2147483647, 99,
           -2147483647, 100,         -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference135) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, 0, 0, 2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(10, 12));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference136) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{0, 1, 0, 0},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(11, 9));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {1,          2,          3,          4,          5,
                   6,          7,          8,          9,          11,
                   12,         13,         14,         15,         16,
                   17,         18,         19,         21,         22,
                   23,         24,         25,         26,         27,
                   28,         29,         31,         32,         33,
                   34,         35,         36,         37,         38,
                   39,         41,         42,         43,         44,
                   45,         46,         47,         48,         49,
                   51,         52,         53,         54,         55,
                   56,         57,         58,         59,         61,
                   62,         63,         64,         65,         66,
                   67,         68,         69,         71,         72,
                   73,         74,         75,         76,         77,
                   78,         79,         81,         82,         83,
                   84,         85,         86,         87,         88,
                   89,         91,         92,         93,         94,
                   95,         96,         97,         98,         99,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference137) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, -1, 2, -1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(4, 5));

  EXPECT_THAT(res.data,
              ElementsAreArray({1,  1,  3,  5,  7,  21, 21, 23, 25, 27,
                                41, 41, 43, 45, 47, 61, 61, 63, 65, 67}));
}

TEST(ReferenceTest, RandomJaxReference138) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{0, -1, 1, 2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(18, 22));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, 1,           -2147483647, 2,           -2147483647,
           3,           -2147483647, 4,           -2147483647, 5,
           -2147483647, 6,           -2147483647, 7,           -2147483647,
           8,           -2147483647, 9,           -2147483647, 10,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           11,          -2147483647, 12,          -2147483647, 13,
           -2147483647, 14,          -2147483647, 15,          -2147483647,
           16,          -2147483647, 17,          -2147483647, 18,
           -2147483647, 19,          -2147483647, 20,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, 21,
           -2147483647, 22,          -2147483647, 23,          -2147483647,
           24,          -2147483647, 25,          -2147483647, 26,
           -2147483647, 27,          -2147483647, 28,          -2147483647,
           29,          -2147483647, 30,          -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, 31,          -2147483647,
           32,          -2147483647, 33,          -2147483647, 34,
           -2147483647, 35,          -2147483647, 36,          -2147483647,
           37,          -2147483647, 38,          -2147483647, 39,
           -2147483647, 40,          -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, 41,          -2147483647, 42,
           -2147483647, 43,          -2147483647, 44,          -2147483647,
           45,          -2147483647, 46,          -2147483647, 47,
           -2147483647, 48,          -2147483647, 49,          -2147483647,
           50,          -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, 51,          -2147483647, 52,          -2147483647,
           53,          -2147483647, 54,          -2147483647, 55,
           -2147483647, 56,          -2147483647, 57,          -2147483647,
           58,          -2147483647, 59,          -2147483647, 60,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           61,          -2147483647, 62,          -2147483647, 63,
           -2147483647, 64,          -2147483647, 65,          -2147483647,
           66,          -2147483647, 67,          -2147483647, 68,
           -2147483647, 69,          -2147483647, 70,          -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, 71,
           -2147483647, 72,          -2147483647, 73,          -2147483647,
           74,          -2147483647, 75,          -2147483647, 76,
           -2147483647, 77,          -2147483647, 78,          -2147483647,
           79,          -2147483647, 80,          -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, 81,          -2147483647,
           82,          -2147483647, 83,          -2147483647, 84,
           -2147483647, 85,          -2147483647, 86,          -2147483647,
           87,          -2147483647, 88,          -2147483647, 89,
           -2147483647, 90,          -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference139) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{1, 2, 0, 2},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(6, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {132,      156,      182,      210,      240,      272,      306,
           342,      380,      20,       130944,   164736,   204204,   249900,
           302400,   362304,   430236,   506844,   592800,   800,      2630784,
           2910336,  3211164,  3534300,  3880800,  4251744,  4648236,  5071404,
           5522400,  2400,     13557024, 14485536, 15460524, 16483500, 17556000,
           18679584, 19855836, 21086364, 22372800, 4800,     42797664, 44970336,
           47224284, 49561500, 51984000, 54493824, 57093036, 59783724, 62568000,
           8000,     8372,     8556,     8742,     8930,     9120,     9312,
           9506,     9702,     9900,     100}));
}

TEST(ReferenceTest, RandomJaxReference140) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{-2, 0, -1, -2},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(15, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(ReferenceTest, RandomJaxReference141) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-2, -2, 1, 1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(3, 6));

  EXPECT_THAT(res.data, ElementsAreArray({-2147483647, 22, 24, 26, 28, 30,
                                          -2147483647, 42, 44, 46, 48, 50,
                                          -2147483647, 62, 64, 66, 68, 70}));
}

TEST(ReferenceTest, RandomJaxReference142) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, 0, 0, -1},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(18, 9));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,    1,    1,    1,    1,    1,    1,    1,    1,    11,   24,
           39,   56,   75,   96,   119,  144,  171,  1,    1,    1,    1,
           1,    1,    1,    1,    1,    231,  264,  299,  336,  375,  416,
           459,  504,  551,  1,    1,    1,    1,    1,    1,    1,    1,
           1,    651,  704,  759,  816,  875,  936,  999,  1064, 1131, 1,
           1,    1,    1,    1,    1,    1,    1,    1,    1271, 1344, 1419,
           1496, 1575, 1656, 1739, 1824, 1911, 1,    1,    1,    1,    1,
           1,    1,    1,    1,    2091, 2184, 2279, 2376, 2475, 2576, 2679,
           2784, 2891, 1,    1,    1,    1,    1,    1,    1,    1,    1,
           3111, 3224, 3339, 3456, 3575, 3696, 3819, 3944, 4071, 1,    1,
           1,    1,    1,    1,    1,    1,    1,    4331, 4464, 4599, 4736,
           4875, 5016, 5159, 5304, 5451, 1,    1,    1,    1,    1,    1,
           1,    1,    1,    5751, 5904, 6059, 6216, 6375, 6536, 6699, 6864,
           7031, 1,    1,    1,    1,    1,    1,    1,    1,    1,    7371,
           7544, 7719, 7896, 8075, 8256, 8439, 8624, 8811}));
}

TEST(ReferenceTest, RandomJaxReference143) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, -1, -2, -1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(5, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({2,  3,  4,  5,  6,  7,  8,  9,  22, 23, 24, 25, 26, 27,
                        28, 29, 42, 43, 44, 45, 46, 47, 48, 49, 62, 63, 64, 65,
                        66, 67, 68, 69, 82, 83, 84, 85, 86, 87, 88, 89}));
}

TEST(ReferenceTest, RandomJaxReference144) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{0, 1, 2, 2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(6, 23));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2147483646, 2147483646, 1,          2147483646, 2,
                   2147483646, 3,          2147483646, 4,          2147483646,
                   5,          2147483646, 6,          2147483646, 7,
                   2147483646, 8,          2147483646, 9,          2147483646,
                   10,         2147483646, 2147483646, 2147483646, 2147483646,
                   21,         2147483646, 22,         2147483646, 23,
                   2147483646, 24,         2147483646, 25,         2147483646,
                   26,         2147483646, 27,         2147483646, 28,
                   2147483646, 29,         2147483646, 30,         2147483646,
                   2147483646, 2147483646, 2147483646, 41,         2147483646,
                   42,         2147483646, 43,         2147483646, 44,
                   2147483646, 45,         2147483646, 46,         2147483646,
                   47,         2147483646, 48,         2147483646, 49,
                   2147483646, 50,         2147483646, 2147483646, 2147483646,
                   2147483646, 61,         2147483646, 62,         2147483646,
                   63,         2147483646, 64,         2147483646, 65,
                   2147483646, 66,         2147483646, 67,         2147483646,
                   68,         2147483646, 69,         2147483646, 70,
                   2147483646, 2147483646, 2147483646, 2147483646, 81,
                   2147483646, 82,         2147483646, 83,         2147483646,
                   84,         2147483646, 85,         2147483646, 86,
                   2147483646, 87,         2147483646, 88,         2147483646,
                   89,         2147483646, 90,         2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference145) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{2, -2, 2, -2},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1, 1,    1,    1,    1,    1, 2,    12,   30,   56,
                        1, 132,  182,  240,  306,  1, 462,  552,  650,  756,
                        1, 992,  1122, 1260, 1406, 1, 1722, 1892, 2070, 2256,
                        1, 2652, 2862, 3080, 3306, 1, 3782, 4032, 4290, 4556,
                        1, 5112, 5402, 5700, 6006, 1, 6642, 6972, 7310, 7656}));
}

TEST(ReferenceTest, RandomJaxReference146) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, -2, 1, 0},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(17, 6));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {2147483646, 2,  4,  6,  8,  10, 2147483646, 2,  4,  6,  8,  10,
           2147483646, 12, 14, 16, 18, 20, 2147483646, 12, 14, 16, 18, 20,
           2147483646, 22, 24, 26, 28, 30, 2147483646, 22, 24, 26, 28, 30,
           2147483646, 32, 34, 36, 38, 40, 2147483646, 32, 34, 36, 38, 40,
           2147483646, 42, 44, 46, 48, 50, 2147483646, 42, 44, 46, 48, 50,
           2147483646, 52, 54, 56, 58, 60, 2147483646, 52, 54, 56, 58, 60,
           2147483646, 62, 64, 66, 68, 70, 2147483646, 62, 64, 66, 68, 70,
           2147483646, 72, 74, 76, 78, 80, 2147483646, 72, 74, 76, 78, 80,
           2147483646, 82, 84, 86, 88, 90}));
}

TEST(ReferenceTest, RandomJaxReference147) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, 0, 2, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(8, 5));

  EXPECT_THAT(res.data, ElementsAreArray(
                            {11, 24,  28,  32,  36,  21, 44,  48,  52,  56,
                             31, 64,  68,  72,  76,  41, 84,  88,  92,  96,
                             51, 104, 108, 112, 116, 61, 124, 128, 132, 136,
                             71, 144, 148, 152, 156, 81, 164, 168, 172, 176}));
}

TEST(ReferenceTest, RandomJaxReference148) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{1, -2, 2, 1},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(17, 22));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,  1, 1,  1, 2,  1, 3,  1, 4,  1, 5,  1, 6,  1, 7,  1, 8,  1, 9,  1,
           10, 1, 1,  1, 1,  1, 2,  1, 3,  1, 4,  1, 5,  1, 6,  1, 7,  1, 8,  1,
           9,  1, 10, 1, 1,  1, 11, 1, 12, 1, 13, 1, 14, 1, 15, 1, 16, 1, 17, 1,
           18, 1, 19, 1, 20, 1, 1,  1, 11, 1, 12, 1, 13, 1, 14, 1, 15, 1, 16, 1,
           17, 1, 18, 1, 19, 1, 20, 1, 1,  1, 21, 1, 22, 1, 23, 1, 24, 1, 25, 1,
           26, 1, 27, 1, 28, 1, 29, 1, 30, 1, 1,  1, 21, 1, 22, 1, 23, 1, 24, 1,
           25, 1, 26, 1, 27, 1, 28, 1, 29, 1, 30, 1, 1,  1, 31, 1, 32, 1, 33, 1,
           34, 1, 35, 1, 36, 1, 37, 1, 38, 1, 39, 1, 40, 1, 1,  1, 31, 1, 32, 1,
           33, 1, 34, 1, 35, 1, 36, 1, 37, 1, 38, 1, 39, 1, 40, 1, 1,  1, 41, 1,
           42, 1, 43, 1, 44, 1, 45, 1, 46, 1, 47, 1, 48, 1, 49, 1, 50, 1, 1,  1,
           41, 1, 42, 1, 43, 1, 44, 1, 45, 1, 46, 1, 47, 1, 48, 1, 49, 1, 50, 1,
           1,  1, 51, 1, 52, 1, 53, 1, 54, 1, 55, 1, 56, 1, 57, 1, 58, 1, 59, 1,
           60, 1, 1,  1, 51, 1, 52, 1, 53, 1, 54, 1, 55, 1, 56, 1, 57, 1, 58, 1,
           59, 1, 60, 1, 1,  1, 61, 1, 62, 1, 63, 1, 64, 1, 65, 1, 66, 1, 67, 1,
           68, 1, 69, 1, 70, 1, 1,  1, 61, 1, 62, 1, 63, 1, 64, 1, 65, 1, 66, 1,
           67, 1, 68, 1, 69, 1, 70, 1, 1,  1, 71, 1, 72, 1, 73, 1, 74, 1, 75, 1,
           76, 1, 77, 1, 78, 1, 79, 1, 80, 1, 1,  1, 71, 1, 72, 1, 73, 1, 74, 1,
           75, 1, 76, 1, 77, 1, 78, 1, 79, 1, 80, 1, 1,  1, 81, 1, 82, 1, 83, 1,
           84, 1, 85, 1, 86, 1, 87, 1, 88, 1, 89, 1, 90, 1}));
}

TEST(ReferenceTest, RandomJaxReference149) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-1, -2, -2, 2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(6, 5));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {23, 25, 27, 29, -2147483647, 33, 35, 37, 39, -2147483647,
                   43, 45, 47, 49, -2147483647, 53, 55, 57, 59, -2147483647,
                   63, 65, 67, 69, -2147483647, 73, 75, 77, 79, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference150) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, -1, -2, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(6, 17));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({0, 0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0, 0,
                        0, 0,  2, 0,  3,  0,  4,  0,  5,  0,  6,  0, 7,  0, 8,
                        0, 9,  0, 10, 22, 0,  23, 0,  24, 0,  25, 0, 26, 0, 27,
                        0, 28, 0, 29, 0,  30, 42, 0,  43, 0,  44, 0, 45, 0, 46,
                        0, 47, 0, 48, 0,  49, 0,  50, 62, 0,  63, 0, 64, 0, 65,
                        0, 66, 0, 67, 0,  68, 0,  69, 0,  70, 82, 0, 83, 0, 84,
                        0, 85, 0, 86, 0,  87, 0,  88, 0,  89, 0,  90}));
}

TEST(ReferenceTest, RandomJaxReference151) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{2, 1, 2, -1},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 19));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,    1,    1,    2,    2,    3,    3,    4,    4,    5,    5,
           6,    6,    7,    7,    8,    8,    9,    9,    1,    11,   11,
           24,   24,   39,   39,   56,   56,   75,   75,   96,   96,   119,
           119,  144,  144,  171,  171,  1,    231,  231,  264,  264,  299,
           299,  336,  336,  375,  375,  416,  416,  459,  459,  504,  504,
           551,  551,  1,    651,  651,  704,  704,  759,  759,  816,  816,
           875,  875,  936,  936,  999,  999,  1064, 1064, 1131, 1131, 1,
           1271, 1271, 1344, 1344, 1419, 1419, 1496, 1496, 1575, 1575, 1656,
           1656, 1739, 1739, 1824, 1824, 1911, 1911, 1,    2091, 2091, 2184,
           2184, 2279, 2279, 2376, 2376, 2475, 2475, 2576, 2576, 2679, 2679,
           2784, 2784, 2891, 2891, 1,    3111, 3111, 3224, 3224, 3339, 3339,
           3456, 3456, 3575, 3575, 3696, 3696, 3819, 3819, 3944, 3944, 4071,
           4071, 1,    4331, 4331, 4464, 4464, 4599, 4599, 4736, 4736, 4875,
           4875, 5016, 5016, 5159, 5159, 5304, 5304, 5451, 5451, 1,    5751,
           5751, 5904, 5904, 6059, 6059, 6216, 6216, 6375, 6375, 6536, 6536,
           6699, 6699, 6864, 6864, 7031, 7031, 1,    7371, 7371, 7544, 7544,
           7719, 7719, 7896, 7896, 8075, 8075, 8256, 8256, 8439, 8439, 8624,
           8624, 8811, 8811}));
}

TEST(ReferenceTest, RandomJaxReference152) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-1, 2, -2, 1},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(9, 7));

  EXPECT_THAT(res.data,
              ElementsAreArray({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

TEST(ReferenceTest, RandomJaxReference153) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, 2, -1, 2},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(9, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {88704,    139776,   209664,   302400,   600,      574464,   763776,
           995904,   1276800,  1200,     2010624,  2477376,  3020544,  3648000,
           2000,     5189184,  6120576,  7171584,  8352000,  3000,     11142144,
           12773376, 14577024, 16564800, 4200,     21141504, 23755776, 26604864,
           29702400, 5600,     36699264, 40627776, 44863104, 49420800, 7200,
           59567424, 65189376, 71199744, 77616000, 9000,     8648,     9024,
           9408,     9800,     100}));
}

TEST(ReferenceTest, RandomJaxReference154) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, 2, -1, 0},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(7, 18));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, 2,
           -2147483647, 3,           -2147483647, 4,           -2147483647,
           5,           -2147483647, 6,           -2147483647, 7,
           -2147483647, 8,           -2147483647, 9,           -2147483647,
           10,          -2147483647, 22,          -2147483647, 23,
           -2147483647, 24,          -2147483647, 25,          -2147483647,
           26,          -2147483647, 27,          -2147483647, 28,
           -2147483647, 29,          -2147483647, 30,          -2147483647,
           42,          -2147483647, 43,          -2147483647, 44,
           -2147483647, 45,          -2147483647, 46,          -2147483647,
           47,          -2147483647, 48,          -2147483647, 49,
           -2147483647, 50,          -2147483647, 62,          -2147483647,
           63,          -2147483647, 64,          -2147483647, 65,
           -2147483647, 66,          -2147483647, 67,          -2147483647,
           68,          -2147483647, 69,          -2147483647, 70,
           -2147483647, 82,          -2147483647, 83,          -2147483647,
           84,          -2147483647, 85,          -2147483647, 86,
           -2147483647, 87,          -2147483647, 88,          -2147483647,
           89,          -2147483647, 90,          -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference155) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, 2, 2, 1},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(11, 6));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({0,   3,   7,   11,  15,  19,  0,   23,  27,  31,  35,
                        39,  0,   43,  47,  51,  55,  59,  0,   63,  67,  71,
                        75,  79,  0,   83,  87,  91,  95,  99,  0,   103, 107,
                        111, 115, 119, 0,   123, 127, 131, 135, 139, 0,   143,
                        147, 151, 155, 159, 0,   163, 167, 171, 175, 179, 0,
                        183, 187, 191, 195, 199, 0,   0,   0,   0,   0,   0}));
}

TEST(ReferenceTest, RandomJaxReference156) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{2, -1, -1, -1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(11, 3));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, 4,           6,           8,           14,
           16,          18,          24,          26,          28,
           34,          36,          38,          44,          46,
           48,          54,          56,          58,          64,
           66,          68,          74,          76,          78,
           84,          86,          88}));
}

TEST(ReferenceTest, RandomJaxReference157) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-1, -1, -2, -1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(16, 7));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {13, 14, 15, 16, 17, 18, 19, 13, 14, 15, 16, 17, 18, 19, 23, 24,
           25, 26, 27, 28, 29, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36,
           37, 38, 39, 33, 34, 35, 36, 37, 38, 39, 43, 44, 45, 46, 47, 48,
           49, 43, 44, 45, 46, 47, 48, 49, 53, 54, 55, 56, 57, 58, 59, 53,
           54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68, 69, 63, 64, 65,
           66, 67, 68, 69, 73, 74, 75, 76, 77, 78, 79, 73, 74, 75, 76, 77,
           78, 79, 83, 84, 85, 86, 87, 88, 89, 83, 84, 85, 86, 87, 88, 89}));
}

TEST(ReferenceTest, RandomJaxReference158) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{2, -2, 2, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(9, 6));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({0,  0,  0,  0,  0,  0,  0,  1,  3,  5,  7,  9,  0,  11,
                        13, 15, 17, 19, 0,  21, 23, 25, 27, 29, 0,  31, 33, 35,
                        37, 39, 0,  41, 43, 45, 47, 49, 0,  51, 53, 55, 57, 59,
                        0,  61, 63, 65, 67, 69, 0,  71, 73, 75, 77, 79}));
}

TEST(ReferenceTest, RandomJaxReference159) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{1, 2, -1, 1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(13, 9));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2,
                   3,          4,          5,          6,          7,
                   8,          9,          10,         12,         13,
                   14,         15,         16,         17,         18,
                   19,         20,         22,         23,         24,
                   25,         26,         27,         28,         29,
                   30,         32,         33,         34,         35,
                   36,         37,         38,         39,         40,
                   42,         43,         44,         45,         46,
                   47,         48,         49,         50,         52,
                   53,         54,         55,         56,         57,
                   58,         59,         60,         62,         63,
                   64,         65,         66,         67,         68,
                   69,         70,         72,         73,         74,
                   75,         76,         77,         78,         79,
                   80,         82,         83,         84,         85,
                   86,         87,         88,         89,         90,
                   92,         93,         94,         95,         96,
                   97,         98,         99,         100,        2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference160) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-1, 0, -2, 1},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(4, 4));

  EXPECT_THAT(res.data,
              ElementsAreArray({74, 82, 90, 98, 154, 162, 170, 178, 234, 242,
                                250, 258, 314, 322, 330, 338}));
}

TEST(ReferenceTest, RandomJaxReference161) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, 2, -1, 1},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(11, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({5,   9,   13,  17,  10,  25,  29,  33,  37,  20,  45,
                        49,  53,  57,  30,  65,  69,  73,  77,  40,  85,  89,
                        93,  97,  50,  105, 109, 113, 117, 60,  125, 129, 133,
                        137, 70,  145, 149, 153, 157, 80,  165, 169, 173, 177,
                        90,  185, 189, 193, 197, 100, 0,   0,   0,   0,   0}));
}

TEST(ReferenceTest, RandomJaxReference162) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, -1, -1, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 17));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   2,   2,   3,   3,   4,   4,   5,   5,   6,   6,   7,
           7,   8,   8,   9,   9,   10,  14,  14,  16,  16,  18,  18,  20,  20,
           22,  22,  24,  24,  26,  26,  28,  28,  30,  34,  34,  36,  36,  38,
           38,  40,  40,  42,  42,  44,  44,  46,  46,  48,  48,  50,  54,  54,
           56,  56,  58,  58,  60,  60,  62,  62,  64,  64,  66,  66,  68,  68,
           70,  74,  74,  76,  76,  78,  78,  80,  80,  82,  82,  84,  84,  86,
           86,  88,  88,  90,  94,  94,  96,  96,  98,  98,  100, 100, 102, 102,
           104, 104, 106, 106, 108, 108, 110, 114, 114, 116, 116, 118, 118, 120,
           120, 122, 122, 124, 124, 126, 126, 128, 128, 130, 134, 134, 136, 136,
           138, 138, 140, 140, 142, 142, 144, 144, 146, 146, 148, 148, 150, 154,
           154, 156, 156, 158, 158, 160, 160, 162, 162, 164, 164, 166, 166, 168,
           168, 170}));
}

TEST(ReferenceTest, RandomJaxReference163) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, 0, 0, 2},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 12));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1,  2,  3,  4,  5,  6,   7,  8,  9,  10, 0,  0,  11, 12,
                        13, 14, 15, 16, 17, 18,  19, 20, 0,  0,  21, 22, 23, 24,
                        25, 26, 27, 28, 29, 30,  0,  0,  31, 32, 33, 34, 35, 36,
                        37, 38, 39, 40, 0,  0,   41, 42, 43, 44, 45, 46, 47, 48,
                        49, 50, 0,  0,  51, 52,  53, 54, 55, 56, 57, 58, 59, 60,
                        0,  0,  61, 62, 63, 64,  65, 66, 67, 68, 69, 70, 0,  0,
                        71, 72, 73, 74, 75, 76,  77, 78, 79, 80, 0,  0,  81, 82,
                        83, 84, 85, 86, 87, 88,  89, 90, 0,  0,  91, 92, 93, 94,
                        95, 96, 97, 98, 99, 100, 0,  0}));
}

TEST(ReferenceTest, RandomJaxReference164) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-1, 0, 2, 1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 7));

  EXPECT_THAT(res.data,
              ElementsAreArray({-2147483647, 11, 13, 15, 17, 19, -2147483647,
                                -2147483647, 21, 23, 25, 27, 29, -2147483647,
                                -2147483647, 31, 33, 35, 37, 39, -2147483647,
                                -2147483647, 41, 43, 45, 47, 49, -2147483647,
                                -2147483647, 51, 53, 55, 57, 59, -2147483647,
                                -2147483647, 61, 63, 65, 67, 69, -2147483647,
                                -2147483647, 71, 73, 75, 77, 79, -2147483647,
                                -2147483647, 81, 83, 85, 87, 89, -2147483647,
                                -2147483647, 91, 93, 95, 97, 99, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference165) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, -2, 2, -1},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(5, 18));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1,    1, 1,    1, 1,    1, 1,    1, 1,    1, 1,    1,
                        1,    1, 1,    1, 1,    1, 1,    1, 2,    1, 6,    1,
                        12,   1, 20,   1, 30,   1, 42,   1, 56,   1, 72,   1,
                        21,   1, 462,  1, 506,  1, 552,  1, 600,  1, 650,  1,
                        702,  1, 756,  1, 812,  1, 41,   1, 1722, 1, 1806, 1,
                        1892, 1, 1980, 1, 2070, 1, 2162, 1, 2256, 1, 2352, 1,
                        61,   1, 3782, 1, 3906, 1, 4032, 1, 4160, 1, 4290, 1,
                        4422, 1, 4556, 1, 4692, 1}));
}

TEST(ReferenceTest, RandomJaxReference166) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{1, -1, 0, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(8, 3));

  EXPECT_THAT(res.data, ElementsAreArray({13, 15, 17, 23, 25, 27, 33, 35,
                                          37, 43, 45, 47, 53, 55, 57, 63,
                                          65, 67, 73, 75, 77, 83, 85, 87}));
}

TEST(ReferenceTest, RandomJaxReference167) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, 2, 0, 1},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(9, 5));

  EXPECT_THAT(res.data,
              ElementsAreArray({66,  74,  82,  90,  98,  106, 114, 122, 130,
                                138, 146, 154, 162, 170, 178, 186, 194, 202,
                                210, 218, 226, 234, 242, 250, 258, 266, 274,
                                282, 290, 298, 306, 314, 322, 330, 338, 346,
                                354, 362, 370, 378, 183, 187, 191, 195, 199}));
}

TEST(ReferenceTest, RandomJaxReference168) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{0, -1, -1, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(4, 7));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({-2147483647, -2147483647, -2147483647, -2147483647,
                        -2147483647, -2147483647, -2147483647, -2147483647,
                        -2147483647, -2147483647, -2147483647, -2147483647,
                        -2147483647, -2147483647, -2147483647, -2147483647,
                        -2147483647, -2147483647, -2147483647, -2147483647,
                        -2147483647, -2147483647, -2147483647, -2147483647,
                        -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference169) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{1, -2, 0, 1},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(9, 9));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1,    1,    1,    1,    1,    1,    1,    1,    1,
                        2,    6,    12,   20,   30,   42,   56,   72,   90,
                        132,  156,  182,  210,  240,  272,  306,  342,  380,
                        462,  506,  552,  600,  650,  702,  756,  812,  870,
                        992,  1056, 1122, 1190, 1260, 1332, 1406, 1482, 1560,
                        1722, 1806, 1892, 1980, 2070, 2162, 2256, 2352, 2450,
                        2652, 2756, 2862, 2970, 3080, 3192, 3306, 3422, 3540,
                        3782, 3906, 4032, 4160, 4290, 4422, 4556, 4692, 4830,
                        5112, 5256, 5402, 5550, 5700, 5852, 6006, 6162, 6320}));
}

TEST(ReferenceTest, RandomJaxReference170) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, -1, 0, -1},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(6, 18));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({1,  1, 1,  1, 1,  1, 1,  1, 1,  1, 1,  1, 1,  1, 1,  1,
                        1,  1, 1,  1, 2,  1, 3,  1, 4,  1, 5,  1, 6,  1, 7,  1,
                        8,  1, 9,  1, 21, 1, 22, 1, 23, 1, 24, 1, 25, 1, 26, 1,
                        27, 1, 28, 1, 29, 1, 41, 1, 42, 1, 43, 1, 44, 1, 45, 1,
                        46, 1, 47, 1, 48, 1, 49, 1, 61, 1, 62, 1, 63, 1, 64, 1,
                        65, 1, 66, 1, 67, 1, 68, 1, 69, 1, 81, 1, 82, 1, 83, 1,
                        84, 1, 85, 1, 86, 1, 87, 1, 88, 1, 89, 1}));
}

TEST(ReferenceTest, RandomJaxReference171) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{0, -2, 2, 0},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(7, 10));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {1, 11,   24,   39,   56,   75,   96,   119,  144,  171,
                   1, 231,  264,  299,  336,  375,  416,  459,  504,  551,
                   1, 651,  704,  759,  816,  875,  936,  999,  1064, 1131,
                   1, 1271, 1344, 1419, 1496, 1575, 1656, 1739, 1824, 1911,
                   1, 2091, 2184, 2279, 2376, 2475, 2576, 2679, 2784, 2891,
                   1, 3111, 3224, 3339, 3456, 3575, 3696, 3819, 3944, 4071,
                   1, 4331, 4464, 4599, 4736, 4875, 5016, 5159, 5304, 5451}));
}

TEST(ReferenceTest, RandomJaxReference172) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{-1, 1, 2, 2},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 12));

  EXPECT_THAT(
      res.data,
      ElementsAreArray({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
}

TEST(ReferenceTest, RandomJaxReference173) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{1, 1, 1, 0},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(21, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           1,           2,           3,           4,           5,
           6,           7,           8,           9,           10,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           11,          12,          13,          14,          15,
           16,          17,          18,          19,          20,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           21,          22,          23,          24,          25,
           26,          27,          28,          29,          30,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           31,          32,          33,          34,          35,
           36,          37,          38,          39,          40,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           41,          42,          43,          44,          45,
           46,          47,          48,          49,          50,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           51,          52,          53,          54,          55,
           56,          57,          58,          59,          60,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           61,          62,          63,          64,          65,
           66,          67,          68,          69,          70,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           71,          72,          73,          74,          75,
           76,          77,          78,          79,          80,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           81,          82,          83,          84,          85,
           86,          87,          88,          89,          90,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           91,          92,          93,          94,          95,
           96,          97,          98,          99,          100,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference174) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{0, -1, -2, -1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(18, 7));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2,          3,          4,          5,          6,
                   7,          8,          2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 12,
                   13,         14,         15,         16,         17,
                   18,         2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 22,         23,
                   24,         25,         26,         27,         28,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 32,         33,         34,
                   35,         36,         37,         38,         2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 42,         43,         44,         45,
                   46,         47,         48,         2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   52,         53,         54,         55,         56,
                   57,         58,         2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 62,
                   63,         64,         65,         66,         67,
                   68,         2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 72,         73,
                   74,         75,         76,         77,         78,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 82,         83,         84,
                   85,         86,         87,         88,         2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646}));
}

TEST(ReferenceTest, RandomJaxReference175) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-2, 2, 0, 0},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(9, 9));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {458304,   534336,   619344,   714000,   819000,   935064,   1062936,
           1203384,  1357200,  1708224,  1907136,  2122824,  2356200,  2608200,
           2879784,  3171936,  3485664,  3822000,  4566744,  4977336,  5414904,
           5880600,  6375600,  6901104,  7458336,  8048544,  8673000,  10029864,
           10764936, 11539584, 12355200, 13213200, 14115024, 15062136, 16056024,
           17098200, 19333584, 20529936, 21780864, 23088000, 24453000, 25877544,
           27363336, 28912104, 30525600, 33953904, 35772336, 37662744, 39627000,
           41667000, 43784664, 45981936, 48260784, 50623200, 55606824, 58232136,
           60949224, 63760200, 66667200, 69672384, 72777936, 75986064, 79299000,
           8372,     8556,     8742,     8930,     9120,     9312,     9506,
           9702,     9900,     1,        1,        1,        1,        1,
           1,        1,        1,        1}));
}

TEST(ReferenceTest, RandomJaxReference176) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, -2, -1, 2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(8, 10));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference177) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{0, 1, 0, 2},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(10, 5));

  EXPECT_THAT(res.data, ElementsAreArray(
                            {4,   8,   12,  16,  9,  24,  28,  32,  36,  19,
                             44,  48,  52,  56,  29, 64,  68,  72,  76,  39,
                             84,  88,  92,  96,  49, 104, 108, 112, 116, 59,
                             124, 128, 132, 136, 69, 144, 148, 152, 156, 79,
                             164, 168, 172, 176, 89, 184, 188, 192, 196, 99}));
}

TEST(ReferenceTest, RandomJaxReference178) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{2, 1, 2, 1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(7, 11));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 1,          2,          1,          2,
                   3,          4,          5,          6,          7,
                   8,          9,          21,         22,         21,
                   22,         23,         24,         25,         26,
                   27,         28,         29,         41,         42,
                   41,         42,         43,         44,         45,
                   46,         47,         48,         49,         61,
                   62,         61,         62,         63,         64,
                   65,         66,         67,         68,         69,
                   81,         82,         81,         82,         83,
                   84,         85,         86,         87,         88,
                   89,         2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference179) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-2, -2, 2, 0},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 2},
      /*window_strides=*/{2, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(3, 11));

  EXPECT_THAT(res.data,
              ElementsAreArray({1, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                1, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                1, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70}));
}

TEST(ReferenceTest, RandomJaxReference180) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{1, -2, 1, 0},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 6));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, 2,           4,           6,
           8,           10,          -2147483647, 12,          14,
           16,          18,          20,          -2147483647, 22,
           24,          26,          28,          30,          -2147483647,
           32,          34,          36,          38,          40,
           -2147483647, 42,          44,          46,          48,
           50,          -2147483647, 52,          54,          56,
           58,          60,          -2147483647, 62,          64,
           66,          68,          70,          -2147483647, 72,
           74,          76,          78,          80}));
}

TEST(ReferenceTest, RandomJaxReference181) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-2, -1, -1, -2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(7, 8));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646}));
}

TEST(ReferenceTest, RandomJaxReference182) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-1, -1, 2, -1},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(3, 20));

  EXPECT_THAT(res.data, ElementsAreArray(
                            {0,   0, 42,  0, 44,  0, 46,  0, 48,  0, 50,  0,
                             52,  0, 54,  0, 56,  0, 58,  0, 0,   0, 82,  0,
                             84,  0, 86,  0, 88,  0, 90,  0, 92,  0, 94,  0,
                             96,  0, 98,  0, 0,   0, 122, 0, 124, 0, 126, 0,
                             128, 0, 130, 0, 132, 0, 134, 0, 136, 0, 138, 0}));
}

TEST(ReferenceTest, RandomJaxReference183) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{1, -1, -2, -1},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(17, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {1,    1,    1,    1,    1,    1,    1,    1,    24,   39,   56,
           75,   96,   119,  144,  171,  1,    1,    1,    1,    1,    1,
           1,    1,    264,  299,  336,  375,  416,  459,  504,  551,  1,
           1,    1,    1,    1,    1,    1,    1,    704,  759,  816,  875,
           936,  999,  1064, 1131, 1,    1,    1,    1,    1,    1,    1,
           1,    1344, 1419, 1496, 1575, 1656, 1739, 1824, 1911, 1,    1,
           1,    1,    1,    1,    1,    1,    2184, 2279, 2376, 2475, 2576,
           2679, 2784, 2891, 1,    1,    1,    1,    1,    1,    1,    1,
           3224, 3339, 3456, 3575, 3696, 3819, 3944, 4071, 1,    1,    1,
           1,    1,    1,    1,    1,    4464, 4599, 4736, 4875, 5016, 5159,
           5304, 5451, 1,    1,    1,    1,    1,    1,    1,    1,    5904,
           6059, 6216, 6375, 6536, 6699, 6864, 7031, 1,    1,    1,    1,
           1,    1,    1,    1}));
}

TEST(ReferenceTest, RandomJaxReference184) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-1, 1, 2, -1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
           2147483646, 2147483646, 2147483646, 2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference185) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{2, -1, -2, 0},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(11, 9));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
           0,  2,  3,  4,  5,  6,  7,  8,  9,  10, 12, 13, 14, 15, 16, 17, 18,
           19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37,
           38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56,
           57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75,
           76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90}));
}

TEST(ReferenceTest, RandomJaxReference186) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, 0, 0, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{1, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 7));

  EXPECT_THAT(res.data, ElementsAreArray(
                            {12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27,
                             28, 32, 33, 34, 35, 36, 37, 38, 42, 43, 44, 45, 46,
                             47, 48, 52, 53, 54, 55, 56, 57, 58, 62, 63, 64, 65,
                             66, 67, 68, 72, 73, 74, 75, 76, 77, 78, 82, 83, 84,
                             85, 86, 87, 88, 92, 93, 94, 95, 96, 97, 98}));
}

TEST(ReferenceTest, RandomJaxReference187) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{0, 0, 0, -2},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(4, 4));

  EXPECT_THAT(res.data, ElementsAreArray({1, 3, 5, 7, 21, 23, 25, 27, 41, 43,
                                          45, 47, 61, 63, 65, 67}));
}

TEST(ReferenceTest, RandomJaxReference188) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{2, 2, -1, -1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 1},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(11, 17));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, 2,           -2147483647, 3,           -2147483647,
           4,           -2147483647, 5,           -2147483647, 6,
           -2147483647, 7,           -2147483647, 8,           -2147483647,
           9,           -2147483647, -2147483647, 12,          -2147483647,
           13,          -2147483647, 14,          -2147483647, 15,
           -2147483647, 16,          -2147483647, 17,          -2147483647,
           18,          -2147483647, 19,          -2147483647, -2147483647,
           22,          -2147483647, 23,          -2147483647, 24,
           -2147483647, 25,          -2147483647, 26,          -2147483647,
           27,          -2147483647, 28,          -2147483647, 29,
           -2147483647, -2147483647, 32,          -2147483647, 33,
           -2147483647, 34,          -2147483647, 35,          -2147483647,
           36,          -2147483647, 37,          -2147483647, 38,
           -2147483647, 39,          -2147483647, -2147483647, 42,
           -2147483647, 43,          -2147483647, 44,          -2147483647,
           45,          -2147483647, 46,          -2147483647, 47,
           -2147483647, 48,          -2147483647, 49,          -2147483647,
           -2147483647, 52,          -2147483647, 53,          -2147483647,
           54,          -2147483647, 55,          -2147483647, 56,
           -2147483647, 57,          -2147483647, 58,          -2147483647,
           59,          -2147483647, -2147483647, 62,          -2147483647,
           63,          -2147483647, 64,          -2147483647, 65,
           -2147483647, 66,          -2147483647, 67,          -2147483647,
           68,          -2147483647, 69,          -2147483647, -2147483647,
           72,          -2147483647, 73,          -2147483647, 74,
           -2147483647, 75,          -2147483647, 76,          -2147483647,
           77,          -2147483647, 78,          -2147483647, 79,
           -2147483647, -2147483647, 82,          -2147483647, 83,
           -2147483647, 84,          -2147483647, 85,          -2147483647,
           86,          -2147483647, 87,          -2147483647, 88,
           -2147483647, 89,          -2147483647, -2147483647, 92,
           -2147483647, 93,          -2147483647, 94,          -2147483647,
           95,          -2147483647, 96,          -2147483647, 97,
           -2147483647, 98,          -2147483647, 99,          -2147483647,
           -2147483647, 92,          -2147483647, 93,          -2147483647,
           94,          -2147483647, 95,          -2147483647, 96,
           -2147483647, 97,          -2147483647, 98,          -2147483647,
           99,          -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference189) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{1, 0, -2, 2},
      /*init_value=*/0,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{1, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(5, 5));

  EXPECT_THAT(res.data,
              ElementsAreArray({7,   11,  15,  19,  0,   74,  82,  90,  98,
                                0,   154, 162, 170, 178, 0,   234, 242, 250,
                                258, 0,   314, 322, 330, 338, 0}));
}

TEST(ReferenceTest, RandomJaxReference190) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{1, -1, 2, 0},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(9, 6));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference191) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{1, 0, 2, -2},
      /*init_value=*/0,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/std::plus<>());
  EXPECT_THAT(res.shape, ElementsAre(11, 5));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {0,   0,   0,   0,   0,   0,   3,   7,   11,  15,  0,   23,  27, 31,
           35,  0,   43,  47,  51,  55,  0,   63,  67,  71,  75,  0,   83, 87,
           91,  95,  0,   103, 107, 111, 115, 0,   123, 127, 131, 135, 0,  143,
           147, 151, 155, 0,   163, 167, 171, 175, 0,   183, 187, 191, 195}));
}

TEST(ReferenceTest, RandomJaxReference192) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{-1, 2, 0, -1},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(11, 4));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {11,         13,         15,         17,         21,
                   23,         25,         27,         31,         33,
                   35,         37,         41,         43,         45,
                   47,         51,         53,         55,         57,
                   61,         63,         65,         67,         71,
                   73,         75,         77,         81,         83,
                   85,         87,         91,         93,         95,
                   97,         2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference193) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{2, 2, 1, 0},
      /*init_value=*/1,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{1, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(21, 6));

  EXPECT_THAT(res.data, ElementsAreArray(
                            {1, 2,    4,    6,    8,    10,   1, 1, 1, 1, 1, 1,
                             1, 24,   56,   96,   144,  200,  1, 1, 1, 1, 1, 1,
                             1, 264,  336,  416,  504,  600,  1, 1, 1, 1, 1, 1,
                             1, 704,  816,  936,  1064, 1200, 1, 1, 1, 1, 1, 1,
                             1, 1344, 1496, 1656, 1824, 2000, 1, 1, 1, 1, 1, 1,
                             1, 2184, 2376, 2576, 2784, 3000, 1, 1, 1, 1, 1, 1,
                             1, 3224, 3456, 3696, 3944, 4200, 1, 1, 1, 1, 1, 1,
                             1, 4464, 4736, 5016, 5304, 5600, 1, 1, 1, 1, 1, 1,
                             1, 5904, 6216, 6536, 6864, 7200, 1, 1, 1, 1, 1, 1,
                             1, 7544, 7896, 8256, 8624, 9000, 1, 1, 1, 1, 1, 1,
                             1, 92,   94,   96,   98,   100}));
}

TEST(ReferenceTest, RandomJaxReference194) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{-2, -1, -2, -1},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(7, 8));

  EXPECT_THAT(res.data,
              ElementsAreArray({22, 23, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35,
                                36, 37, 38, 39, 42, 43, 44, 45, 46, 47, 48, 49,
                                52, 53, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65,
                                66, 67, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79,
                                82, 83, 84, 85, 86, 87, 88, 89}));
}

TEST(ReferenceTest, RandomJaxReference195) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{-2, 1, -2, 2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(5, 9));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {23,          24,          25,          26,          27,
           28,          29,          30,          30,          43,
           44,          45,          46,          47,          48,
           49,          50,          50,          63,          64,
           65,          66,          67,          68,          69,
           70,          70,          83,          84,          85,
           86,          87,          88,          89,          90,
           90,          -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647}));
}

TEST(ReferenceTest, RandomJaxReference196) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 1},
      /*padding=*/{1, 1, 1, -1},
      /*init_value=*/1,
      /*window_dimensions=*/{1, 2},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{2, 2},
      /*body=*/std::multiplies<>());
  EXPECT_THAT(res.shape, ElementsAre(6, 5));

  EXPECT_THAT(res.data,
              ElementsAreArray({1,    1,    1,    1,    1,    11,   156,  210,
                                272,  342,  31,   1056, 1190, 1332, 1482, 51,
                                2756, 2970, 3192, 3422, 71,   5256, 5550, 5852,
                                6162, 91,   8556, 8930, 9312, 9702}));
}

TEST(ReferenceTest, RandomJaxReference197) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 1},
      /*padding=*/{-2, -2, -2, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 2},
      /*window_strides=*/{2, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(7, 3));

  EXPECT_THAT(res.data,
              ElementsAreArray({23, 25, 27, 33, 35, 37, 43, 45, 47, 53, 55,
                                57, 63, 65, 67, 73, 75, 77, 83, 85, 87}));
}

TEST(ReferenceTest, RandomJaxReference198) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{1, 2},
      /*padding=*/{1, 1, -2, 0},
      /*init_value=*/2147483646,
      /*window_dimensions=*/{1, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a <= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(12, 9));

  EXPECT_THAT(res.data,
              ElementsAreArray(
                  {2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2,
                   3,          4,          5,          6,          7,
                   8,          9,          10,         12,         13,
                   14,         15,         16,         17,         18,
                   19,         20,         22,         23,         24,
                   25,         26,         27,         28,         29,
                   30,         32,         33,         34,         35,
                   36,         37,         38,         39,         40,
                   42,         43,         44,         45,         46,
                   47,         48,         49,         50,         52,
                   53,         54,         55,         56,         57,
                   58,         59,         60,         62,         63,
                   64,         65,         66,         67,         68,
                   69,         70,         72,         73,         74,
                   75,         76,         77,         78,         79,
                   80,         82,         83,         84,         85,
                   86,         87,         88,         89,         90,
                   92,         93,         94,         95,         96,
                   97,         98,         99,         100,        2147483646,
                   2147483646, 2147483646, 2147483646, 2147483646, 2147483646,
                   2147483646, 2147483646, 2147483646}));
}

TEST(ReferenceTest, RandomJaxReference199) {
  const Tensor<int> res = ReduceWindow<int>(
      /*input=*/Tensor<int>::iota(/*shape=*/{10, 10}),
      /*base_dilations=*/{2, 2},
      /*padding=*/{-1, 1, -1, -2},
      /*init_value=*/-2147483647,
      /*window_dimensions=*/{2, 1},
      /*window_dilations=*/{2, 1},
      /*window_strides=*/{1, 2},
      /*body=*/[](auto a, auto b) { return a >= b ? a : b; });
  EXPECT_THAT(res.shape, ElementsAre(17, 8));

  EXPECT_THAT(
      res.data,
      ElementsAreArray(
          {-2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647, -2147483647, -2147483647, -2147483647, -2147483647,
           -2147483647}));
}

}  // namespace
}  // namespace tflite::reduce_window::reference
