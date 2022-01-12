/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/optimized/avx2_quantization_utils.h"

#include <gmock/gmock.h>
#include "tensorflow/lite/kernels/internal/common.h"

#ifdef __AVX2__
namespace tflite {
namespace avx2_utils {
namespace {

using ::testing::ElementsAreArray;

TEST(CastInt32ToInt16AndStoreTest, CastInt32ToInt16AndStoreTest) {
  const std::vector<int16_t> src = {1, 2, 3, 4, 5, 6, 7, 8};
  int16_t dst[8];
  const __m256i src_vector = _mm256_set_epi32(src[7], src[6], src[5], src[4],
                                              src[3], src[2], src[1], src[0]);
  CastInt32ToInt16AndStore(dst, src_vector);
  EXPECT_THAT(src, ElementsAreArray(dst));
}

TEST(MultiplyByQuantizedMultiplierTest, PositiveLeftShiftTest) {
  std::vector<int32_t> values = {100, 200, 300, 400, 500, 600, 700, 800};
  const __m256i src_vector =
      _mm256_set_epi32(values[7], values[6], values[5], values[4], values[3],
                       values[2], values[1], values[0]);
  const int32_t left_shift = 20;
  const int32_t multiplier = 12345;
  const __m256i result =
      MultiplyByQuantizedMultiplier(src_vector, multiplier, left_shift);

  // Get the reference values.
  for (int i = 0; i < values.size(); i++) {
    values[i] = tflite::MultiplyByQuantizedMultiplier(values[i], multiplier,
                                                      left_shift);
  }

  // As _mm256_extract_epi32 only supports const int, which should be known
  // at the comile time, it puts down 8 comparison instead of for-loop.
  EXPECT_NEAR(values[0], _mm256_extract_epi32(result, 0), 1);
  EXPECT_NEAR(values[1], _mm256_extract_epi32(result, 1), 1);
  EXPECT_NEAR(values[2], _mm256_extract_epi32(result, 2), 1);
  EXPECT_NEAR(values[3], _mm256_extract_epi32(result, 3), 1);
  EXPECT_NEAR(values[4], _mm256_extract_epi32(result, 4), 1);
  EXPECT_NEAR(values[5], _mm256_extract_epi32(result, 5), 1);
  EXPECT_NEAR(values[6], _mm256_extract_epi32(result, 6), 1);
  EXPECT_NEAR(values[7], _mm256_extract_epi32(result, 7), 1);
}

TEST(MultiplyByQuantizedMultiplierTest, NegativeLeftShiftTest) {
  std::vector<int32_t> values = {1000, 2000, 3000, 4000,
                                 5000, 6000, 7000, 8000};
  const __m256i src_vector =
      _mm256_set_epi32(values[7], values[6], values[5], values[4], values[3],
                       values[2], values[1], values[0]);
  const int32_t left_shift = -3;
  const int32_t multiplier = 1234567890;
  const __m256i result =
      MultiplyByQuantizedMultiplier(src_vector, multiplier, left_shift);

  // Get the reference values.
  for (int i = 0; i < values.size(); i++) {
    values[i] = tflite::MultiplyByQuantizedMultiplier(values[i], multiplier,
                                                      left_shift);
  }

  // As _mm256_extract_epi32 only supports const int, which should be known
  // at the comile time, it puts down 8 comparison instead of for-loop.
  EXPECT_NEAR(values[0], _mm256_extract_epi32(result, 0), 1);
  EXPECT_NEAR(values[1], _mm256_extract_epi32(result, 1), 1);
  EXPECT_NEAR(values[2], _mm256_extract_epi32(result, 2), 1);
  EXPECT_NEAR(values[3], _mm256_extract_epi32(result, 3), 1);
  EXPECT_NEAR(values[4], _mm256_extract_epi32(result, 4), 1);
  EXPECT_NEAR(values[5], _mm256_extract_epi32(result, 5), 1);
  EXPECT_NEAR(values[6], _mm256_extract_epi32(result, 6), 1);
  EXPECT_NEAR(values[7], _mm256_extract_epi32(result, 7), 1);
}

}  // namespace
}  // namespace avx2_utils
}  // namespace tflite

#endif  //  __AVX2__
