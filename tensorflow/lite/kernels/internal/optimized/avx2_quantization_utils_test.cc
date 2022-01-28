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

__m256i FillVectorWithInt32(const std::vector<int32_t>& src) {
  return _mm256_set_epi32(src[7], src[6], src[5], src[4], src[3], src[2],
                          src[1], src[0]);
}

void CompareWithReferenceValue(std::vector<int32_t>& reference_values,
                               const __m256i& result) {
  // As _mm256_extract_epi32 only supports const int, which should be known
  // at the comile time, it puts down 8 comparison instead of for-loop.
  EXPECT_NEAR(reference_values[0], _mm256_extract_epi32(result, 0), 1);
  EXPECT_NEAR(reference_values[1], _mm256_extract_epi32(result, 1), 1);
  EXPECT_NEAR(reference_values[2], _mm256_extract_epi32(result, 2), 1);
  EXPECT_NEAR(reference_values[3], _mm256_extract_epi32(result, 3), 1);
  EXPECT_NEAR(reference_values[4], _mm256_extract_epi32(result, 4), 1);
  EXPECT_NEAR(reference_values[5], _mm256_extract_epi32(result, 5), 1);
  EXPECT_NEAR(reference_values[6], _mm256_extract_epi32(result, 6), 1);
  EXPECT_NEAR(reference_values[7], _mm256_extract_epi32(result, 7), 1);
}

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
  const __m256i src_vector = FillVectorWithInt32(values);
  const int32_t left_shift = 20;
  const int32_t multiplier = 12345;
  const __m256i result =
      MultiplyByQuantizedMultiplier(src_vector, multiplier, left_shift);

  // Get the reference values.
  for (int i = 0; i < values.size(); i++) {
    values[i] = tflite::MultiplyByQuantizedMultiplier(values[i], multiplier,
                                                      left_shift);
  }

  CompareWithReferenceValue(values, result);
}

TEST(MultiplyByQuantizedMultiplierTest, NegativeLeftShiftTest) {
  std::vector<int32_t> values = {1000, 2000, 3000, 4000,
                                 5000, 6000, 7000, 8000};
  const __m256i src_vector = FillVectorWithInt32(values);
  const int32_t left_shift = -3;
  const int32_t multiplier = 1234567890;
  const __m256i result =
      MultiplyByQuantizedMultiplier(src_vector, multiplier, left_shift);

  // Get the reference values.
  for (int i = 0; i < values.size(); i++) {
    values[i] = tflite::MultiplyByQuantizedMultiplier(values[i], multiplier,
                                                      left_shift);
  }

  CompareWithReferenceValue(values, result);
}

TEST(MultiplyByQuantizedMultiplierTest, VectorPositiveLeftShiftTest) {
  std::vector<int32_t> values = {100, 200, 300, 400, 500, 600, 700, 800};
  const std::vector<int32_t> left_shifts = {20, 19, 18, 17, 16, 15, 14, 13};
  const std::vector<int32_t> multipliers = {10000, 20000, 30000, 40000,
                                            50000, 60000, 70000, 80000};
  const __m256i src_vector = FillVectorWithInt32(values);
  const __m256i left_shifts_vector = FillVectorWithInt32(left_shifts);
  const __m256i multipliers_vector = FillVectorWithInt32(multipliers);

  const __m256i result = MultiplyByQuantizedMultiplier(
      src_vector, multipliers_vector, left_shifts_vector);

  // Get the reference values.
  for (int i = 0; i < values.size(); i++) {
    values[i] = tflite::MultiplyByQuantizedMultiplier(values[i], multipliers[i],
                                                      left_shifts[i]);
  }

  CompareWithReferenceValue(values, result);
}

TEST(MultiplyByQuantizedMultiplierTest, VectorNegativeLeftShiftTest) {
  std::vector<int32_t> values = {1000, 2000, 3000, 4000,
                                 5000, 6000, 7000, 8000};
  const std::vector<int32_t> left_shifts = {-3, -4, -5, -6, -7, -8, -9, -10};
  const std::vector<int32_t> multipliers = {1000000000, 1100000000, 1200000000,
                                            1300000000, 1400000000, 1500000000,
                                            1600000000, 1700000000};
  const __m256i src_vector = FillVectorWithInt32(values);
  const __m256i left_shifts_vector = FillVectorWithInt32(left_shifts);
  const __m256i multipliers_vector = FillVectorWithInt32(multipliers);

  const __m256i result = MultiplyByQuantizedMultiplier(
      src_vector, multipliers_vector, left_shifts_vector);

  // Get the reference values.
  for (int i = 0; i < values.size(); i++) {
    values[i] = tflite::MultiplyByQuantizedMultiplier(values[i], multipliers[i],
                                                      left_shifts[i]);
  }

  CompareWithReferenceValue(values, result);
}

}  // namespace
}  // namespace avx2_utils
}  // namespace tflite

#endif  //  __AVX2__
