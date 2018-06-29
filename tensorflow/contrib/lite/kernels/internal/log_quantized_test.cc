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
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define GEMMLOWP_ENABLE_FIXEDPOINT_CONSTANTS_CHECKS

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/string.h"

namespace tflite {

class NumberGenerator {
 public:
  std::vector<int> RandomIntVector(int n, int min_val, int max_val) {
    std::vector<int> vec(n);
    double scale = static_cast<double>(max_val + 1 - min_val) / engine_.max();
    for (auto& it : vec) {
      it = min_val + std::floor(engine_() * scale);
    }
    return vec;
  }

  std::mt19937 engine_;
};

class LogQuantizedTest : public ::testing::Test {
 public:
  NumberGenerator generator_;
};

// input_integer_bits <= 30.  output_integer_bits > 0.
inline int32 LogPositiveValuesViaFloat(int32 input_val, int input_integer_bits,
                                       int output_integer_bits) {
  const double float_log_sum_of_exps = std::log(
      static_cast<double>(input_val) * 0.5 / (1 << (30 - input_integer_bits)));
  static constexpr double min_int =
      static_cast<double>(std::numeric_limits<int32>::min());
  static constexpr double max_int =
      static_cast<double>(std::numeric_limits<int32>::max());
  double double_result = tflite::TfLiteRound(float_log_sum_of_exps *
                                             (1 << (31 - output_integer_bits)));
  return static_cast<std::int32_t>(
      std::min(max_int, std::max(min_int, double_result)));
}

void CheckOutputData(const std::vector<int32>& test_output,
                     const std::vector<int32>& reference_output,
                     const std::vector<int32>& test_input,
                     const string& check_label, int input_integer_bits,
                     int output_integer_bits, int tolerance) {
  // In the special case of small input, specifically raw value of 5, a rounding
  // up leads to difference in the output.  We do not aim to be accurate for
  // very small input values, and there should be sufficient input fractional
  // bits that this is a small input.
  static constexpr double error_from_rounding_up = 0.0224585;
  const int n = test_output.size();
  ASSERT_EQ(n, reference_output.size());
  for (int i = 0; i < n; ++i) {
    // Adjust tolerance when input <= 5*2^-(31-input_integer_bits).
    const int adjusted_tolerance =
        test_input[i] > 5
            ? tolerance
            : std::max(tolerance, static_cast<int>(std::ceil(
                                      error_from_rounding_up *
                                      (1 << (31 - output_integer_bits)))));
    ASSERT_LE(std::abs(test_output[i] - reference_output[i]),
              adjusted_tolerance)
        << "Failure in \"" << check_label << "\" at i=" << i
        << ", test_input[i]=" << test_input[i] << "="
        << static_cast<double>(test_input[i]) / (1 << (31 - input_integer_bits))
        << ", test_output[i]=" << test_output[i] << "="
        << static_cast<double>(test_output[i]) /
               (1 << (31 - output_integer_bits))
        << ", reference_output[i]=" << reference_output[i] << "="
        << static_cast<double>(reference_output[i]) /
               (1 << (31 - output_integer_bits))
        << ", difference[i]=" << std::abs(reference_output[i] - test_output[i])
        << "="
        << static_cast<double>(std::abs(reference_output[i] - test_output[i])) /
               (1 << (31 - output_integer_bits))
        << "; tolerance=" << tolerance
        << ", adj tolerance=" << adjusted_tolerance;
  }
}

void RightShiftVector(const std::vector<int32>& shifts,
                      std::vector<int32>* vec) {
  const int n = vec->size();
  ASSERT_EQ(n, shifts.size());
  for (int i = 0; i < n; ++i) {
    vec->at(i) = std::max(1, vec->at(i) >> shifts[i]);
  }
}

template <int OutputIntegerBits, int InputIntegerBits>
void RunSingleTest(const std::vector<int32>& test_input,
                   const string& check_label, int tolerance) {
  const int n = test_input.size();
  std::vector<int32> float_gen_output(n, 0);
  std::vector<int32> reference_output(n, 0);
  std::vector<int32> optimized_output(n, 0);

  // Workaround the stupid things that intelligent humans do.
  // Consequence of __builtin_clz(0u) may equal 31 instead of 32.
  std::vector<int32> fudged_input(n, 0);
  for (int i = 0; i < n; ++i) {
    fudged_input[i] = std::max(test_input[i], 2);
  }

  for (int i = 0; i < n; ++i) {
    reference_output[i] =
        tflite::reference_ops::log_x_for_x_greater_than_or_equal_to_1_impl<
            OutputIntegerBits, InputIntegerBits>(
            gemmlowp::FixedPoint<int32, InputIntegerBits>::FromRaw(
                fudged_input[i]))
            .raw();
    optimized_output[i] =
        tflite::optimized_ops::log_x_for_x_greater_than_or_equal_to_1_impl<
            OutputIntegerBits, InputIntegerBits>(
            gemmlowp::FixedPoint<int32, InputIntegerBits>::FromRaw(
                fudged_input[i]))
            .raw();
    float_gen_output[i] = LogPositiveValuesViaFloat(
        fudged_input[i], InputIntegerBits, OutputIntegerBits);
  }
  // Note that first check is intolerant.
  {
    std::ostringstream label;
    label << check_label << " / optimized vs reference / InputIntegerBits="
          << InputIntegerBits << ", OutputIntegerBits=" << OutputIntegerBits;
    CheckOutputData(
        optimized_output, reference_output, test_input, label.str(),
        InputIntegerBits, OutputIntegerBits, 0);
  }
  {
    std::ostringstream label;
    label << check_label << " / reference vs float-gen / InputIntegerBits="
          << InputIntegerBits << ", OutputIntegerBits=" << OutputIntegerBits;
    CheckOutputData(
        reference_output, float_gen_output, test_input, label.str(),
        InputIntegerBits, OutputIntegerBits, tolerance);
  }
  {
    std::ostringstream label;
    label << check_label << " optimized vs float-gen / InputIntegerBits="
          << InputIntegerBits << ", OutputIntegerBits=" << OutputIntegerBits;
    CheckOutputData(
        optimized_output, float_gen_output, test_input, label.str(),
        InputIntegerBits, OutputIntegerBits, tolerance);
  }
}

template <int OutputIntegerBits>
void RunSingleTest(const std::vector<int32>& test_input, int input_integer_bits,
                   const string& check_label, int tolerance) {
#define INPUT_CASE(K)                                                   \
  case K:                                                               \
    return RunSingleTest<OutputIntegerBits, K>(test_input, check_label, \
                                               tolerance)
  switch (input_integer_bits) {
    INPUT_CASE(0);
    INPUT_CASE(1);
    INPUT_CASE(2);
    INPUT_CASE(3);
    INPUT_CASE(4);
    INPUT_CASE(5);
    INPUT_CASE(6);
    INPUT_CASE(7);
    INPUT_CASE(8);
    INPUT_CASE(9);
    INPUT_CASE(10);
    INPUT_CASE(11);
    INPUT_CASE(12);
    INPUT_CASE(13);
    INPUT_CASE(14);
    INPUT_CASE(15);
    INPUT_CASE(16);
    INPUT_CASE(17);
    INPUT_CASE(18);
    INPUT_CASE(19);
    INPUT_CASE(20);
    INPUT_CASE(21);
    INPUT_CASE(22);
    INPUT_CASE(23);
    INPUT_CASE(24);
    INPUT_CASE(25);
    INPUT_CASE(26);
    INPUT_CASE(27);
    INPUT_CASE(28);
    INPUT_CASE(29);
    default:
      ASSERT_LE(input_integer_bits, 30)
                << "Input integer bits not handled: " << input_integer_bits;
  }
#undef INPUT_CASE
}

void RunSingleTest(const std::vector<int32>& test_input, int input_integer_bits,
                   int output_integer_bits, const string& check_label,
                   int tolerance) {
#define OUTPUT_CASE(K)                                                   \
  case K:                                                                \
    return RunSingleTest<K>(test_input, input_integer_bits, check_label, \
                            tolerance)
  switch (output_integer_bits) {
    OUTPUT_CASE(0);
    OUTPUT_CASE(1);
    OUTPUT_CASE(2);
    OUTPUT_CASE(3);
    OUTPUT_CASE(4);
    OUTPUT_CASE(5);
    OUTPUT_CASE(6);
    OUTPUT_CASE(7);
    OUTPUT_CASE(8);
    OUTPUT_CASE(9);
    OUTPUT_CASE(10);
    OUTPUT_CASE(11);
    OUTPUT_CASE(12);
    OUTPUT_CASE(13);
    OUTPUT_CASE(14);
    OUTPUT_CASE(15);
    OUTPUT_CASE(16);
    OUTPUT_CASE(17);
    OUTPUT_CASE(18);
    OUTPUT_CASE(19);
    OUTPUT_CASE(20);
    OUTPUT_CASE(21);
    OUTPUT_CASE(22);
    OUTPUT_CASE(23);
    OUTPUT_CASE(24);
    OUTPUT_CASE(25);
    OUTPUT_CASE(26);
    OUTPUT_CASE(27);
    OUTPUT_CASE(28);
    OUTPUT_CASE(29);
    default:
      ASSERT_LE(input_integer_bits, 30)
                << "Input integer bits not handled: " << input_integer_bits;
  }
#undef OUTPUT_CASE
}

void RunUniformTest(int test_size, int input_integer_bits,
                    int output_integer_bits, const string& check_label,
                    int tolerance, NumberGenerator* generator) {
  std::vector<int> test_data = generator->RandomIntVector(
      test_size, 2, std::numeric_limits<int>::max() - 1);
  test_data[0] = 2;
  test_data[1] = 3;
  test_data[2] = 4;
  test_data[3] = std::numeric_limits<int32>::max() - 2;
  test_data[4] = std::numeric_limits<int32>::max() - 1;
  test_data[5] = std::numeric_limits<int32>::max();

  RunSingleTest(test_data, input_integer_bits, output_integer_bits,
                check_label + " / uniform test", tolerance);
}

void RunUniformShiftUniformTest(int test_size, int input_integer_bits,
                                int output_integer_bits,
                                const string& check_label, int tolerance,
                                NumberGenerator* generator) {
  std::vector<int> test_data = generator->RandomIntVector(
      test_size, 2, std::numeric_limits<int>::max() - 1);
  std::vector<int> shifts = generator->RandomIntVector(test_size, 0, 29);
  RightShiftVector(shifts, &test_data);

  RunSingleTest(test_data, input_integer_bits, output_integer_bits,
                check_label + " / shifted test", tolerance);
}

TEST_F(LogQuantizedTest, VariedIntegerBits) {
  static constexpr int kVariations = 250;
  static constexpr int kRunSize = 250;
  static constexpr int kIntegerTolerance = 8;
  static constexpr double kOutputFloatTolerance = 7.0e-7;

  std::vector<int> input_integer_bits =
      generator_.RandomIntVector(kVariations, 0, 24);
  std::vector<int> output_integer_bits =
      generator_.RandomIntVector(kVariations, 1, 10);

  for (int i = 0; i < kVariations; ++i) {
    int var_output_integer_bits = output_integer_bits[i];
    int tolerance =
        std::max(1.0 * kIntegerTolerance,
                 (1 << (31 - var_output_integer_bits)) * kOutputFloatTolerance);

    RunUniformTest(kRunSize, input_integer_bits[i], var_output_integer_bits,
                   "VariedIntegerBits", tolerance, &generator_);
    RunUniformShiftUniformTest(kRunSize, input_integer_bits[i],
                               var_output_integer_bits, "VariedIntegerBits",
                               tolerance, &generator_);
  }
}

TEST_F(LogQuantizedTest, SelectedIntegerBits) {
  static constexpr int kInputBits = 12;
  static constexpr int kOutputBits = 5;
  static constexpr int kRunSize = 100000;
  static constexpr int kIntegerTolerance = 4;

  RunUniformTest(kRunSize, kInputBits, kOutputBits, "SelectedIntegerBits",
                 kIntegerTolerance, &generator_);
  RunUniformShiftUniformTest(kRunSize, kInputBits, kOutputBits,
                             "SelectedIntegerBits", kIntegerTolerance,
                             &generator_);
}

}  // namespace
