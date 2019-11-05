/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/micro/memory_planner/greedy_memory_planner.h"

#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

namespace tflite {
// We don't declare this in the header since it's not a public interface, but we
// need to call it to test it, so declare it here instead.
void ReverseSortInPlace(int* values, int* ids, int size);
}  // namespace tflite

namespace {
constexpr int kScratchBufferSize = 4096;
unsigned char g_scratch_buffer[kScratchBufferSize];
}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestReverseSortInPlace) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  constexpr int a_size = 10;
  int a_values[a_size] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  int a_ids[a_size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const int a_expected_values[a_size] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  const int a_expected_ids[a_size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  tflite::ReverseSortInPlace(a_values, a_ids, a_size);
  for (int i = 0; i < a_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(a_expected_values[i], a_values[i]);
    TF_LITE_MICRO_EXPECT_EQ(a_expected_ids[i], a_ids[i]);
  }

  constexpr int b_size = 10;
  int b_values[b_size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int b_ids[b_size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const int b_expected_values[b_size] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  const int b_expected_ids[b_size] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
  tflite::ReverseSortInPlace(b_values, b_ids, b_size);
  for (int i = 0; i < b_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(b_expected_values[i], b_values[i]);
    TF_LITE_MICRO_EXPECT_EQ(b_expected_ids[i], b_ids[i]);
  }

  constexpr int c_size = 100;
  int c_values[c_size] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int c_ids[c_size] = {
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
      17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
      34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
      51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
      68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
      85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};
  const int c_expected_values[c_size] = {
      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
      8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const int c_expected_ids[c_size] = {
      9,  19, 29, 39, 49, 59, 69, 79, 89, 99, 8,  18, 28, 38, 48, 58, 68,
      78, 88, 98, 7,  17, 27, 37, 47, 57, 67, 77, 87, 97, 6,  16, 26, 36,
      46, 56, 66, 76, 86, 96, 5,  15, 25, 35, 45, 55, 65, 75, 85, 95, 4,
      14, 24, 34, 44, 54, 64, 74, 84, 94, 3,  13, 23, 33, 43, 53, 63, 73,
      83, 93, 2,  12, 22, 32, 42, 52, 62, 72, 82, 92, 1,  11, 21, 31, 41,
      51, 61, 71, 81, 91, 0,  10, 20, 30, 40, 50, 60, 70, 80, 90};
  tflite::ReverseSortInPlace(c_values, c_ids, c_size);
  for (int i = 0; i < c_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(c_expected_values[i], c_values[i]);
    TF_LITE_MICRO_EXPECT_EQ(c_expected_ids[i], c_ids[i]);
  }
}

TF_LITE_MICRO_TEST(TestGreedyBasics) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  tflite::GreedyMemoryPlanner planner(g_scratch_buffer, kScratchBufferSize);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 10, 0, 1));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 20, 2, 3));

  TF_LITE_MICRO_EXPECT_EQ(false, planner.DoAnyBuffersOverlap(error_reporter));

  TF_LITE_MICRO_EXPECT_EQ(20, planner.GetMaximumMemorySize());

  int offset = -1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(error_reporter, 0, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(error_reporter, 1, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);
}

TF_LITE_MICRO_TEST(TestGreedyMedium) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  tflite::GreedyMemoryPlanner planner(g_scratch_buffer, kScratchBufferSize);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 10, 0, 1));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 20, 1, 2));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 30, 2, 3));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 40, 3, 4));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 50, 0, 1));

  int offset = -1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(error_reporter, 0, &offset));
  TF_LITE_MICRO_EXPECT_EQ(50, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(error_reporter, 1, &offset));
  TF_LITE_MICRO_EXPECT_EQ(70, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(error_reporter, 2, &offset));
  TF_LITE_MICRO_EXPECT_EQ(40, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(error_reporter, 3, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, planner.GetOffsetForBuffer(error_reporter, 4, &offset));
  TF_LITE_MICRO_EXPECT_EQ(0, offset);

  planner.PrintMemoryPlan(error_reporter);

  TF_LITE_MICRO_EXPECT_EQ(false, planner.DoAnyBuffersOverlap(error_reporter));

  TF_LITE_MICRO_EXPECT_EQ(90, planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TEST(TestPersonDetectionModel) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  tflite::GreedyMemoryPlanner planner(g_scratch_buffer, kScratchBufferSize);
  // These buffer sizes and time ranges are taken from the 250KB MobileNet model
  // used in the person detection example.
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 9216, 0, 29));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 3, 28, 29));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 256, 27, 28));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 2304, 26, 27));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 2304, 25, 26));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 2304, 24, 25));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 1152, 23, 24));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 4608, 22, 23));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 4608, 21, 22));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 4608, 20, 21));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 4608, 19, 20));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 4608, 18, 19));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 4608, 17, 18));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 4608, 16, 17));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 4608, 15, 16));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 4608, 14, 15));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 4608, 13, 14));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 4608, 12, 13));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 2304, 11, 12));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 9216, 10, 11));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 9216, 9, 10));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 9216, 8, 9));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 4608, 7, 8));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 18432, 6, 7));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 18432, 5, 6));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 18432, 4, 5));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 9216, 3, 4));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 36864, 2, 3));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 18432, 1, 2));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 18432, 0, 1));

  planner.PrintMemoryPlan(error_reporter);

  TF_LITE_MICRO_EXPECT_EQ(false, planner.DoAnyBuffersOverlap(error_reporter));

  // The sum of all the buffers is 241,027 bytes, so we at least expect the plan
  // to come up with something smaller than this.
  TF_LITE_MICRO_EXPECT_GT(241027, planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TEST(TestOverlapCase) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  tflite::GreedyMemoryPlanner planner(g_scratch_buffer, kScratchBufferSize);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 100, 0, 1));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 50, 2, 3));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 20, 1, 2));

  planner.PrintMemoryPlan(error_reporter);

  TF_LITE_MICRO_EXPECT_EQ(false, planner.DoAnyBuffersOverlap(error_reporter));

  TF_LITE_MICRO_EXPECT_EQ(120, planner.GetMaximumMemorySize());
}

TF_LITE_MICRO_TEST(TestSmallScratch) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  constexpr int scratch_buffer_size = 40;
  unsigned char scratch_buffer[scratch_buffer_size];
  tflite::GreedyMemoryPlanner planner(scratch_buffer, scratch_buffer_size);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          planner.AddBuffer(error_reporter, 100, 0, 1));
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          planner.AddBuffer(error_reporter, 50, 2, 3));
}

TF_LITE_MICRO_TESTS_END
