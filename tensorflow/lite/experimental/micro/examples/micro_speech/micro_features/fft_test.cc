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
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft_util.h"

#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

namespace {

const int16_t kFakeWindow[] = {
    0, 1151,   0, -5944, 0, 13311,  0, -21448, 0, 28327, 0, -32256, 0, 32255,
    0, -28328, 0, 21447, 0, -13312, 0, 5943,   0, -1152, 0};
const int kScaleShift = 0;

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FftTest_CheckOutputValues) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  struct FftState state;
  TF_LITE_MICRO_EXPECT(FftPopulateState(
      error_reporter, &state, sizeof(kFakeWindow) / sizeof(kFakeWindow[0])));

  FftInit(&state);
  FftCompute(&state, kFakeWindow, kScaleShift);

  const struct complex_int16_t expected[] = {
      {0, 0},    {-10, 9},     {-20, 0},   {-9, -10},     {0, 25},  {-119, 119},
      {-887, 0}, {3000, 3000}, {0, -6401}, {-3000, 3000}, {886, 0}, {118, 119},
      {0, 25},   {9, -10},     {19, 0},    {9, 9},        {0, 0}};
  TF_LITE_MICRO_EXPECT_EQ(state.fft_size / 2 + 1,
                          sizeof(expected) / sizeof(expected[0]));
  int i;
  for (i = 0; i <= state.fft_size / 2; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(state.output[i].real, expected[i].real, 2);
    TF_LITE_MICRO_EXPECT_NEAR(state.output[i].imag, expected[i].imag, 2);
  }
}

TF_LITE_MICRO_TESTS_END
