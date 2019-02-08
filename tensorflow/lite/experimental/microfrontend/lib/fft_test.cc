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
#include "tensorflow/lite/experimental/microfrontend/lib/fft.h"
#include "tensorflow/lite/experimental/microfrontend/lib/fft_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

const int16_t kFakeWindow[] = {
    0, 1151,   0, -5944, 0, 13311,  0, -21448, 0, 28327, 0, -32256, 0, 32255,
    0, -28328, 0, 21447, 0, -13312, 0, 5943,   0, -1152, 0};
const int kScaleShift = 0;

TEST(FftTest, CheckOutputValues) {
  struct FftState state;
  ASSERT_TRUE(
      FftPopulateState(&state, sizeof(kFakeWindow) / sizeof(kFakeWindow[0])));

  FftInit(&state);
  FftCompute(&state, kFakeWindow, kScaleShift);

  const struct complex_int16_t expected[] = {
      {0, 0},    {-10, 9},     {-20, 0},   {-9, -10},     {0, 25},  {-119, 119},
      {-887, 0}, {3000, 3000}, {0, -6401}, {-3000, 3000}, {886, 0}, {118, 119},
      {0, 25},   {9, -10},     {19, 0},    {9, 9},        {0, 0}};
  ASSERT_EQ(state.fft_size / 2 + 1, sizeof(expected) / sizeof(expected[0]));
  int i;
  for (i = 0; i <= state.fft_size / 2; ++i) {
    EXPECT_EQ(state.output[i].real, expected[i].real);
    EXPECT_EQ(state.output[i].imag, expected[i].imag);
  }

  FftFreeStateContents(&state);
}

}  // namespace
