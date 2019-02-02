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
#include "tensorflow/lite/experimental/microfrontend/lib/window.h"
#include "tensorflow/lite/experimental/microfrontend/lib/window_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

const int kSampleRate = 1000;
const int kWindowSamples = 25;
const int kStepSamples = 10;
const int16_t kFakeAudioData[] = {
    0, 32767, 0, -32768, 0, 32767, 0, -32768, 0, 32767, 0, -32768,
    0, 32767, 0, -32768, 0, 32767, 0, -32768, 0, 32767, 0, -32768,
    0, 32767, 0, -32768, 0, 32767, 0, -32768, 0, 32767, 0, -32768};

// Test window function behaviors using default config values.
class WindowTest : public ::testing::Test {
 protected:
  WindowTest() {
    config_.size_ms = 25;
    config_.step_size_ms = 10;
  }

  struct WindowConfig config_;
};

TEST_F(WindowTest, CheckCoefficients) {
  struct WindowState state;
  ASSERT_TRUE(WindowPopulateState(&config_, &state, kSampleRate));

  const int16_t expected[] = {16,   144,  391,  743,  1176, 1664, 2177,
                              2681, 3145, 3541, 3843, 4032, 4096, 4032,
                              3843, 3541, 3145, 2681, 2177, 1664, 1176,
                              743,  391,  144,  16};
  ASSERT_EQ(state.size, sizeof(expected) / sizeof(expected[0]));
  int i;
  for (i = 0; i < state.size; ++i) {
    EXPECT_EQ(state.coefficients[i], expected[i]);
  }

  WindowFreeStateContents(&state);
}

TEST_F(WindowTest, CheckResidualInput) {
  struct WindowState state;
  ASSERT_TRUE(WindowPopulateState(&config_, &state, kSampleRate));
  size_t num_samples_read;

  ASSERT_TRUE(WindowProcessSamples(
      &state, kFakeAudioData,
      sizeof(kFakeAudioData) / sizeof(kFakeAudioData[0]), &num_samples_read));

  int i;
  for (i = kStepSamples; i < kWindowSamples; ++i) {
    EXPECT_EQ(state.input[i - kStepSamples], kFakeAudioData[i]);
  }

  WindowFreeStateContents(&state);
}

TEST_F(WindowTest, CheckOutputValues) {
  struct WindowState state;
  ASSERT_TRUE(WindowPopulateState(&config_, &state, kSampleRate));
  size_t num_samples_read;

  ASSERT_TRUE(WindowProcessSamples(
      &state, kFakeAudioData,
      sizeof(kFakeAudioData) / sizeof(kFakeAudioData[0]), &num_samples_read));

  const int16_t expected[] = {
      0, 1151,   0, -5944, 0, 13311,  0, -21448, 0, 28327, 0, -32256, 0, 32255,
      0, -28328, 0, 21447, 0, -13312, 0, 5943,   0, -1152, 0};
  ASSERT_EQ(state.size, sizeof(expected) / sizeof(expected[0]));
  int i;
  for (i = 0; i < state.size; ++i) {
    EXPECT_EQ(state.output[i], expected[i]);
  }

  WindowFreeStateContents(&state);
}

TEST_F(WindowTest, CheckMaxAbsValue) {
  struct WindowState state;
  ASSERT_TRUE(WindowPopulateState(&config_, &state, kSampleRate));
  size_t num_samples_read;

  ASSERT_TRUE(WindowProcessSamples(
      &state, kFakeAudioData,
      sizeof(kFakeAudioData) / sizeof(kFakeAudioData[0]), &num_samples_read));

  EXPECT_EQ(state.max_abs_output_value, 32256);

  WindowFreeStateContents(&state);
}

TEST_F(WindowTest, CheckConsecutiveWindow) {
  struct WindowState state;
  ASSERT_TRUE(WindowPopulateState(&config_, &state, kSampleRate));
  size_t num_samples_read;

  ASSERT_TRUE(WindowProcessSamples(
      &state, kFakeAudioData,
      sizeof(kFakeAudioData) / sizeof(kFakeAudioData[0]), &num_samples_read));
  ASSERT_TRUE(WindowProcessSamples(
      &state, kFakeAudioData + kWindowSamples,
      sizeof(kFakeAudioData) / sizeof(kFakeAudioData[0]) - kWindowSamples,
      &num_samples_read));

  const int16_t expected[] = {
      0, -1152, 0, 5943,   0, -13312, 0, 21447, 0, -28328, 0, 32255, 0, -32256,
      0, 28327, 0, -21448, 0, 13311,  0, -5944, 0, 1151,   0};
  ASSERT_EQ(state.size, sizeof(expected) / sizeof(expected[0]));
  int i;
  for (i = 0; i < state.size; ++i) {
    EXPECT_EQ(state.output[i], expected[i]);
  }

  WindowFreeStateContents(&state);
}

TEST_F(WindowTest, CheckNotEnoughSamples) {
  struct WindowState state;
  ASSERT_TRUE(WindowPopulateState(&config_, &state, kSampleRate));
  size_t num_samples_read;

  ASSERT_TRUE(WindowProcessSamples(
      &state, kFakeAudioData,
      sizeof(kFakeAudioData) / sizeof(kFakeAudioData[0]), &num_samples_read));
  ASSERT_TRUE(WindowProcessSamples(
      &state, kFakeAudioData + kWindowSamples,
      sizeof(kFakeAudioData) / sizeof(kFakeAudioData[0]) - kWindowSamples,
      &num_samples_read));
  ASSERT_FALSE(WindowProcessSamples(
      &state, kFakeAudioData + kWindowSamples + kStepSamples,
      sizeof(kFakeAudioData) / sizeof(kFakeAudioData[0]) - kWindowSamples -
          kStepSamples,
      &num_samples_read));

  EXPECT_EQ(
      state.input_used,
      sizeof(kFakeAudioData) / sizeof(kFakeAudioData[0]) - 2 * kStepSamples);

  WindowFreeStateContents(&state);
}

}  // namespace
