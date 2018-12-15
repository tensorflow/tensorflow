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
#include "tensorflow/lite/experimental/microfrontend/lib/noise_reduction.h"
#include "tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

const int kNumChannels = 2;

// Test noise reduction using default config values.
class NoiseReductionTest : public ::testing::Test {
 protected:
  NoiseReductionTest() {
    config_.smoothing_bits = 10;
    config_.even_smoothing = 0.025;
    config_.odd_smoothing = 0.06;
    config_.min_signal_remaining = 0.05;
  }

  struct NoiseReductionConfig config_;
};

TEST_F(NoiseReductionTest, TestNoiseReductionEstimate) {
  struct NoiseReductionState state;
  ASSERT_TRUE(NoiseReductionPopulateState(&config_, &state, kNumChannels));

  uint32_t signal[] = {247311, 508620};
  NoiseReductionApply(&state, signal);

  const uint32_t expected[] = {6321887, 31248341};
  ASSERT_EQ(state.num_channels, sizeof(expected) / sizeof(expected[0]));
  int i;
  for (i = 0; i < state.num_channels; ++i) {
    EXPECT_EQ(state.estimate[i], expected[i]);
  }

  NoiseReductionFreeStateContents(&state);
}

TEST_F(NoiseReductionTest, TestNoiseReduction) {
  struct NoiseReductionState state;
  ASSERT_TRUE(NoiseReductionPopulateState(&config_, &state, kNumChannels));

  uint32_t signal[] = {247311, 508620};
  NoiseReductionApply(&state, signal);

  const uint32_t expected[] = {241137, 478104};
  ASSERT_EQ(state.num_channels, sizeof(expected) / sizeof(expected[0]));
  int i;
  for (i = 0; i < state.num_channels; ++i) {
    EXPECT_EQ(signal[i], expected[i]);
  }

  NoiseReductionFreeStateContents(&state);
}

}  // namespace
