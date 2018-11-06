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
#include "tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control.h"
#include "tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

const int kNumChannels = 2;
const int kSmoothingBits = 10;
const int kCorrectionBits = -1;

// Test pcan auto gain control using default config values.
class PcanGainControlTest : public ::testing::Test {
 protected:
  PcanGainControlTest() {
    config_.enable_pcan = 1;
    config_.strength = 0.95;
    config_.offset = 80.0;
    config_.gain_bits = 21;
  }

  struct PcanGainControlConfig config_;
};

TEST_F(PcanGainControlTest, TestPcanGainControl) {
  uint32_t estimate[] = {6321887, 31248341};
  struct PcanGainControlState state;
  ASSERT_TRUE(PcanGainControlPopulateState(&config_, &state, estimate,
                                           kNumChannels, kSmoothingBits,
                                           kCorrectionBits));

  uint32_t signal[] = {241137, 478104};
  PcanGainControlApply(&state, signal);

  const uint32_t expected[] = {3578, 1533};
  ASSERT_EQ(state.num_channels, sizeof(expected) / sizeof(expected[0]));
  for (int i = 0; i < state.num_channels; ++i) {
    EXPECT_EQ(signal[i], expected[i]);
  }

  PcanGainControlFreeStateContents(&state);
}

}  // namespace
