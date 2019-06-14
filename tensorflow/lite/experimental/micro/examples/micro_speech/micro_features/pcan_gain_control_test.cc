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
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control_util.h"

#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

namespace {

const int kNumChannels = 2;
const int kSmoothingBits = 10;
const int kCorrectionBits = -1;

// Test pcan auto gain control using default config values.
class PcanGainControlTestConfig {
 public:
  PcanGainControlTestConfig() {
    config_.enable_pcan = 1;
    config_.strength = 0.95;
    config_.offset = 80.0;
    config_.gain_bits = 21;
  }

  struct PcanGainControlConfig config_;
};

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(PcanGainControlTest_TestPcanGainControl) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  uint32_t estimate[] = {6321887, 31248341};
  PcanGainControlTestConfig config;
  struct PcanGainControlState state;
  TF_LITE_MICRO_EXPECT(PcanGainControlPopulateState(
      error_reporter, &config.config_, &state, estimate, kNumChannels,
      kSmoothingBits, kCorrectionBits));

  uint32_t signal[] = {241137, 478104};
  PcanGainControlApply(&state, signal);

  const uint32_t expected[] = {3578, 1533};
  TF_LITE_MICRO_EXPECT_EQ(state.num_channels,
                          sizeof(expected) / sizeof(expected[0]));
  int i;
  for (i = 0; i < state.num_channels; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(signal[i], expected[i]);
  }
}

TF_LITE_MICRO_TESTS_END
