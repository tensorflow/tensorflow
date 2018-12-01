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
#include "tensorflow/lite/experimental/microfrontend/lib/filterbank.h"
#include "tensorflow/lite/experimental/microfrontend/lib/filterbank_util.h"

#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

const int kSampleRate = 1000;
const int kSpectrumSize = 17;
const int kStartIndex = 1;
const int kEndIndex = 15;
const int32_t kEnergy[] = {-1,     181,      400,      181,      625,    28322,
                           786769, 18000000, 40972801, 18000000, 784996, 28085,
                           625,    181,      361,      -1,       -1};
const uint64_t kWork[] = {1835887, 61162970173, 258694800000};
const int kScaleShift = 0;

// Test filterbank generation using scaled-down defaults.
class FilterbankTest : public ::testing::Test {
 protected:
  FilterbankTest() {
    config_.num_channels = 2;
    config_.lower_band_limit = 8.0;
    config_.upper_band_limit = 450.0;
  }

  struct FilterbankConfig config_;
};

TEST_F(FilterbankTest, CheckStartIndex) {
  struct FilterbankState state;
  ASSERT_TRUE(
      FilterbankPopulateState(&config_, &state, kSampleRate, kSpectrumSize));

  EXPECT_EQ(state.start_index, kStartIndex);

  FilterbankFreeStateContents(&state);
}

TEST_F(FilterbankTest, CheckEndIndex) {
  struct FilterbankState state;
  ASSERT_TRUE(
      FilterbankPopulateState(&config_, &state, kSampleRate, kSpectrumSize));

  EXPECT_EQ(state.end_index, kEndIndex);

  FilterbankFreeStateContents(&state);
}

TEST_F(FilterbankTest, CheckChannelFrequencyStarts) {
  struct FilterbankState state;
  ASSERT_TRUE(
      FilterbankPopulateState(&config_, &state, kSampleRate, kSpectrumSize));

  const int16_t expected[] = {0, 4, 8};
  ASSERT_EQ(state.num_channels + 1, sizeof(expected) / sizeof(expected[0]));
  for (int i = 0; i <= state.num_channels; ++i) {
    EXPECT_EQ(state.channel_frequency_starts[i], expected[i]);
  }

  FilterbankFreeStateContents(&state);
}

TEST_F(FilterbankTest, CheckChannelWeightStarts) {
  struct FilterbankState state;
  ASSERT_TRUE(
      FilterbankPopulateState(&config_, &state, kSampleRate, kSpectrumSize));

  const int16_t expected[] = {0, 8, 16};
  ASSERT_EQ(state.num_channels + 1, sizeof(expected) / sizeof(expected[0]));
  for (int i = 0; i <= state.num_channels; ++i) {
    EXPECT_EQ(state.channel_weight_starts[i], expected[i]);
  }

  FilterbankFreeStateContents(&state);
}

TEST_F(FilterbankTest, CheckChannelWidths) {
  struct FilterbankState state;
  ASSERT_TRUE(
      FilterbankPopulateState(&config_, &state, kSampleRate, kSpectrumSize));

  const int16_t expected[] = {8, 8, 8};
  ASSERT_EQ(state.num_channels + 1, sizeof(expected) / sizeof(expected[0]));
  for (int i = 0; i <= state.num_channels; ++i) {
    EXPECT_EQ(state.channel_widths[i], expected[i]);
  }

  FilterbankFreeStateContents(&state);
}

TEST_F(FilterbankTest, CheckWeights) {
  struct FilterbankState state;
  ASSERT_TRUE(
      FilterbankPopulateState(&config_, &state, kSampleRate, kSpectrumSize));

  const int16_t expected[] = {0, 3277, 2217, 1200, 222,  0,   0,   0,
                              0, 3376, 2468, 1591, 744,  0,   0,   0,
                              0, 4020, 3226, 2456, 1708, 983, 277, 0};
  ASSERT_EQ(state.channel_weight_starts[state.num_channels] +
                state.channel_widths[state.num_channels],
            sizeof(expected) / sizeof(expected[0]));
  for (int i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
    EXPECT_EQ(state.weights[i], expected[i]);
  }

  FilterbankFreeStateContents(&state);
}

TEST_F(FilterbankTest, CheckUnweights) {
  struct FilterbankState state;
  ASSERT_TRUE(
      FilterbankPopulateState(&config_, &state, kSampleRate, kSpectrumSize));

  const int16_t expected[] = {0, 819, 1879, 2896, 3874, 0,    0,    0,
                              0, 720, 1628, 2505, 3352, 0,    0,    0,
                              0, 76,  870,  1640, 2388, 3113, 3819, 0};
  ASSERT_EQ(state.channel_weight_starts[state.num_channels] +
                state.channel_widths[state.num_channels],
            sizeof(expected) / sizeof(expected[0]));
  for (int i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
    EXPECT_EQ(state.unweights[i], expected[i]);
  }

  FilterbankFreeStateContents(&state);
}

TEST_F(FilterbankTest, CheckConvertFftComplexToEnergy) {
  struct FilterbankState state;
  state.start_index = kStartIndex;
  state.end_index = kEndIndex;

  struct complex_int16_t fake_fft[] = {
      {0, 0},    {-10, 9},     {-20, 0},   {-9, -10},     {0, 25},  {-119, 119},
      {-887, 0}, {3000, 3000}, {0, -6401}, {-3000, 3000}, {886, 0}, {118, 119},
      {0, 25},   {9, -10},     {19, 0},    {9, 9},        {0, 0}};
  int32_t* energy = reinterpret_cast<int32_t*>(fake_fft);
  FilterbankConvertFftComplexToEnergy(&state, fake_fft, energy);

  for (int i = state.start_index; i < state.end_index; ++i) {
    EXPECT_EQ(energy[i], kEnergy[i]);
  }
}

TEST_F(FilterbankTest, CheckAccumulateChannels) {
  struct FilterbankState state;
  ASSERT_TRUE(
      FilterbankPopulateState(&config_, &state, kSampleRate, kSpectrumSize));

  FilterbankAccumulateChannels(&state, kEnergy);

  ASSERT_EQ(state.num_channels + 1, sizeof(kWork) / sizeof(kWork[0]));
  for (int i = 0; i <= state.num_channels; ++i) {
    EXPECT_EQ(state.work[i], kWork[i]);
  }

  FilterbankFreeStateContents(&state);
}

TEST_F(FilterbankTest, CheckSqrt) {
  struct FilterbankState state;
  ASSERT_TRUE(
      FilterbankPopulateState(&config_, &state, kSampleRate, kSpectrumSize));
  std::memcpy(state.work, kWork, sizeof(kWork));

  uint32_t* scaled_filterbank = FilterbankSqrt(&state, kScaleShift);

  const uint32_t expected[] = {247311, 508620};
  ASSERT_EQ(state.num_channels, sizeof(expected) / sizeof(expected[0]));
  for (int i = 0; i < state.num_channels; ++i) {
    EXPECT_EQ(scaled_filterbank[i], expected[i]);
  }

  FilterbankFreeStateContents(&state);
}

}  // namespace
