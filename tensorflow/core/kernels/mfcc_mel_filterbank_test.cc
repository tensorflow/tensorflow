/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/mfcc_mel_filterbank.h"

#include <limits>
#include <vector>

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

TEST(MfccMelFilterbankTest, AgreesWithPythonGoldenValues) {
  // This test verifies the Mel filterbank against "golden values".
  // Golden values are from an independent Python Mel implementation.
  MfccMelFilterbank filterbank;

  std::vector<double> input;
  const int kSampleCount = 513;
  input.reserve(kSampleCount);
  for (int i = 0; i < kSampleCount; ++i) {
    input.push_back(i + 1);
  }
  const int kChannelCount = 20;
  filterbank.Initialize(
      input.size(), 22050 /* sample rate */, kChannelCount /* channels */,
      20.0 /*  lower frequency limit */, 4000.0 /* upper frequency limit */);

  std::vector<double> output;
  filterbank.Compute(input, &output);

  std::vector<double> expected = {
      7.38894574,   10.30330648, 13.72703292,  17.24158686,  21.35253118,
      25.77781089,  31.30624108, 37.05877236,  43.9436536,   51.80306637,
      60.79867148,  71.14363376, 82.90910141,  96.50069158,  112.08428368,
      129.96721968, 150.4277597, 173.74997634, 200.86037462, 231.59802942};

  ASSERT_EQ(output.size(), kChannelCount);

  for (int i = 0; i < kChannelCount; ++i) {
    EXPECT_NEAR(output[i], expected[i], 1e-04);
  }
}

TEST(MfccMelFilterbankTest, IgnoresExistingContentOfOutputVector) {
  // Test for bug where the output vector was not cleared before
  // accumulating next frame's weighted spectral values.
  MfccMelFilterbank filterbank;

  const int kSampleCount = 513;
  std::vector<double> input;
  std::vector<double> output;

  filterbank.Initialize(kSampleCount, 22050 /* sample rate */,
                        20 /* channels */, 20.0 /*  lower frequency limit */,
                        4000.0 /* upper frequency limit */);

  // First call with nonzero input value, and an empty output vector,
  // will resize the output and fill it with the correct, nonzero outputs.
  input.assign(kSampleCount, 1.0);
  filterbank.Compute(input, &output);
  for (const double value : output) {
    EXPECT_LE(0.0, value);
  }

  // Second call with zero input should also generate zero output.  However,
  // the output vector now is already the correct size, but full of nonzero
  // values.  Make sure these don't affect the output.
  input.assign(kSampleCount, 0.0);
  filterbank.Compute(input, &output);
  for (const double value : output) {
    EXPECT_EQ(0.0, value);
  }
}

TEST(MfccMelFilterbankTest, FailsWhenChannelsGreaterThanMaxIntValue) {
  // Test for bug where vector throws a length_error when it suspects the size
  // to be more than it's max_size. For now, we fail initialization when the
  // number of requested channels is >= the maximum value int can take (since
  // num_channels_ is an int).
  MfccMelFilterbank filterbank;

  const int kSampleCount = 513;
  std::size_t num_channels = std::numeric_limits<int>::max();
  bool initialized = filterbank.Initialize(
      kSampleCount, 2 /* sample rate */, num_channels /* channels */,
      1.0 /*  lower frequency limit */, 5.0 /* upper frequency limit */);

  EXPECT_FALSE(initialized);
}

TEST(MfccMelFilterbankTest, FailsWhenChannelsGreaterThanMaxSize) {
  // Test for bug where vector throws a length_error when it suspects the size
  // to be more than it's max_size. For now, we fail initialization when the
  // number of requested channels is > than std::vector<double>::max_size().
  MfccMelFilterbank filterbank;

  const int kSampleCount = 513;
  // Set num_channels to exceed the max_size a double vector can
  // theoretically take.
  std::size_t num_channels = std::vector<double>().max_size() + 1;
  bool initialized = filterbank.Initialize(
      kSampleCount, 2 /* sample rate */, num_channels /* channels */,
      1.0 /*  lower frequency limit */, 5.0 /* upper frequency limit */);

  EXPECT_FALSE(initialized);
}

}  // namespace tensorflow
