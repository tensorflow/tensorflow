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
      16.94595387,  30.27006983,   48.21933112,   69.8960073,   97.63951250,
      130.8914936,  174.69920165,  225.46272677,  289.4927467,  367.74647923,
      463.2220240,  579.80039333,  720.72132977,  892.6426393,  1101.06436795,
      1353.5724593, 1658.46575957, 2025.09242587, 2472.1304465, 3007.09568116};

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

}  // namespace tensorflow
