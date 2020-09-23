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

#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_features_generator.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/no_feature_data_slice.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/yes_feature_data_slice.h"
#include "tensorflow/lite/micro/examples/micro_speech/no_30ms_sample_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/yes_30ms_sample_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

// This is a test-only API, not exposed in any public headers, so declare it.
void SetMicroFeaturesNoiseEstimates(const uint32_t* estimate_presets);

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestMicroFeaturesGeneratorYes) {
  tflite::MicroErrorReporter micro_error_reporter;

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          InitializeMicroFeatures(&micro_error_reporter));

  // The micro features pipeline retains state from previous calls to help
  // estimate the background noise. Unfortunately this makes it harder to
  // exactly reproduce results in a test environment, so use a known snapshot
  // of the parameters at the point that the golden feature values were
  // created.
  const uint32_t yes_estimate_presets[] = {
      1062898, 2644477, 1257642, 1864718, 412722, 725703, 395721, 474082,
      173046,  255856,  158966,  153736,  69181,  199100, 144493, 227740,
      110573,  164330,  79666,   144650,  122947, 476799, 398553, 497493,
      322152,  1140005, 566716,  690605,  308902, 347481, 109891, 170457,
      73901,   100975,  42963,   72325,   34183,  20207,  6640,   9468,
  };
  SetMicroFeaturesNoiseEstimates(yes_estimate_presets);

  int8_t yes_calculated_data[g_yes_feature_data_slice_size];
  size_t num_samples_read;
  TfLiteStatus yes_status = GenerateMicroFeatures(
      &micro_error_reporter, g_yes_30ms_sample_data,
      g_yes_30ms_sample_data_size, g_yes_feature_data_slice_size,
      yes_calculated_data, &num_samples_read);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, yes_status);

  for (int i = 0; i < g_yes_feature_data_slice_size; ++i) {
    const int expected = g_yes_feature_data_slice[i];
    const int actual = yes_calculated_data[i];
    TF_LITE_MICRO_EXPECT_EQ(expected, actual);
    if (expected != actual) {
      TF_LITE_REPORT_ERROR(&micro_error_reporter,
                           "Expected value %d but found %d", expected, actual);
    }
  }
}

TF_LITE_MICRO_TEST(TestMicroFeaturesGeneratorNo) {
  tflite::MicroErrorReporter micro_error_reporter;

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk,
                          InitializeMicroFeatures(&micro_error_reporter));
  // As we did for the previous features, set known good noise state
  // parameters.
  const uint32_t no_estimate_presets[] = {
      2563964, 1909393, 559801, 538670, 203643, 175959, 75088, 139491,
      59691,   95307,   43865,  129263, 52517,  80058,  51330, 100731,
      76674,   76262,   15497,  22598,  13778,  21460,  8946,  17806,
      10023,   18810,   8002,   10842,  7578,   9983,   6267,  10759,
      8946,    18488,   9691,   39785,  9939,   17835,  9671,  18512,
  };
  SetMicroFeaturesNoiseEstimates(no_estimate_presets);

  int8_t no_calculated_data[g_no_feature_data_slice_size];
  size_t num_samples_read;
  TfLiteStatus no_status = GenerateMicroFeatures(
      &micro_error_reporter, g_no_30ms_sample_data, g_no_30ms_sample_data_size,
      g_no_feature_data_slice_size, no_calculated_data, &num_samples_read);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, no_status);

  for (size_t i = 0; i < g_no_feature_data_slice_size; ++i) {
    const int expected = g_no_feature_data_slice[i];
    const int actual = no_calculated_data[i];
    TF_LITE_MICRO_EXPECT_EQ(expected, actual);
    if (expected != actual) {
      TF_LITE_REPORT_ERROR(&micro_error_reporter,
                           "Expected value %d but found %d", expected, actual);
    }
  }
}

TF_LITE_MICRO_TESTS_END
