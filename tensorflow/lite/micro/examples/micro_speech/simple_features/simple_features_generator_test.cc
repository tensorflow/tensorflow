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

#include "tensorflow/lite/micro/examples/micro_speech/simple_features/simple_features_generator.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/micro_speech/no_30ms_sample_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/simple_features/no_power_spectrum_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/simple_features/yes_power_spectrum_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/yes_30ms_sample_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestSimpleFeaturesGenerator) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  uint8_t yes_calculated_data[g_yes_power_spectrum_data_size];
  TfLiteStatus yes_status = GenerateSimpleFeatures(
      error_reporter, g_yes_30ms_sample_data, g_yes_30ms_sample_data_size,
      g_yes_power_spectrum_data_size, yes_calculated_data);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, yes_status);

  for (int i = 0; i < g_yes_power_spectrum_data_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(g_yes_power_spectrum_data[i],
                            yes_calculated_data[i]);
    if (g_yes_power_spectrum_data[i] != yes_calculated_data[i]) {
      error_reporter->Report("Expected value %d but found %d",
                             g_yes_power_spectrum_data[i],
                             yes_calculated_data[i]);
    }
  }

  uint8_t no_calculated_data[g_yes_power_spectrum_data_size];
  TfLiteStatus no_status = GenerateSimpleFeatures(
      error_reporter, g_no_30ms_sample_data, g_no_30ms_sample_data_size,
      g_no_power_spectrum_data_size, no_calculated_data);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, no_status);

  for (int i = 0; i < g_no_power_spectrum_data_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(g_no_power_spectrum_data[i], no_calculated_data[i]);
    if (g_no_power_spectrum_data[i] != no_calculated_data[i]) {
      error_reporter->Report("Expected value %d but found %d",
                             g_no_power_spectrum_data[i],
                             no_calculated_data[i]);
    }
  }
}

TF_LITE_MICRO_TESTS_END
