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

/* This file is a modification of the Tensorflow Micro Lite file preprocessor.cc
 */

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/micro_speech/CMSIS/sin_1k.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

extern "C" {
#include "apollo3.h"
#include "system_apollo3.h"
}

#define output_data_size 43
int count;

extern TfLiteStatus Preprocess(tflite::ErrorReporter* error_reporter,
                               const int16_t* input, int input_size,
                               int output_size, uint8_t* output);

TF_LITE_MICRO_TESTS_BEGIN
CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
// DWT->LAR = 0xC5ACCE55;
DWT->CYCCNT = 0;
DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

TF_LITE_MICRO_TEST(TestPreprocessor) {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  uint8_t calculated_data[output_data_size];
  TfLiteStatus yes_status = Preprocess(error_reporter, g_sin_1k, g_sin_1k_size,
                                       output_data_size, calculated_data);
  count = DWT->CYCCNT;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, yes_status);
}

TF_LITE_MICRO_TESTS_END
