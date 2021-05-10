/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
  // This test times out on a Bluepill because of the very slow FFT
  // calculations, so replace it with a version that does nothing for this
  // platform.
}

TF_LITE_MICRO_TESTS_END
