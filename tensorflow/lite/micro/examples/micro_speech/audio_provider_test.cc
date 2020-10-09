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

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"

#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestAudioProvider) {
  tflite::MicroErrorReporter micro_error_reporter;

  int audio_samples_size = 0;
  int16_t* audio_samples = nullptr;
  TfLiteStatus get_status =
      GetAudioSamples(&micro_error_reporter, 0, kFeatureSliceDurationMs,
                      &audio_samples_size, &audio_samples);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, get_status);
  TF_LITE_MICRO_EXPECT_LE(audio_samples_size, kMaxAudioSampleSize);
  TF_LITE_MICRO_EXPECT_NE(audio_samples, nullptr);

  // Make sure we can read all of the returned memory locations.
  int total = 0;
  for (int i = 0; i < audio_samples_size; ++i) {
    total += audio_samples[i];
  }
}

TF_LITE_MICRO_TEST(TestTimer) {
  // Make sure that the technically-undefined overflow behavior we rely on below
  // works on this platform. It's still not guaranteed, but at least this is a
  // smoke check.  Turn off when running with ASan, as it will complain about
  // the following undefined behavior.
#ifndef ADDRESS_SANITIZER
  int32_t overflow_value = std::numeric_limits<int32_t>::max();
  overflow_value += 1;
  TF_LITE_MICRO_EXPECT_EQ(std::numeric_limits<int32_t>::min(), overflow_value);
#endif

  const int32_t first_time = LatestAudioTimestamp();
  const int32_t second_time = LatestAudioTimestamp();

  // It's possible that the timer may have wrapped around from +BIG_NUM to
  // -BIG_NUM between the first and second calls, since we're storing
  // milliseconds in a 32-bit integer. It's not reasonable that the call itself
  // would have taken more than 2^31 milliseconds though, so look at the
  // difference and rely on integer overflow to ensure it's accurate.
  const int32_t time_delta = (second_time - first_time);
  TF_LITE_MICRO_EXPECT_LE(0, time_delta);
}

TF_LITE_MICRO_TESTS_END
