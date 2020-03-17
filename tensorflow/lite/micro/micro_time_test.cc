/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_time.h"

#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestBasicTimerFunctionality) {
  int32_t ticks_per_second = tflite::ticks_per_second();

  // Retry enough times to guarantee a tick advance, while not taking too long
  // to complete.  With 1e6 retries, assuming each loop takes tens of cycles,
  // this will retry for less than 10 seconds on a 10MHz platform.
  constexpr int kMaxRetries = 1e6;
  int start_time = tflite::GetCurrentTimeTicks();

  if (ticks_per_second != 0) {
    for (int i = 0; i < kMaxRetries; i++) {
      if (tflite::GetCurrentTimeTicks() - start_time > 0) {
        break;
      }
    }
  }

  // Ensure the timer is increasing. This works for the overflow case too, since
  // (MIN_INT + x) - (MAX_INT - y) == x + y + 1.  For example,
  // 0x80000001(min int + 1) - 0x7FFFFFFE(max int - 1) = 0x00000003 == 3.
  // GetTicksPerSecond() == 0 means the timer is not implemented on this
  // platform.
  TF_LITE_MICRO_EXPECT(ticks_per_second == 0 ||
                       tflite::GetCurrentTimeTicks() - start_time > 0);
}

TF_LITE_MICRO_TESTS_END
