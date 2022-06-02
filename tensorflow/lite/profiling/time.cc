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
#include "tensorflow/lite/profiling/time.h"

#include <sys/time.h>
#include <time.h>

namespace tflite {
namespace profiling {
namespace time {

uint64_t NowMicros() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1e6 +
         static_cast<uint64_t>(ts.tv_nsec) / 1e3;
}

void SleepForMicros(uint64_t micros) {
  timespec sleep_time;
  sleep_time.tv_sec = micros / 1e6;
  micros -= sleep_time.tv_sec * 1e6;
  sleep_time.tv_nsec = micros * 1e3;
  nanosleep(&sleep_time, nullptr);
}

}  // namespace time
}  // namespace profiling
}  // namespace tflite
