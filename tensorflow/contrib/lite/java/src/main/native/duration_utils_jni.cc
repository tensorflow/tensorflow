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

#include <jni.h>
#include <time.h>

namespace tflite {

// Gets the elapsed wall-clock timespec.
timespec getCurrentTime() {
  timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return time;
}

// Computes the time diff from two timespecs. Returns '-1' if 'stop' is earlier
// than 'start'.
jlong timespec_diff_nanoseconds(struct timespec* start, struct timespec* stop) {
  jlong result = stop->tv_sec - start->tv_sec;
  if (result < 0) return -1;
  result = 1000000000 * result + (stop->tv_nsec - start->tv_nsec);
  if (result < 0) return -1;
  return result;
}

}  // namespace tflite
