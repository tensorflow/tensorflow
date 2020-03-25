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

#ifndef TENSORFLOW_LITE_MICRO_TESTING_MICRO_BENCHMARK_H_
#define TENSORFLOW_LITE_MICRO_TESTING_MICRO_BENCHMARK_H_

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace micro_benchmark {
extern tflite::ErrorReporter* reporter;
}  // namespace micro_benchmark

#define TF_LITE_MICRO_BENCHMARKS_BEGIN           \
  namespace micro_benchmark {                    \
  tflite::ErrorReporter* reporter;               \
  }                                              \
                                                 \
  int main(int argc, char** argv) {              \
    tflite::MicroErrorReporter error_reporter;   \
    micro_benchmark::reporter = &error_reporter; \
    int32_t start_ticks;                         \
    int32_t duration_ticks;                      \
    int32_t duration_ms;

#define TF_LITE_MICRO_BENCHMARKS_END }

#define TF_LITE_MICRO_BENCHMARK(func)                                         \
  start_ticks = tflite::GetCurrentTimeTicks();                                \
  func();                                                                     \
  duration_ticks = tflite::GetCurrentTimeTicks() - start_ticks;               \
  duration_ms = (duration_ticks * 1000) / tflite::ticks_per_second();         \
  TF_LITE_REPORT_ERROR(micro_benchmark::reporter, "%s took %d ticks (%d ms)", \
                       #func, duration_ticks, duration_ms);

#endif  // TENSORFLOW_LITE_MICRO_TESTING_MICRO_BENCHMARK_H_
