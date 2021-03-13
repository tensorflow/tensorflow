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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/benchmarks/micro_benchmark.h"
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection/no_person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/person_detect_model_data.h"
#include "tensorflow/lite/micro/examples/person_detection/person_image_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

/*
 * Person Detection benchmark.  Evaluates runtime performance of the visual
 * wakewords person detection model.  This is the same model found in
 * exmaples/person_detection.
 */

namespace tflite {

using PersonDetectionOpResolver = tflite::AllOpsResolver;
using PersonDetectionBenchmarkRunner = MicroBenchmarkRunner<int8_t>;

// Create an area of memory to use for input, output, and intermediate arrays.
// Align arena to 16 bytes to avoid alignment warnings on certain platforms.
constexpr int kTensorArenaSize = 135 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

uint8_t op_resolver_buffer[sizeof(PersonDetectionOpResolver)];
uint8_t benchmark_runner_buffer[sizeof(PersonDetectionBenchmarkRunner)];

// Initialize benchmark runner instance explicitly to avoid global init order
// issues on Sparkfun. Use new since static variables within a method
// are automatically surrounded by locking, which breaks bluepill and stm32f4.
PersonDetectionBenchmarkRunner* CreateBenchmarkRunner(MicroProfiler* profiler) {
  // We allocate PersonDetectionOpResolver from a global buffer
  // because the object's lifetime must exceed that of the
  // PersonDetectionBenchmarkRunner object.
  return new (benchmark_runner_buffer) PersonDetectionBenchmarkRunner(
      g_person_detect_model_data,
      new (op_resolver_buffer) PersonDetectionOpResolver(), tensor_arena,
      kTensorArenaSize, profiler);
}

void PersonDetectionNIerations(const int8_t* input, int iterations,
                               const char* tag,
                               PersonDetectionBenchmarkRunner& benchmark_runner,
                               MicroProfiler& profiler) {
  benchmark_runner.SetInput(input);
  int32_t ticks = 0;
  for (int i = 0; i < iterations; ++i) {
    profiler.ClearEvents();
    benchmark_runner.RunSingleIteration();
    ticks += profiler.GetTotalTicks();
  }
  MicroPrintf("%s took %d ticks (%d ms)", tag, ticks, TicksToMs(ticks));
}

}  // namespace tflite

int main(int argc, char** argv) {
  tflite::InitializeTarget();

  tflite::MicroProfiler profiler;

  uint32_t event_handle = profiler.BeginEvent("InitializeBenchmarkRunner");
  tflite::PersonDetectionBenchmarkRunner* benchmark_runner =
      CreateBenchmarkRunner(&profiler);
  profiler.EndEvent(event_handle);
  profiler.Log();
  MicroPrintf("");  // null MicroPrintf serves as a newline.

  tflite::PersonDetectionNIerations(
      reinterpret_cast<const int8_t*>(g_person_data), 1,
      "WithPersonDataIterations(1)", *benchmark_runner, profiler);
  profiler.Log();
  MicroPrintf("");  // null MicroPrintf serves as a newline.

  tflite::PersonDetectionNIerations(
      reinterpret_cast<const int8_t*>(g_no_person_data), 1,
      "NoPersonDataIterations(1)", *benchmark_runner, profiler);
  profiler.Log();
  MicroPrintf("");  // null MicroPrintf serves as a newline.

  tflite::PersonDetectionNIerations(
      reinterpret_cast<const int8_t*>(g_person_data), 10,
      "WithPersonDataIterations(10)", *benchmark_runner, profiler);
  MicroPrintf("");  // null MicroPrintf serves as a newline.

  tflite::PersonDetectionNIerations(
      reinterpret_cast<const int8_t*>(g_no_person_data), 10,
      "NoPersonDataIterations(10)", *benchmark_runner, profiler);
  MicroPrintf("");  // null MicroPrintf serves as a newline.
}
