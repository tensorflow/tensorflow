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
#include "tensorflow/lite/micro/examples/person_detection_experimental/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection_experimental/no_person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection_experimental/person_detect_model_data.h"
#include "tensorflow/lite/micro/examples/person_detection_experimental/person_image_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

/*
 * Person Detection benchmark.  Evaluates runtime performance of the visual
 * wakewords person detection model.  This is the same model found in
 * exmaples/person_detection.
 */

namespace {

using PersonDetectionExperimentalOpResolver = tflite::AllOpsResolver;
using PersonDetectionExperimentalBenchmarkRunner = MicroBenchmarkRunner<int8_t>;

// Create an area of memory to use for input, output, and intermediate arrays.
// Align arena to 16 bytes to avoid alignment warnings on certain platforms.
constexpr int kTensorArenaSize = 135 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

uint8_t op_resolver_buffer[sizeof(PersonDetectionExperimentalOpResolver)];
uint8_t
    benchmark_runner_buffer[sizeof(PersonDetectionExperimentalBenchmarkRunner)];
PersonDetectionExperimentalBenchmarkRunner* benchmark_runner = nullptr;

// Initialize benchmark runner instance explicitly to avoid global init order
// issues on Sparkfun. Use new since static variables within a method
// are automatically surrounded by locking, which breaks bluepill and stm32f4.
void CreateBenchmarkRunner() {
  // We allocate PersonDetectionExperimentalOpResolver from a global buffer
  // because the object's lifetime must exceed that of the
  // PersonDetectionBenchmarkRunner object.
  benchmark_runner =
      new (benchmark_runner_buffer) PersonDetectionExperimentalBenchmarkRunner(
          g_person_detect_model_data,
          new (op_resolver_buffer) PersonDetectionExperimentalOpResolver(),
          tensor_arena, kTensorArenaSize);
}

void InitializeBenchmarkRunner() {
  CreateBenchmarkRunner();
  benchmark_runner->SetInput(reinterpret_cast<const int8_t*>(g_person_data));
}

void PersonDetectionTenIerationsWithPerson() {
  benchmark_runner->SetInput(reinterpret_cast<const int8_t*>(g_person_data));
  for (int i = 0; i < 10; i++) {
    benchmark_runner->RunSingleIteration();
  }
}

void PersonDetectionTenIerationsWithoutPerson() {
  benchmark_runner->SetInput(reinterpret_cast<const int8_t*>(g_no_person_data));
  for (int i = 0; i < 10; i++) {
    benchmark_runner->RunSingleIteration();
  }
}

}  // namespace

TF_LITE_MICRO_BENCHMARKS_BEGIN

TF_LITE_MICRO_BENCHMARK(InitializeBenchmarkRunner());
TF_LITE_MICRO_BENCHMARK(benchmark_runner->RunSingleIteration());
TF_LITE_MICRO_BENCHMARK(PersonDetectionTenIerationsWithPerson());
TF_LITE_MICRO_BENCHMARK(PersonDetectionTenIerationsWithoutPerson());

TF_LITE_MICRO_BENCHMARKS_END
