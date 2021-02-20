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

#include <cstdint>
#include <cstdlib>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/benchmarks/keyword_scrambled_model_data.h"
#include "tensorflow/lite/micro/benchmarks/micro_benchmark.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/system_setup.h"

/*
 * Keyword Spotting Benchmark for performance optimizations. The model used in
 * this benchmark only serves as a reference. The values assigned to the model
 * weights and parameters are not representative of the original model.
 */

namespace tflite {

using KeywordBenchmarkRunner = MicroBenchmarkRunner<int16_t>;
using KeywordOpResolver = MicroMutableOpResolver<6>;

// Create an area of memory to use for input, output, and intermediate arrays.
// Align arena to 16 bytes to avoid alignment warnings on certain platforms.
constexpr int kTensorArenaSize = 21 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

uint8_t benchmark_runner_buffer[sizeof(KeywordBenchmarkRunner)];
uint8_t op_resolver_buffer[sizeof(KeywordOpResolver)];

// Initialize benchmark runner instance explicitly to avoid global init order
// issues on Sparkfun. Use new since static variables within a method
// are automatically surrounded by locking, which breaks bluepill and stm32f4.
KeywordBenchmarkRunner* CreateBenchmarkRunner(MicroProfiler* profiler) {
  // We allocate the KeywordOpResolver from a global buffer because the object's
  // lifetime must exceed that of the KeywordBenchmarkRunner object.
  KeywordOpResolver* op_resolver = new (op_resolver_buffer) KeywordOpResolver();
  op_resolver->AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8());
  op_resolver->AddQuantize();
  op_resolver->AddSoftmax();
  op_resolver->AddSvdf();

  return new (benchmark_runner_buffer)
      KeywordBenchmarkRunner(g_keyword_scrambled_model_data, op_resolver,
                             tensor_arena, kTensorArenaSize, profiler);
}

void KeywordRunNIerations(int iterations, const char* tag,
                          KeywordBenchmarkRunner& benchmark_runner,
                          MicroProfiler& profiler) {
  int32_t ticks = 0;
  for (int i = 0; i < iterations; ++i) {
    benchmark_runner.SetRandomInput(i);
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

  uint32_t event_handle = profiler.BeginEvent("InitializeKeywordRunner");
  tflite::KeywordBenchmarkRunner* benchmark_runner =
      CreateBenchmarkRunner(&profiler);
  profiler.EndEvent(event_handle);
  profiler.Log();
  MicroPrintf("");  // null MicroPrintf serves as a newline.

  tflite::KeywordRunNIerations(1, "KeywordRunNIerations(1)", *benchmark_runner,
                               profiler);
  profiler.Log();
  MicroPrintf("");  // null MicroPrintf serves as a newline.

  tflite::KeywordRunNIerations(10, "KeywordRunNIerations(10)",
                               *benchmark_runner, profiler);
  MicroPrintf("");  // null MicroPrintf serves as a newline.
}
