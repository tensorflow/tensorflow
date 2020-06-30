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
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

/*
 * Keyword Spotting Benchmark for performance optimizations. The model used in
 * this benchmark only serves as a reference. The values assigned to the model
 * weights and parameters are not representative of the original model.
 */

namespace {

// Create an area of memory to use for input, output, and intermediate arrays.
// Align arena to 16 bytes to avoid alignment warnings on certain platforms.
constexpr int tensor_arena_size = 21 * 1024;
alignas(16) uint8_t tensor_arena[tensor_arena_size];
// A random number generator seed to generate input values.
constexpr int kRandomSeed = 42;

MicroBenchmarkRunner<int16_t>& GetBenchmarkRunner() {
  // NOLINTNEXTLINE
  static MicroBenchmarkRunner<int16_t> runner(
      g_keyword_scrambled_model_data, tensor_arena, tensor_arena_size, 0);
  return runner;
}

void KeywordRunTenIerations() {
  // TODO(b/152644476): Add a way to run more than a single deterministic input.
  for (int i = 0; i < 10; i++) {
    GetBenchmarkRunner().RunSingleIterationRandomInput();
  }
}

}  //  namespace

TF_LITE_MICRO_BENCHMARKS_BEGIN

TF_LITE_MICRO_BENCHMARK(GetBenchmarkRunner().RunSingleIterationRandomInput());

TF_LITE_MICRO_BENCHMARK(KeywordRunTenIerations());

TF_LITE_MICRO_BENCHMARKS_END
