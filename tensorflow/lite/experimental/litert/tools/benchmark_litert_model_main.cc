/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdlib>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/experimental/litert/tools/benchmark_litert_model.h"
#include "tensorflow/lite/tools/logging.h"

namespace litert::benchmark {

int Main(int argc, char** argv) {
  TFLITE_LOG(INFO) << "STARTING!";
  BenchmarkLiteRtModel benchmark;
  if (benchmark.Run(argc, argv) != kTfLiteOk) {
    TFLITE_LOG(ERROR) << "Benchmarking failed.";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
}  // namespace litert::benchmark

int main(int argc, char** argv) { return litert::benchmark::Main(argc, argv); }
