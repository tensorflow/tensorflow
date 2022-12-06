/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_DELEGATE_PERFORMANCE_ANDROID_JNI_ACCURACY_BENCHMARK_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_DELEGATE_PERFORMANCE_ANDROID_JNI_ACCURACY_BENCHMARK_H_

#include <cstddef>
#include <string>
#include <vector>

namespace tflite {
namespace benchmark {
namespace accuracy {

enum AccuracyBenchmarkStatus {
  kAccuracyBenchmarkUnknownStatus = 0,

  // General status codes.
  kAccuracyBenchmarkPass = 10,
  kAccuracyBenchmarkFail = 11,
  kAccuracyBenchmarkSuccess = 12,

  // Set of error codes that are used as the return codes to communicate between
  // AccuracyBenchmark and the caller app.
  kAccuracyBenchmarkRunnerInitializationFailed = 100,
  kAccuracyBenchmarkResultCountMismatch = 101,
  kAccuracyBenchmarkArgumentParsingFailed = 102,
  kAccuracyBenchmarkMoreThanOneDelegateProvided = 103,
  kAccuracyBenchmarkTfLiteSettingsParsingFailed = 104,
};

// Triggers MiniBenchmark testings. Parses the arguments passed from the
// testing app to configure MiniBenchmark ValidatorRunner. The tests will
// access and execute the pre-embedded models in the app via the model file
// descriptor. The contents of a model are initialized using model_size
// bytes starting at model_offset position in the file described by
// model_fd. Any intermediate data and results will be dumped to the result
// path given.
//
// Returns kAccuracyBenchmarkPass if the benchmark tests finish successfully
// with a pass from MiniBenchmark. Otherwise, returns kAccuracyBenchmarkFail
// or corresponding error codes.
AccuracyBenchmarkStatus Benchmark(const std::vector<std::string>& args,
                                  int model_fd, size_t model_offset,
                                  size_t model_size,
                                  const char* result_path_chars);

}  // namespace accuracy
}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_DELEGATE_PERFORMANCE_ANDROID_JNI_ACCURACY_BENCHMARK_H_
