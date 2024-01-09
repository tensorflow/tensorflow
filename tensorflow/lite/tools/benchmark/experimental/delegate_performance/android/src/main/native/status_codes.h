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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_DELEGATE_PERFORMANCE_ANDROID_SRC_MAIN_NATIVE_STATUS_CODES_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_DELEGATE_PERFORMANCE_ANDROID_SRC_MAIN_NATIVE_STATUS_CODES_H_

namespace tflite {
namespace benchmark {

enum DelegatePerformanceBenchmarkStatus {
  kBenchmarkUnknownStatus = 0,

  // Set of error codes that are used as the return codes to communicate between
  // the native layer and the caller app.
  kBenchmarkRunnerInitializationFailed = 1000,
  kBenchmarkResultCountMismatch = 1001,
  kBenchmarkInvalidTfLiteSettings = 1002,
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_DELEGATE_PERFORMANCE_ANDROID_SRC_MAIN_NATIVE_STATUS_CODES_H_
