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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_DELEGATE_PERFORMANCE_ANDROID_JNI_LATENCY_BENCHMARK_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_DELEGATE_PERFORMANCE_ANDROID_JNI_LATENCY_BENCHMARK_H_

#include <string>
#include <vector>

namespace tflite {
namespace benchmark {
namespace latency {

// Triggers TFLite Benchmark Tool. Passes the arguments from the testing app
// intent extra directly down to the TFLite Benchmark Tool. Generates
// report.json and benchmark_result.csv under `result_path`.
void Benchmark(const std::vector<std::string>& args, const char* result_path);

}  // namespace latency
}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_DELEGATE_PERFORMANCE_ANDROID_JNI_LATENCY_BENCHMARK_H_
