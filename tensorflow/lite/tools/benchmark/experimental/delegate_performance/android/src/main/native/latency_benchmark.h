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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_DELEGATE_PERFORMANCE_ANDROID_SRC_MAIN_NATIVE_LATENCY_BENCHMARK_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_DELEGATE_PERFORMANCE_ANDROID_SRC_MAIN_NATIVE_LATENCY_BENCHMARK_H_

#include <string>
#include <vector>

#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto/delegate_performance.pb.h"

namespace tflite {
namespace benchmark {
namespace latency {

// Triggers TFLite Benchmark Tool. Passes the "args" from the testing app to
// directly to TFLite Benchmark Tool. Converts the "tflite_settings" to
// command-line options to configure TFLite Benchmark Tool. If the latency
// benchmarking uses a stable delegate, the "tflite_settings_path" is passed to
// enable the stable delegate provider. The contents of the tested model are
// initialized using model_size bytes starting at model_offset position in the
// file referenced by the file descriptor model_fd.
//
// Returns a LatencyResults proto message. If the benchmark tests finish
// successfully from TFLite Benchmark Tool, the message contains the latency
// metrics. Otherwise, the message contains the corresponding error.
proto::benchmark::LatencyResults Benchmark(
    const TFLiteSettings& tflite_settings,
    const std::string& tflite_settings_path, int model_fd, size_t model_offset,
    size_t model_size, const std::vector<std::string>& args);

}  // namespace latency
}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_EXPERIMENTAL_DELEGATE_PERFORMANCE_ANDROID_SRC_MAIN_NATIVE_LATENCY_BENCHMARK_H_
