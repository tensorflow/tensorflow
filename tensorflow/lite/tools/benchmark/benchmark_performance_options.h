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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_PERFORMANCE_OPTIONS_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_PERFORMANCE_OPTIONS_H_

#include "tensorflow/lite/tools/benchmark/benchmark_model.h"

namespace tflite {
namespace benchmark {

// Benchmarks all performance options on a model by repeatedly invoking the
// single-performance-option run on a passed-in 'BenchmarkModel' object.
class BenchmarkPerformanceOptions {
 public:
  // Doesn't own the memory of 'single_option_run'.
  explicit BenchmarkPerformanceOptions(BenchmarkModel* single_option_run)
      : BenchmarkPerformanceOptions(DefaultParams(), single_option_run) {}

  BenchmarkPerformanceOptions(BenchmarkParams params,
                              BenchmarkModel* single_option_run)
      : params_(std::move(params)),
        single_option_run_(single_option_run),
        single_option_run_params_(single_option_run->mutable_params()) {}

  virtual ~BenchmarkPerformanceOptions() {}

  virtual void Run(int argc, char** argv);

 protected:
  static BenchmarkParams DefaultParams();

  // Unparsable flags will remain in 'argv' in the original order and 'argc'
  // will be updated accordingly.
  bool ParseFlags(int* argc, char** argv);
  virtual std::vector<Flag> GetFlags();

  bool ParsePerfOptions();
  virtual std::vector<std::string> GetValidPerfOptions() const;
  bool HasOption(const string& option) const;
  virtual void ResetPerformanceOptions();

  virtual void BenchmarkCPUOptions();
  virtual void BenchmarkGPUOptions();
  virtual void BenchmarkNnapiOptions();

  BenchmarkParams params_;
  std::vector<std::string> perf_options_;

  // The object that drives a single-performance-option run.
  BenchmarkModel* const single_option_run_;          // Doesn't own the memory.
  BenchmarkParams* const single_option_run_params_;  // Doesn't own the memory.
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_PERFORMANCE_OPTIONS_H_
