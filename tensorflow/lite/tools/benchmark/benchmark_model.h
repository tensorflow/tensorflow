/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/memory_usage_monitor.h"
#include "tensorflow/lite/tools/benchmark/benchmark_listener.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace tflite {
namespace benchmark {

// Benchmark listener that just logs the results of benchmark run.
class BenchmarkLoggingListener : public BenchmarkListener {
 public:
  void OnBenchmarkEnd(const BenchmarkResults& results) override;
};

template <typename T>
Flag CreateFlag(const char* name, BenchmarkParams* params,
                const std::string& usage) {
  return Flag(
      name,
      [params, name](const T& val, int argv_position) {
        params->Set<T>(name, val, argv_position);
      },
      params->Get<T>(name), usage, Flag::kOptional);
}

// Benchmarks a model.
//
// Subclasses need to implement initialization and running of the model.
// The results can be collected by adding BenchmarkListener(s).
class BenchmarkModel {
 public:
  static BenchmarkParams DefaultParams();
  BenchmarkModel();
  explicit BenchmarkModel(BenchmarkParams params)
      : params_(std::move(params)) {}
  virtual ~BenchmarkModel() = default;
  virtual TfLiteStatus Init() = 0;
  virtual TfLiteStatus Run(int argc, char** argv);
  virtual TfLiteStatus Run();
  void AddListener(BenchmarkListener* listener) {
    listeners_.AddListener(listener);
  }
  // Remove all listeners after [index] including the one at 'index'.
  void RemoveListeners(int index) { listeners_.RemoveListeners(index); }
  int NumListeners() const { return listeners_.NumListeners(); }

  BenchmarkParams* mutable_params() { return &params_; }

  // Unparsable flags will remain in 'argv' in the original order and 'argc'
  // will be updated accordingly.
  TfLiteStatus ParseFlags(int* argc, char** argv);

 protected:
  virtual void LogParams();
  virtual TfLiteStatus ValidateParams();

  TfLiteStatus ParseFlags(int argc, char** argv) {
    return ParseFlags(&argc, argv);
  }
  virtual std::vector<Flag> GetFlags();

  // Get the model file size if it's available.
  virtual int64_t MayGetModelFileSize() { return -1; }
  virtual uint64_t ComputeInputBytes() = 0;
  virtual tensorflow::StatWithPercentiles<int64_t> Run(
      int min_num_times, float min_secs, float max_secs, RunType run_type,
      TfLiteStatus* invoke_status);
  // Prepares input data for benchmark. This can be used to initialize input
  // data that has non-trivial cost.
  virtual TfLiteStatus PrepareInputData();

  virtual TfLiteStatus ResetInputsAndOutputs();
  virtual TfLiteStatus RunImpl() = 0;

  // Create a MemoryUsageMonitor to report peak memory footprint if specified.
  virtual std::unique_ptr<profiling::memory::MemoryUsageMonitor>
  MayCreateMemoryUsageMonitor() const;

  // Create an opt-in BenchmarkListener to report GPU memory if specified,
  // leaving default BenchmarkResults metrics unchanged.
  virtual std::unique_ptr<BenchmarkListener>
  MayCreateGpuMemoryProfilingListener() const;

  BenchmarkParams params_;
  BenchmarkListeners listeners_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_
