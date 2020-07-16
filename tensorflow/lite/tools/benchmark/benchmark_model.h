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
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace tflite {
namespace benchmark {

enum RunType {
  WARMUP,
  REGULAR,
};

class BenchmarkResults {
 public:
  BenchmarkResults() {}
  BenchmarkResults(double model_size_mb, int64_t startup_latency_us,
                   uint64_t input_bytes,
                   tensorflow::Stat<int64_t> warmup_time_us,
                   tensorflow::Stat<int64_t> inference_time_us,
                   const profiling::memory::MemoryUsage& init_mem_usage,
                   const profiling::memory::MemoryUsage& overall_mem_usage)
      : model_size_mb_(model_size_mb),
        startup_latency_us_(startup_latency_us),
        input_bytes_(input_bytes),
        warmup_time_us_(warmup_time_us),
        inference_time_us_(inference_time_us),
        init_mem_usage_(init_mem_usage),
        overall_mem_usage_(overall_mem_usage) {}

  const double model_size_mb() const { return model_size_mb_; }
  tensorflow::Stat<int64_t> inference_time_us() const {
    return inference_time_us_;
  }
  tensorflow::Stat<int64_t> warmup_time_us() const { return warmup_time_us_; }
  int64_t startup_latency_us() const { return startup_latency_us_; }
  uint64_t input_bytes() const { return input_bytes_; }
  double throughput_MB_per_second() const {
    double bytes_per_sec = (input_bytes_ * inference_time_us_.count() * 1e6) /
                           inference_time_us_.sum();
    return bytes_per_sec / (1024.0 * 1024.0);
  }

  const profiling::memory::MemoryUsage& init_mem_usage() const {
    return init_mem_usage_;
  }
  const profiling::memory::MemoryUsage& overall_mem_usage() const {
    return overall_mem_usage_;
  }

 private:
  double model_size_mb_ = 0.0;
  int64_t startup_latency_us_ = 0;
  uint64_t input_bytes_ = 0;
  tensorflow::Stat<int64_t> warmup_time_us_;
  tensorflow::Stat<int64_t> inference_time_us_;
  profiling::memory::MemoryUsage init_mem_usage_;
  profiling::memory::MemoryUsage overall_mem_usage_;
};

class BenchmarkListener {
 public:
  // Called before the (outer) inference loop begins.
  // Note that this is called *after* the interpreter has been initialized, but
  // *before* any warmup runs have been executed.
  virtual void OnBenchmarkStart(const BenchmarkParams& params) {}
  // Called before a single (inner) inference call starts.
  virtual void OnSingleRunStart(RunType runType) {}
  // Called before a single (inner) inference call ends.
  virtual void OnSingleRunEnd() {}
  // Called after the (outer) inference loop begins.
  virtual void OnBenchmarkEnd(const BenchmarkResults& results) {}
  virtual ~BenchmarkListener() {}
};

// A listener that forwards its method calls to a collection of listeners.
class BenchmarkListeners : public BenchmarkListener {
 public:
  // Added a listener to the listener collection.
  // |listener| is not owned by the instance of |BenchmarkListeners|.
  // |listener| should not be null and should outlast the instance of
  // |BenchmarkListeners|.
  void AddListener(BenchmarkListener* listener) {
    listeners_.push_back(listener);
  }

  // Remove all listeners after [index] including the one at 'index'.
  void RemoveListeners(int index) {
    if (index >= NumListeners()) return;
    listeners_.resize(index);
  }

  int NumListeners() const { return listeners_.size(); }

  void OnBenchmarkStart(const BenchmarkParams& params) override {
    for (auto listener : listeners_) {
      listener->OnBenchmarkStart(params);
    }
  }

  void OnSingleRunStart(RunType runType) override {
    for (auto listener : listeners_) {
      listener->OnSingleRunStart(runType);
    }
  }

  void OnSingleRunEnd() override {
    for (auto listener : listeners_) {
      listener->OnSingleRunEnd();
    }
  }

  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    for (auto listener : listeners_) {
      listener->OnBenchmarkEnd(results);
    }
  }

  ~BenchmarkListeners() override {}

 private:
  // Use vector so listeners are invoked in the order they are added.
  std::vector<BenchmarkListener*> listeners_;
};

// Benchmark listener that just logs the results of benchmark run.
class BenchmarkLoggingListener : public BenchmarkListener {
 public:
  void OnBenchmarkEnd(const BenchmarkResults& results) override;
};

template <typename T>
Flag CreateFlag(const char* name, BenchmarkParams* params,
                const std::string& usage) {
  return Flag(
      name, [params, name](const T& val) { params->Set<T>(name, val); },
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
  virtual ~BenchmarkModel() {}
  virtual TfLiteStatus Init() = 0;
  TfLiteStatus Run(int argc, char** argv);
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
  virtual tensorflow::Stat<int64_t> Run(int min_num_times, float min_secs,
                                        float max_secs, RunType run_type,
                                        TfLiteStatus* invoke_status);
  // Prepares input data for benchmark. This can be used to initialize input
  // data that has non-trivial cost.
  virtual TfLiteStatus PrepareInputData();

  virtual TfLiteStatus ResetInputsAndOutputs();
  virtual TfLiteStatus RunImpl() = 0;
  BenchmarkParams params_;
  BenchmarkListeners listeners_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_
