/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_LISTENER_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_LISTENER_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/profiling/memory_info.h"
#include "tensorflow/lite/profiling/memory_usage_monitor.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"

namespace tflite {
namespace benchmark {

enum RunType {
  WARMUP,
  REGULAR,
};

class BenchmarkResults {
 public:
  BenchmarkResults() = default;
  BenchmarkResults(double model_size_mb, int64_t startup_latency_us,
                   uint64_t input_bytes,
                   tensorflow::StatWithPercentiles<int64_t> warmup_time_us,
                   tensorflow::StatWithPercentiles<int64_t> inference_time_us,
                   const profiling::memory::MemoryUsage& init_mem_usage,
                   const profiling::memory::MemoryUsage& overall_mem_usage,
                   float peak_mem_mb)
      : model_size_mb_(model_size_mb),
        startup_latency_us_(startup_latency_us),
        input_bytes_(input_bytes),
        warmup_time_us_(warmup_time_us),
        inference_time_us_(inference_time_us),
        init_mem_usage_(init_mem_usage),
        overall_mem_usage_(overall_mem_usage),
        peak_mem_mb_(peak_mem_mb) {}

  double model_size_mb() const { return model_size_mb_; }
  tensorflow::StatWithPercentiles<int64_t> inference_time_us() const {
    return inference_time_us_;
  }
  tensorflow::StatWithPercentiles<int64_t> warmup_time_us() const {
    return warmup_time_us_;
  }
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
  float peak_mem_mb() const { return peak_mem_mb_; }

 private:
  double model_size_mb_ = 0.0;
  int64_t startup_latency_us_ = 0;
  uint64_t input_bytes_ = 0;
  tensorflow::StatWithPercentiles<int64_t> warmup_time_us_;
  tensorflow::StatWithPercentiles<int64_t> inference_time_us_;
  profiling::memory::MemoryUsage init_mem_usage_;
  profiling::memory::MemoryUsage overall_mem_usage_;
  // An invalid value could happen when we don't monitor memory footprint for
  // the inference, or the memory usage info isn't available on the benchmarking
  // platform.
  float peak_mem_mb_ =
      profiling::memory::MemoryUsageMonitor::kInvalidMemUsageMB;
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
  virtual ~BenchmarkListener() = default;
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

  // Remove a specific listener instance if present.
  void RemoveListener(const BenchmarkListener* listener) {
    listeners_.erase(
        std::remove(listeners_.begin(), listeners_.end(), listener),
        listeners_.end());
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

  ~BenchmarkListeners() override = default;

 private:
  // Use vector so listeners are invoked in the order they are added.
  std::vector<BenchmarkListener*> listeners_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_LISTENER_H_
