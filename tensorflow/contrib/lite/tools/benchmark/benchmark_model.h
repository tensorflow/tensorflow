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

#ifndef TENSORFLOW_CONTRIB_LITE_TOOLS_BENCHMARK_MODEL_H_
#define TENSORFLOW_CONTRIB_LITE_TOOLS_BENCHMARK_MODEL_H_

#include <cmath>
#include <limits>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/contrib/lite/tools/benchmark/command_line_flags.h"
#include "tensorflow/core/util/stats_calculator.h"

namespace tflite {
namespace benchmark {

enum RunType {
  WARMUP,
  REGULAR,
};

class BenchmarkResults {
 public:
  BenchmarkResults(int64_t startup_latency_us, uint64_t input_bytes,
                   tensorflow::Stat<int64_t> warmup_time_us,
                   tensorflow::Stat<int64_t> inference_time_us)
      : startup_latency_us_(startup_latency_us),
        input_bytes_(input_bytes),
        warmup_time_us_(warmup_time_us),
        inference_time_us_(inference_time_us) {}

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

 private:
  int64_t startup_latency_us_;
  uint64_t input_bytes_;
  tensorflow::Stat<int64_t> warmup_time_us_;
  tensorflow::Stat<int64_t> inference_time_us_;
};

struct BenchmarkParams {
  BenchmarkParams()
      : num_runs(50), warmup_runs(1), run_delay(-1.0), num_threads(1) {}
  int num_runs;
  int warmup_runs;
  float run_delay;
  int num_threads;
  std::string benchmark_name;
  std::string output_prefix;
};

class BenchmarkListener {
 public:
  virtual void OnBenchmarkStart(const BenchmarkParams& params) {}
  virtual void OnSingleRunStart(RunType runType) {}
  virtual void OnSingleRunEnd() {}
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

  ~BenchmarkListeners() {}

 private:
  // Use vector so listeners are invoked in the order they are added.
  std::vector<BenchmarkListener*> listeners_;
};

// Benchmark listener that just logs the results of benchmark run.
class BenchmarkLoggingListener : public BenchmarkListener {
  void OnBenchmarkEnd(const BenchmarkResults& results) override;
};

// Benchmarks a model.
//
// Subclasses need to implement initialization and running of the model.
// The results can be collected by adding BenchmarkListener(s).
class BenchmarkModel {
 public:
  virtual ~BenchmarkModel() {}
  bool ParseFlags(int argc, char** argv);
  virtual void Init() = 0;
  void Run(int argc, char** argv);
  void AddListener(BenchmarkListener* listener) {
    listeners_.AddListener(listener);
  }

 protected:
  virtual void LogFlags();
  virtual bool ValidateFlags() { return true; }
  virtual std::vector<Flag> GetFlags();
  virtual uint64_t ComputeInputBytes() = 0;
  virtual tensorflow::Stat<int64_t> Run(int num_times, RunType run_type);
  virtual void RunImpl() = 0;
  BenchmarkParams params_;
  BenchmarkListeners listeners_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_TOOLS_BENCHMARK_BENCHMARK_MODEL_H_
