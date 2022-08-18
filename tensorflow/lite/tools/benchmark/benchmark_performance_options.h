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

#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"

namespace tflite {
namespace benchmark {

class MultiRunStatsRecorder : public BenchmarkListener {
 public:
  // BenchmarkListener::OnBenchmarkStart is invoked after each run's
  // BenchmarkModel::Init. However, some run could fail during Init, e.g.
  // delegate fails to be created etc. To still record such run, we will call
  // the following function right before a run starts.
  void MarkBenchmarkStart(const BenchmarkParams& params) {
    results_.emplace_back(EachRunResult());
    auto& current = results_.back();
    current.completed = false;
    current.params = std::make_unique<BenchmarkParams>();
    current.params->Merge(params, true /* overwrite*/);
  }

  void OnBenchmarkEnd(const BenchmarkResults& results) final {
    auto& current = results_.back();
    current.completed = true;
    current.metrics = results;
  }

  virtual void OutputStats();

 protected:
  struct EachRunResult {
    bool completed = false;
    std::unique_ptr<BenchmarkParams> params;
    BenchmarkResults metrics;
  };
  std::vector<EachRunResult> results_;

  // Use this to order the runs by the average inference time in increasing
  // order (i.e. the fastest run ranks first.). If the run didn't complete,
  // we consider it to be slowest.
  struct EachRunStatsEntryComparator {
    bool operator()(const EachRunResult& i, const EachRunResult& j) {
      if (!i.completed) return false;
      if (!j.completed) return true;
      return i.metrics.inference_time_us().avg() <
             j.metrics.inference_time_us().avg();
    }
  };

  virtual std::string PerfOptionName(const BenchmarkParams& params) const;
};

// Benchmarks all performance options on a model by repeatedly invoking the
// single-performance-option run on a passed-in 'BenchmarkModel' object.
class BenchmarkPerformanceOptions {
 public:
  // Doesn't own the memory of 'single_option_run'.
  explicit BenchmarkPerformanceOptions(
      BenchmarkModel* single_option_run,
      std::unique_ptr<MultiRunStatsRecorder> all_run_stats =
          std::make_unique<MultiRunStatsRecorder>());

  virtual ~BenchmarkPerformanceOptions() {}

  // Just run the benchmark just w/ default parameter values.
  void Run();
  void Run(int argc, char** argv);

 protected:
  static BenchmarkParams DefaultParams();

  BenchmarkPerformanceOptions(
      BenchmarkParams params, BenchmarkModel* single_option_run,
      std::unique_ptr<MultiRunStatsRecorder> all_run_stats);

  // Unparsable flags will remain in 'argv' in the original order and 'argc'
  // will be updated accordingly.
  bool ParseFlags(int* argc, char** argv);
  virtual std::vector<Flag> GetFlags();

  bool ParsePerfOptions();
  virtual std::vector<std::string> GetValidPerfOptions() const;
  bool HasOption(const std::string& option) const;

  virtual void ResetPerformanceOptions();
  virtual void CreatePerformanceOptions();

  BenchmarkParams params_;
  std::vector<std::string> perf_options_;

  // The object that drives a single-performance-option run.
  BenchmarkModel* const single_option_run_;          // Doesn't own the memory.
  BenchmarkParams* const single_option_run_params_;  // Doesn't own the memory.

  // Each element is a set of performance-affecting benchmark parameters to be
  // all set for a particular benchmark run.
  std::vector<BenchmarkParams> all_run_params_;

  std::unique_ptr<MultiRunStatsRecorder> all_run_stats_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_PERFORMANCE_OPTIONS_H_
