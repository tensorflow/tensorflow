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
#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_MULTIRUN_STATS_RECORDER_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_MULTIRUN_STATS_RECORDER_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/tools/benchmark/benchmark_model.h"

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
}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_BENCHMARK_MULTIRUN_STATS_RECORDER_H_

