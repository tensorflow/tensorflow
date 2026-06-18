/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TOOLS_BENCHMARK_PROFILING_LISTENER_H_
#define TENSORFLOW_LITE_TOOLS_BENCHMARK_PROFILING_LISTENER_H_

#include <cstdint>
#include <memory>
#include <string>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/profiling/buffered_profiler.h"
#include "tensorflow/lite/profiling/profile_summarizer.h"
#include "tensorflow/lite/profiling/profile_summary_formatter.h"
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"

namespace tflite {
namespace benchmark {

// Dumps profiling events if profiling is enabled.
class ProfilingListener : public BenchmarkListener {
 public:
  ProfilingListener(
      Interpreter* interpreter, uint32_t max_num_initial_entries,
      bool allow_dynamic_buffer_increase,
      const std::string& output_file_path = "",
      std::shared_ptr<profiling::ProfileSummaryFormatter> summarizer_formatter =
          std::make_shared<profiling::ProfileSummaryDefaultFormatter>());

  void OnBenchmarkStart(const BenchmarkParams& params) override;

  void OnSingleRunStart(RunType run_type) override;

  void OnSingleRunEnd() override;

  void OnBenchmarkEnd(const BenchmarkResults& results) override;

 protected:
  profiling::ProfileSummarizer run_summarizer_;
  profiling::ProfileSummarizer init_summarizer_;
  std::string output_file_path_;

 private:
  Interpreter* interpreter_;
  profiling::BufferedProfiler profiler_;
  std::shared_ptr<profiling::ProfileSummaryFormatter> summarizer_formatter_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_PROFILING_LISTENER_H_
