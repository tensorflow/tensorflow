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

#include "tensorflow/lite/profiling/buffered_profiler.h"
#include "tensorflow/lite/profiling/profile_summarizer.h"
#include "tensorflow/lite/tools/benchmark/benchmark_model.h"

namespace tflite {
namespace benchmark {

// Dumps profiling events if profiling is enabled.
class ProfilingListener : public BenchmarkListener {
 public:
  explicit ProfilingListener(Interpreter* interpreter, uint32_t max_num_entries,
                             const std::string& csv_file_path = "");

  void OnBenchmarkStart(const BenchmarkParams& params) override;

  void OnSingleRunStart(RunType run_type) override;

  void OnSingleRunEnd() override;

  void OnBenchmarkEnd(const BenchmarkResults& results) override;

 protected:
  // Allow subclasses to create a customized summary writer during init.
  virtual std::unique_ptr<profiling::ProfileSummaryFormatter>
  CreateProfileSummaryFormatter(bool format_as_csv) const;

 private:
  void WriteOutput(const std::string& header, const string& data,
                   std::ostream* stream);
  Interpreter* interpreter_;
  profiling::BufferedProfiler profiler_;
  profiling::ProfileSummarizer run_summarizer_;
  profiling::ProfileSummarizer init_summarizer_;
  std::string csv_file_path_;
};

}  // namespace benchmark
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_BENCHMARK_PROFILING_LISTENER_H_
