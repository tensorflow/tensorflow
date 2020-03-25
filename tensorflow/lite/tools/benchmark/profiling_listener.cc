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

#include "tensorflow/lite/tools/benchmark/profiling_listener.h"

#include <fstream>

namespace tflite {
namespace benchmark {

ProfilingListener::ProfilingListener(
    Interpreter* interpreter, uint32_t max_num_entries,
    const std::string& csv_file_path,
    std::shared_ptr<profiling::ProfileSummaryFormatter> summarizer_formatter)
    : run_summarizer_(summarizer_formatter),
      init_summarizer_(summarizer_formatter),
      csv_file_path_(csv_file_path),
      interpreter_(interpreter),
      profiler_(max_num_entries) {
  TFLITE_BENCHMARK_CHECK(interpreter);
  interpreter_->SetProfiler(&profiler_);

  // We start profiling here in order to catch events that are recorded during
  // the benchmark run preparation stage where TFLite interpreter is
  // initialized and model graph is prepared.
  profiler_.Reset();
  profiler_.StartProfiling();
}

void ProfilingListener::OnBenchmarkStart(const BenchmarkParams& params) {
  // At this point, we have completed the preparation for benchmark runs
  // including TFLite interpreter initialization etc. So we are going to process
  // profiling events recorded during this stage.
  profiler_.StopProfiling();
  auto profile_events = profiler_.GetProfileEvents();
  init_summarizer_.ProcessProfiles(profile_events, *interpreter_);
  profiler_.Reset();
}

void ProfilingListener::OnSingleRunStart(RunType run_type) {
  if (run_type == REGULAR) {
    profiler_.Reset();
    profiler_.StartProfiling();
  }
}

void ProfilingListener::OnSingleRunEnd() {
  profiler_.StopProfiling();
  auto profile_events = profiler_.GetProfileEvents();
  run_summarizer_.ProcessProfiles(profile_events, *interpreter_);
}

void ProfilingListener::OnBenchmarkEnd(const BenchmarkResults& results) {
  std::ofstream output_file(csv_file_path_);
  std::ostream* output_stream = nullptr;
  if (output_file.good()) {
    output_stream = &output_file;
  }
  if (init_summarizer_.HasProfiles()) {
    WriteOutput("Profiling Info for Benchmark Initialization:",
                init_summarizer_.GetOutputString(),
                output_stream == nullptr ? &TFLITE_LOG(INFO) : output_stream);
  }
  if (run_summarizer_.HasProfiles()) {
    WriteOutput("Operator-wise Profiling Info for Regular Benchmark Runs:",
                run_summarizer_.GetOutputString(),
                output_stream == nullptr ? &TFLITE_LOG(INFO) : output_stream);
  }
}

void ProfilingListener::WriteOutput(const std::string& header,
                                    const string& data, std::ostream* stream) {
  (*stream) << header << std::endl;
  (*stream) << data << std::endl;
}

}  // namespace benchmark
}  // namespace tflite
