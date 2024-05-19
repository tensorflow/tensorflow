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
#include <string>

#include "tensorflow/lite/profiling/profile_summarizer.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace benchmark {

ProfilingListener::ProfilingListener(
    Interpreter* interpreter, uint32_t max_num_initial_entries,
    bool allow_dynamic_buffer_increase, const std::string& output_file_path,
    std::shared_ptr<profiling::ProfileSummaryFormatter> summarizer_formatter)
    : run_summarizer_(summarizer_formatter),
      init_summarizer_(summarizer_formatter),
      output_file_path_(output_file_path),
      interpreter_(interpreter),
      profiler_(max_num_initial_entries, allow_dynamic_buffer_increase),
      summarizer_formatter_(summarizer_formatter) {
  TFLITE_TOOLS_CHECK(interpreter);
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
  summarizer_formatter_->HandleOutput(init_summarizer_.GetOutputString(),
                                      run_summarizer_.GetOutputString(),
                                      output_file_path_);
}

}  // namespace benchmark
}  // namespace tflite
