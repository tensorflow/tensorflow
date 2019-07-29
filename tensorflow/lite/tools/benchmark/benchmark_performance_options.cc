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

#include "tensorflow/lite/tools/benchmark/benchmark_performance_options.h"

#include <algorithm>

#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/benchmark/logging.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace tflite {
namespace benchmark {

BenchmarkParams BenchmarkPerformanceOptions::DefaultParams() {
  BenchmarkParams params;
  params.AddParam("perf_options_list",
                  BenchmarkParam::Create<std::string>("all"));
  params.AddParam("option_benchmark_run_delay",
                  BenchmarkParam::Create<float>(-1.0f));
  return params;
}

std::vector<Flag> BenchmarkPerformanceOptions::GetFlags() {
  return {
      CreateFlag<std::string>(
          "perf_options_list", &params_,
          "A comma-separated list of TFLite performance options to benchmark. "
          "By default, all performance options are benchmarked."),
      CreateFlag<float>("option_benchmark_run_delay", &params_,
                        "The delay between two consecutive runs of "
                        "benchmarking performance options in seconds."),
  };
}

bool BenchmarkPerformanceOptions::ParseFlags(int* argc, char** argv) {
  auto flag_list = GetFlags();
  const bool parse_result =
      Flags::Parse(argc, const_cast<const char**>(argv), flag_list);
  if (!parse_result) {
    std::string usage = Flags::Usage(argv[0], flag_list);
    TFLITE_LOG(ERROR) << usage;
    return false;
  }

  // Parse the value of --perf_options_list to find performance options to be
  // benchmarked.
  return ParsePerfOptions();
}

bool BenchmarkPerformanceOptions::ParsePerfOptions() {
  const auto& perf_options_list = params_.Get<std::string>("perf_options_list");
  if (!util::SplitAndParse(perf_options_list, ',', &perf_options_)) {
    TFLITE_LOG(ERROR) << "Cannot parse --perf_options_list: '"
                      << perf_options_list
                      << "'. Please double-check its value.";
    perf_options_.clear();
    return false;
  }

  const auto valid_options = GetValidPerfOptions();
  bool is_valid = true;
  for (const auto& option : perf_options_) {
    if (std::find(valid_options.begin(), valid_options.end(), option) ==
        valid_options.end()) {
      is_valid = false;
      break;
    }
  }
  if (!is_valid) {
    std::string valid_options_str;
    for (int i = 0; i < valid_options.size() - 1; ++i) {
      valid_options_str += (valid_options[i] + ", ");
    }
    valid_options_str += valid_options.back();
    TFLITE_LOG(ERROR)
        << "There are invalid perf options in --perf_options_list: '"
        << perf_options_list << "'. Valid perf options are: ["
        << valid_options_str << "]";
    perf_options_.clear();
    return false;
  }
  return true;
}

std::vector<std::string> BenchmarkPerformanceOptions::GetValidPerfOptions()
    const {
  return {"all", "cpu", "gpu", "nnapi"};
}

bool BenchmarkPerformanceOptions::HasOption(const string& option) const {
  return std::find(perf_options_.begin(), perf_options_.end(), option) !=
         perf_options_.end();
}

void BenchmarkPerformanceOptions::ResetPerformanceOptions() {
  single_option_run_params_->Set<int32_t>("num_threads", 1);
  single_option_run_params_->Set<bool>("use_gpu", false);
  single_option_run_params_->Set<bool>("use_nnapi", false);
}

void BenchmarkPerformanceOptions::BenchmarkCPUOptions() {
  // Reset all performance-related options before any runs.
  ResetPerformanceOptions();

  const int num_threads[] = {1, 2, 4};
  for (int i = 0; i < sizeof(num_threads) / sizeof(int); ++i) {
    single_option_run_params_->Set<int32_t>("num_threads", num_threads[i]);
    util::SleepForSeconds(params_.Get<float>("option_benchmark_run_delay"));
    single_option_run_->Run();
  }
}

void BenchmarkPerformanceOptions::BenchmarkGPUOptions() {
  // Reset all performance-related options before any runs.
  ResetPerformanceOptions();

  single_option_run_params_->Set<bool>("use_gpu", true);
  util::SleepForSeconds(params_.Get<float>("option_benchmark_run_delay"));
  single_option_run_->Run();
}

void BenchmarkPerformanceOptions::BenchmarkNnapiOptions() {
  // Reset all performance-related options before any runs.
  ResetPerformanceOptions();

  single_option_run_params_->Set<bool>("use_nnapi", true);
  util::SleepForSeconds(params_.Get<float>("option_benchmark_run_delay"));
  single_option_run_->Run();
}

void BenchmarkPerformanceOptions::Run(int argc, char** argv) {
  // We first parse flags for single-option runs to get information like
  // parameters of the input model etc.
  if (!single_option_run_->ParseFlags(&argc, argv)) {
    return;
  }

  // Now, we parse flags that are specified for this particular binary.
  if (!ParseFlags(&argc, argv)) {
    return;
  }

  TFLITE_LOG(INFO) << "The list of TFLite runtime options to be benchmarked: ["
                   << params_.Get<std::string>("perf_options_list") << "]";

  const bool benchmark_all = HasOption("all");
  if (benchmark_all || HasOption("cpu")) {
    BenchmarkCPUOptions();
  }

  if (benchmark_all || HasOption("gpu")) {
    BenchmarkGPUOptions();
  }

  if (benchmark_all || HasOption("nnapi")) {
    BenchmarkNnapiOptions();
  }
}

}  // namespace benchmark
}  // namespace tflite
