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
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>

#include "tensorflow/core/util/stats_calculator.h"
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/benchmark/logging.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace tflite {
namespace benchmark {

void MultiRunStatsRecorder::OnBenchmarkStart(const BenchmarkParams& params) {
  current_run_name_.clear();

  if (params.Get<bool>("use_nnapi")) {
    current_run_name_ = "nnapi";
    return;
  }

  if (params.Get<bool>("use_gpu")) {
    current_run_name_ = "gpu";
    return;
  }

  // Handle cases run on CPU
  // Note: could use std::to_string to convert an integer to string but it
  // requires C++11.
  std::stringstream sstm;
  sstm << "cpu w/ " << params.Get<int32_t>("num_threads") << " threads";
  current_run_name_ = sstm.str();
}

void MultiRunStatsRecorder::OnBenchmarkEnd(const BenchmarkResults& results) {
  each_run_stats_.emplace_back(std::make_pair(current_run_name_, results));
}

void MultiRunStatsRecorder::OutputStats() {
  // Make a 80-character-long header.
  TFLITE_LOG(INFO) << "\n==============Summary of All Runs w/ Different "
                      "Performance Options==============";
  std::sort(each_run_stats_.begin(), each_run_stats_.end(),
            EachRunStatsEntryComparator());

  for (const auto& run_stats : each_run_stats_) {
    std::stringstream stream;
    // Output the name of this run first.
    stream << std::setw(26) << run_stats.first << ": ";
    run_stats.second.inference_time_us().OutputToStream(&stream);
    TFLITE_LOG(INFO) << stream.str();
  }
}

BenchmarkPerformanceOptions::BenchmarkPerformanceOptions(
    BenchmarkModel* single_option_run)
    : BenchmarkPerformanceOptions(DefaultParams(), single_option_run,
                                  DefaultRunStatsRecorder()) {}

BenchmarkPerformanceOptions::BenchmarkPerformanceOptions(
    BenchmarkParams params, BenchmarkModel* single_option_run,
    std::unique_ptr<MultiRunStatsRecorder> all_run_stats)
    : params_(std::move(params)),
      single_option_run_(single_option_run),
      single_option_run_params_(single_option_run->mutable_params()),
      all_run_stats_(std::move(all_run_stats)) {
  single_option_run_->AddListener(all_run_stats_.get());
}

BenchmarkParams BenchmarkPerformanceOptions::DefaultParams() {
  BenchmarkParams params;
  params.AddParam("perf_options_list",
                  BenchmarkParam::Create<std::string>("all"));
  params.AddParam("option_benchmark_run_delay",
                  BenchmarkParam::Create<float>(-1.0f));
  return params;
}

std::unique_ptr<MultiRunStatsRecorder>
BenchmarkPerformanceOptions::DefaultRunStatsRecorder() {
  return std::unique_ptr<MultiRunStatsRecorder>(new MultiRunStatsRecorder());
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

bool BenchmarkPerformanceOptions::HasOption(const std::string& option) const {
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

  // Now, the remaining are unrecognized flags and we simply print them out.
  for (int i = 1; i < argc; ++i) {
    TFLITE_LOG(WARN) << "WARNING: unrecognized commandline flag: " << argv[i];
  }

  Run();

  all_run_stats_->OutputStats();
}

void BenchmarkPerformanceOptions::Run() {
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
