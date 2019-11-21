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
#include "tensorflow/lite/c/c_api_internal.h"
#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"
#endif
#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/benchmark/benchmark_params.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/benchmark/logging.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace tflite {
namespace benchmark {

void MultiRunStatsRecorder::OnBenchmarkStart(const BenchmarkParams& params) {
  current_run_name_.clear();

#if defined(__ANDROID__)
  if (params.Get<bool>("use_nnapi")) {
    const std::string accelerator =
        params.Get<std::string>("nnapi_accelerator_name");
    current_run_name_ = accelerator.empty() ? "nnapi(w/o accel name)"
                                            : "nnapi(" + accelerator + ")";
    return;
  }
#endif

  if (params.Get<bool>("use_gpu")) {
#if defined(__ANDROID__)
    const bool allow_precision_loss =
        params.Get<bool>("gpu_precision_loss_allowed");
    const std::string precision_tag = allow_precision_loss ? "fp16" : "fp32";
    current_run_name_ = "gpu(" + precision_tag + ")";

    const auto default_opts = TfLiteGpuDelegateOptionsV2Default();
    if (default_opts.is_precision_loss_allowed == allow_precision_loss) {
      current_run_name_ += "-default";
    }
#else
    current_run_name_ = "gpu-default";
#endif
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
    // NOTE: As of 2019/11/07, the memory usage is collected in an
    // OS-process-wide way and this program performs multiple runs in a single
    // OS process, therefore, the memory usage information of each run becomes
    // incorrect, hence no output here.
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
  params.AddParam("random_shuffle_benchmark_runs",
                  BenchmarkParam::Create<bool>(true));
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
      CreateFlag<bool>(
          "random_shuffle_benchmark_runs", &params_,
          "Whether to perform all benchmark runs, each of which has different "
          "performance options, in a random order. It is enabled by default."),
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
#if defined(__ANDROID__)
  single_option_run_params_->Set<bool>("gpu_precision_loss_allowed", true);
  single_option_run_params_->Set<bool>("use_nnapi", false);
  single_option_run_params_->Set<std::string>("nnapi_accelerator_name", "");
#endif
}

void BenchmarkPerformanceOptions::CreatePerformanceOptions() {
  TFLITE_LOG(INFO) << "The list of TFLite runtime options to be benchmarked: ["
                   << params_.Get<std::string>("perf_options_list") << "]";

  const bool benchmark_all = HasOption("all");

  if (benchmark_all || HasOption("cpu")) {
    const std::vector<int> num_threads = {1, 2, 4};
    for (const int count : num_threads) {
      BenchmarkParams params;
      params.AddParam("num_threads", BenchmarkParam::Create<int32_t>(count));
      all_run_params_.emplace_back(std::move(params));
    }
  }

  if (benchmark_all || HasOption("gpu")) {
#if defined(__ANDROID__)
    const std::vector<bool> allow_precision_loss = {true, false};
    for (const auto precision_loss : allow_precision_loss) {
      BenchmarkParams params;
      params.AddParam("use_gpu", BenchmarkParam::Create<bool>(true));
      params.AddParam("gpu_precision_loss_allowed",
                      BenchmarkParam::Create<bool>(precision_loss));
      all_run_params_.emplace_back(std::move(params));
    }
#else
    BenchmarkParams params;
    params.AddParam("use_gpu", BenchmarkParam::Create<bool>(true));
    all_run_params_.emplace_back(std::move(params));
#endif
  }

#if defined(__ANDROID__)
  if (benchmark_all || HasOption("nnapi")) {
    std::string nnapi_accelerators = nnapi::GetStringDeviceNamesList();
    if (!nnapi_accelerators.empty()) {
      std::vector<std::string> device_names;
      util::SplitAndParse(nnapi_accelerators, ',', &device_names);
      for (const auto name : device_names) {
        BenchmarkParams params;
        params.AddParam("use_nnapi", BenchmarkParam::Create<bool>(true));
        params.AddParam("nnapi_accelerator_name",
                        BenchmarkParam::Create<std::string>(name));
        all_run_params_.emplace_back(std::move(params));
      }
    }
    // Explicitly test the case when there's no "nnapi_accelerator_name"
    // parameter as the nnpai execution is different from the case when
    // an accelerator name is explicitly specified.
    BenchmarkParams params;
    params.AddParam("use_nnapi", BenchmarkParam::Create<bool>(true));
    all_run_params_.emplace_back(std::move(params));
  }
#endif
}

void BenchmarkPerformanceOptions::Run(int argc, char** argv) {
  // We first parse flags for single-option runs to get information like
  // parameters of the input model etc.
  if (single_option_run_->ParseFlags(&argc, argv) != kTfLiteOk) return;

  // Now, we parse flags that are specified for this particular binary.
  if (!ParseFlags(&argc, argv)) return;

  // Now, the remaining are unrecognized flags and we simply print them out.
  for (int i = 1; i < argc; ++i) {
    TFLITE_LOG(WARN) << "WARNING: unrecognized commandline flag: " << argv[i];
  }

  CreatePerformanceOptions();

  if (params_.Get<bool>("random_shuffle_benchmark_runs")) {
    std::random_shuffle(all_run_params_.begin(), all_run_params_.end());
  }

  // Now perform all runs, each with different performance-affecting parameters.
  for (const auto& run_params : all_run_params_) {
    // Reset all performance-related options before any runs.
    ResetPerformanceOptions();
    single_option_run_params_->Set(run_params);
    util::SleepForSeconds(params_.Get<float>("option_benchmark_run_delay"));
    single_option_run_->Run();
  }

  all_run_stats_->OutputStats();
}
}  // namespace benchmark
}  // namespace tflite
