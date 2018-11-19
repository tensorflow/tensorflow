/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/benchmark/benchmark_model.h"

#include <time.h>

#include <iostream>
#include <sstream>

#include "tensorflow/lite/profiling/time.h"
#include "tensorflow/lite/tools/benchmark/logging.h"

namespace {
void SleepForSeconds(double sleep_seconds) {
  if (sleep_seconds <= 0.0) {
    return;
  }
  // Convert the run_delay string into a timespec.
  timespec req;
  req.tv_sec = static_cast<time_t>(sleep_seconds);
  req.tv_nsec = (sleep_seconds - req.tv_sec) * 1000000000;
  // If requested, sleep between runs for an arbitrary amount of time.
  // This can be helpful to determine the effect of mobile processor
  // scaling and thermal throttling.
#ifdef PLATFORM_WINDOWS
  Sleep(sleep_seconds * 1000);
#else
  nanosleep(&req, nullptr);
#endif
}

}  // namespace

namespace tflite {
namespace benchmark {
using tensorflow::Stat;

BenchmarkParams BenchmarkModel::DefaultParams() {
  BenchmarkParams params;
  params.AddParam("num_runs", BenchmarkParam::Create<int32_t>(50));
  params.AddParam("run_delay", BenchmarkParam::Create<float>(-1.0f));
  params.AddParam("num_threads", BenchmarkParam::Create<int32_t>(1));
  params.AddParam("benchmark_name", BenchmarkParam::Create<std::string>(""));
  params.AddParam("output_prefix", BenchmarkParam::Create<std::string>(""));
  params.AddParam("warmup_runs", BenchmarkParam::Create<int32_t>(1));
  return params;
}

BenchmarkModel::BenchmarkModel() : params_(DefaultParams()) {}

void BenchmarkLoggingListener::OnBenchmarkEnd(const BenchmarkResults &results) {
  auto inference_us = results.inference_time_us();
  auto init_us = results.startup_latency_us();
  auto warmup_us = results.warmup_time_us();
  TFLITE_LOG(INFO) << "Average inference timings in us: "
                   << "Warmup: " << warmup_us.avg() << ", "
                   << "Init: " << init_us << ", "
                   << "no stats: " << inference_us.avg();
}

std::vector<Flag> BenchmarkModel::GetFlags() {
  return {
      CreateFlag<int32_t>("num_runs", &params_, "number of runs"),
      CreateFlag<float>("run_delay", &params_, "delay between runs in seconds"),
      CreateFlag<int32_t>("num_threads", &params_, "number of threads"),
      CreateFlag<std::string>("benchmark_name", &params_, "benchmark name"),
      CreateFlag<std::string>("output_prefix", &params_,
                              "benchmark output prefix"),
      CreateFlag<int32_t>("warmup_runs", &params_,
                          "how many runs to initialize model"),
  };
}

void BenchmarkModel::LogParams() {
  TFLITE_LOG(INFO) << "Num runs: [" << params_.Get<int32_t>("num_runs") << "]";
  TFLITE_LOG(INFO) << "Inter-run delay (seconds): ["
                   << params_.Get<float>("run_delay") << "]";
  TFLITE_LOG(INFO) << "Num threads: [" << params_.Get<int32_t>("num_threads")
                   << "]";
  TFLITE_LOG(INFO) << "Benchmark name: ["
                   << params_.Get<std::string>("benchmark_name") << "]";
  TFLITE_LOG(INFO) << "Output prefix: ["
                   << params_.Get<std::string>("output_prefix") << "]";
  TFLITE_LOG(INFO) << "Warmup runs: [" << params_.Get<int32_t>("warmup_runs")
                   << "]";
}

void BenchmarkModel::PrepareInputsAndOutputs() {}

Stat<int64_t> BenchmarkModel::Run(int num_times, RunType run_type) {
  Stat<int64_t> run_stats;
  TFLITE_LOG(INFO) << "Running benchmark for " << num_times << " iterations ";
  for (int run = 0; run < num_times; run++) {
    PrepareInputsAndOutputs();
    listeners_.OnSingleRunStart(run_type);
    int64_t start_us = profiling::time::NowMicros();
    RunImpl();
    int64_t end_us = profiling::time::NowMicros();
    listeners_.OnSingleRunEnd();

    run_stats.UpdateStat(end_us - start_us);
    SleepForSeconds(params_.Get<float>("run_delay"));
  }

  std::stringstream stream;
  run_stats.OutputToStream(&stream);
  TFLITE_LOG(INFO) << stream.str() << std::endl;

  return run_stats;
}

bool BenchmarkModel::ValidateParams() { return true; }

void BenchmarkModel::Run(int argc, char **argv) {
  if (!ParseFlags(argc, argv)) {
    return;
  }
  Run();
}

void BenchmarkModel::Run() {
  ValidateParams();
  LogParams();

  listeners_.OnBenchmarkStart(params_);
  int64_t initialization_start_us = profiling::time::NowMicros();
  Init();
  int64_t initialization_end_us = profiling::time::NowMicros();
  int64_t startup_latency_us = initialization_end_us - initialization_start_us;
  TFLITE_LOG(INFO) << "Initialized session in " << startup_latency_us / 1e3
                   << "ms";

  uint64_t input_bytes = ComputeInputBytes();
  Stat<int64_t> warmup_time_us =
      Run(params_.Get<int32_t>("warmup_runs"), WARMUP);
  Stat<int64_t> inference_time_us =
      Run(params_.Get<int32_t>("num_runs"), REGULAR);
  listeners_.OnBenchmarkEnd(
      {startup_latency_us, input_bytes, warmup_time_us, inference_time_us});
}

bool BenchmarkModel::ParseFlags(int argc, char **argv) {
  auto flag_list = GetFlags();
  const bool parse_result =
      Flags::Parse(&argc, const_cast<const char **>(argv), flag_list);
  if (!parse_result) {
    std::string usage = Flags::Usage(argv[0], flag_list);
    TFLITE_LOG(ERROR) << usage;
    return false;
  }
  return true;
}

}  // namespace benchmark
}  // namespace tflite
