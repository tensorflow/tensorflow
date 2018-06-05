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

#include "tensorflow/contrib/lite/tools/benchmark_model.h"

#include <time.h>

#include <iostream>
#include <sstream>

#include "tensorflow/contrib/lite/profiling/time.h"
#include "tensorflow/contrib/lite/tools/logging.h"

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
      Flag("num_runs", &params_.num_runs, "number of runs"),
      Flag("run_delay", &params_.run_delay, "delay between runs in seconds"),
      Flag("num_threads", &params_.num_threads, "number of threads"),
      Flag("benchmark_name", &params_.benchmark_name, "benchmark name"),
      Flag("output_prefix", &params_.output_prefix, "benchmark output prefix"),
      Flag("warmup_runs", &params_.warmup_runs,
           "how many runs to initialize model"),
  };
}

void BenchmarkModel::LogFlags() {
  TFLITE_LOG(INFO) << "Num runs: [" << params_.num_runs << "]";
  TFLITE_LOG(INFO) << "Inter-run delay (seconds): [" << params_.run_delay
                   << "]";
  TFLITE_LOG(INFO) << "Num threads: [" << params_.num_threads << "]";
  TFLITE_LOG(INFO) << "Benchmark name: [" << params_.benchmark_name << "]";
  TFLITE_LOG(INFO) << "Output prefix: [" << params_.output_prefix << "]";
  TFLITE_LOG(INFO) << "Warmup runs: [" << params_.warmup_runs << "]";
}

Stat<int64_t> BenchmarkModel::Run(int num_times, RunType run_type) {
  Stat<int64_t> run_stats;
  TFLITE_LOG(INFO) << "Running benchmark for " << num_times << " iterations ";
  for (int run = 0; run < num_times; run++) {
    listeners_.OnSingleRunStart(run_type);
    int64_t start_us = profiling::time::NowMicros();
    RunImpl();
    int64_t end_us = profiling::time::NowMicros();
    listeners_.OnSingleRunEnd();

    run_stats.UpdateStat(end_us - start_us);
    SleepForSeconds(params_.run_delay);
  }

  std::stringstream stream;
  run_stats.OutputToStream(&stream);
  TFLITE_LOG(INFO) << stream.str() << std::endl;

  return run_stats;
}

void BenchmarkModel::Run(int argc, char **argv) {
  if (!ParseFlags(argc, argv)) {
    return;
  }

  LogFlags();

  listeners_.OnBenchmarkStart(params_);
  int64_t initialization_start_us = profiling::time::NowMicros();
  Init();
  int64_t initialization_end_us = profiling::time::NowMicros();
  int64_t startup_latency_us = initialization_end_us - initialization_start_us;
  TFLITE_LOG(INFO) << "Initialized session in " << startup_latency_us / 1e3
                   << "ms";

  uint64_t input_bytes = ComputeInputBytes();
  Stat<int64_t> warmup_time_us = Run(params_.warmup_runs, WARMUP);
  Stat<int64_t> inference_time_us = Run(params_.num_runs, REGULAR);
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
  return ValidateFlags();
}

}  // namespace benchmark
}  // namespace tflite
