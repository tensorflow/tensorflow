/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <errno.h>
#include <sys/stat.h>

#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"

namespace tflite {
namespace benchmark {
namespace latency {
namespace {

bool CreateDir(const char* path) {
  struct stat st;
  if (stat(path, &st) != 0) {
    if (mkdir(path, 0777) != 0 && errno != EEXIST) {
      return false;
    }
  } else if (!S_ISDIR(st.st_mode)) {
    errno = ENOTDIR;
    return false;
  }
  return true;
}

// The listener subscribes to the benchmark lifecycle and outputs the success
// status of the benchmark to a local json file.
class DelegatePerformanceReportingListener : public BenchmarkListener {
 public:
  // Generates `report.json` for the success status of a benchmark run under
  // `result_path` folder.
  explicit DelegatePerformanceReportingListener(const char* result_path)
      : result_path_(result_path) {
    if (!result_path) {
      TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                      "Report will be be streamed only to local log not to a "
                      "file since the result path is null.");
    }
  }

  // TFLite Benchmark Tool triggers this method at the end of a benchmark for
  // logging the results.
  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    ReportResult(results);
  }

  void ReportFailure(TfLiteStatus status) {
    std::string status_msg =
        status == kTfLiteError
            ? "TFLite error"
            : (status == kTfLiteDelegateError ? "TFLite delegate error"
                                              : "Unknown error code");
    Report(status_msg, std::vector<std::pair<std::string, std::string>>());
  }

 private:
  void Report(
      const std::string& status,
      const std::vector<std::pair<std::string, std::string>>& contents) {
    std::string filename = result_path_ + "/report.json";
    std::ofstream file;
    file.open(filename.c_str());
    if (!file.is_open()) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to open file %s",
                      filename.c_str());
      return;
    }
    std::stringstream report;
    report << "{\n"
           << "  \"name\": \"TFLite benchmark\",\n"
           << "  \"status\": \"" << status << "\"";
    for (const auto& content : contents) {
      report << ",\n"
             << "  \"" << content.first << "\": \"" << content.second << "\"";
    }
    report << "\n}\n";

    auto report_str = report.str();
    file << report_str;
    file.close();

    TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "%s", report_str.c_str());
  }

  void ReportResult(const BenchmarkResults& results) {
    std::vector<std::pair<std::string, std::string>> contents;
    std::stringstream avg_time;
    avg_time << "init: " << results.startup_latency_us() << ", "
             << "warmup: " << results.warmup_time_us().avg() << ", "
             << "inference: " << results.inference_time_us().avg();
    contents.emplace_back("average time in us", avg_time.str());
    std::stringstream overall_mem_usage;
    overall_mem_usage << results.overall_mem_usage();
    contents.emplace_back("overall memory usage", overall_mem_usage.str());

    Report("OK", contents);
  }

  // Root of output path for intermediate results and data.
  std::string result_path_;
};

// TODO(b/250877013): expose the results for performance thresholding.
// `CsvExportingListener` subscribes to the benchmark lifecycle and outputs the
// results of a benchmark run to local in csv format.
class CsvExportingListener : public BenchmarkListener {
 public:
  // Generates `benchmark_result.csv` for performance results of a benchmark run
  // under `result_path` folder.
  explicit CsvExportingListener(const char* result_path)
      : result_path_(result_path) {}

  // TFLite Benchmark Tool triggers this method at the end of a benchmark for
  // logging the results.
  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    WriteBenchmarkResultCsv(results);
  }

 private:
  void WriteBenchmarkResultCsv(const BenchmarkResults& results) {
    auto init_us = results.startup_latency_us();
    auto warmup_us = results.warmup_time_us();
    auto inference_us = results.inference_time_us();
    auto init_mem_usage = results.init_mem_usage();
    auto overall_mem_usage = results.overall_mem_usage();

    std::string filename = result_path_ + "/benchmark_result.csv";
    std::ofstream file;
    file.open(filename.c_str());
    if (!file.is_open()) {
      TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to open file %s",
                      filename.c_str());
      return;
    }

    file << "model_size,init_time,"
         << "warmup_avg,warmup_min,warmup_max,warmup_stddev,"
         << "inference_avg,inference_min,inference_max,inference_stddev,"
         << "init_max_rss,init_total_alloc,init_in_use_alloc,"
         << "overall_max_rss,overall_total_alloc,overall_in_use_alloc\n";
    file << results.model_size_mb() << "," << init_us << "," << warmup_us.avg()
         << "," << warmup_us.min() << "," << warmup_us.max() << ","
         << warmup_us.std_deviation() << "," << inference_us.avg() << ","
         << inference_us.min() << "," << inference_us.max() << ","
         << inference_us.std_deviation() << ","
         << (init_mem_usage.mem_footprint_kb / 1024.0) << ","
         << (init_mem_usage.total_allocated_bytes / 1024.0 / 1024.0) << ","
         << (init_mem_usage.in_use_allocated_bytes / 1024.0 / 1024.0) << ","
         << (overall_mem_usage.mem_footprint_kb / 1024.0) << ","
         << (overall_mem_usage.total_allocated_bytes / 1024.0 / 1024.0) << ","
         << (overall_mem_usage.in_use_allocated_bytes / 1024.0 / 1024.0)
         << "\n";
    file.close();
  }

  // Root of output path for intermediate results and data.
  std::string result_path_;
};

}  // namespace

void Benchmark(const std::vector<std::string>& args, const char* result_path) {
  // Constructs a fake argv command-line object for the benchmark.
  std::vector<char*> argv;
  std::string arg0 = "(BenchmarkModelAndroid)";
  argv.push_back(const_cast<char*>(arg0.data()));
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.data()));
  }

  // Create directory `result_path` if it doesn't already exist.
  if (!CreateDir(result_path)) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to create output directory %s.",
                    result_path);
    return;
  }

  BenchmarkTfLiteModel benchmark;
  // Generates general benchmark status JSON report.
  DelegatePerformanceReportingListener delegatePerformanceReporting(
      result_path);
  benchmark.AddListener(&delegatePerformanceReporting);
  // Generates performance benchmark result CSV report.
  CsvExportingListener csvExporting(result_path);
  benchmark.AddListener(&csvExporting);
  auto status = benchmark.Run(static_cast<int>(argv.size()), argv.data());
  if (status != kTfLiteOk) {
    delegatePerformanceReporting.ReportFailure(status);
  }
}

}  // namespace latency
}  // namespace benchmark
}  // namespace tflite
