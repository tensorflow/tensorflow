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

#include <errno.h>
#include <jni.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace tflite {
namespace benchmark {
namespace {

const char kOutputDir[] = "/sdcard/benchmark_output";
const char kSerializeDir[] = "/sdcard/serialize";

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

class FirebaseReportingListener : public BenchmarkListener {
 public:
  explicit FirebaseReportingListener(std::string tag, int report_fd)
      : tag_(tag), report_fd_(report_fd) {
    if (report_fd < 0) {
#ifdef __ANDROID__
      __android_log_print(
          ANDROID_LOG_ERROR, "tflite",
          "Report would be streamed only to local log not to Firebase "
          "since the Firebase log file is not opened.");
#else
      fprintf(stderr,
              "Report would be streamed only to local log not to Firebase "
              "since the Firebase log file is not opened.");
#endif
    }
  }

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
    // The output format of Firebase Game Loop test is json.
    // https://firebase.google.com/docs/test-lab/android/game-loop#output-example
    std::stringstream report;
    report << "{\n"
           << "  \"name\": \"TFLite benchmark\",\n"
           << "  \"benchmark config\": \"" << tag_ << "\",\n"
           << "  \"status\": \"" << status << "\"";
    for (const auto& content : contents) {
      report << ",\n"
             << "  \"" << content.first << "\": \"" << content.second << "\"";
    }
    report << "\n}\n";

    auto report_str = report.str();
    if (report_fd_ >= 0) {
      write(report_fd_, report_str.c_str(), report_str.size());
    }

#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_ERROR, "tflite", "%s", report_str.c_str());
#else
    fprintf(stderr, "%s", report_str.c_str());
#endif
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

  std::string tag_;
  int report_fd_;
};

class CsvExportingListener : public BenchmarkListener {
 public:
  explicit CsvExportingListener(std::string tag) : tag_(tag) {}

  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    if (!CreateDir(kOutputDir)) {
#ifdef __ANDROID__
      __android_log_print(ANDROID_LOG_ERROR, "tflite",
                          "Failed to create output directory %s.", kOutputDir);
#else
      fprintf(stderr, "Failed to create output directory %s.", kOutputDir);
#endif
      return;
    }
    WriteBenchmarkResultCsv(results);
  }

 private:
  void WriteBenchmarkResultCsv(const BenchmarkResults& results) {
    auto init_us = results.startup_latency_us();
    auto warmup_us = results.warmup_time_us();
    auto inference_us = results.inference_time_us();
    auto init_mem_usage = results.init_mem_usage();
    auto overall_mem_usage = results.overall_mem_usage();

    std::stringstream file_name;
    file_name << kOutputDir << "/benchmark_result_" << tag_;

    std::ofstream file;
    file.open(file_name.str().c_str());
    file << "config_key,model_size,init_time,"
         << "warmup_avg,warmup_min,warmup_max,warmup_stddev,"
         << "inference_avg,inference_min,inference_max,inference_stddev,"
         << "init_max_rss,init_total_alloc,init_in_use_alloc,"
         << "overall_max_rss,overall_total_alloc,overall_in_use_alloc\n";
    file << tag_ << "," << results.model_size_mb() << "," << init_us << ","
         << warmup_us.avg() << "," << warmup_us.min() << "," << warmup_us.max()
         << "," << warmup_us.std_deviation() << "," << inference_us.avg() << ","
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

  std::string tag_;
};

std::string GetScenarioConfig(const std::string& library_dir, int scenario,
                              std::vector<std::string>& args) {
  // The number of scenarios should equal to the value specified in
  // AndroidManifest.xml file.
  std::unordered_map<int, std::pair<std::string, std::vector<std::string>>>
      all_scenarios = {
          {1, {"cpu_1thread", {"--num_threads=1"}}},
          {2, {"cpu_2threads", {"--num_threads=2"}}},
          {3, {"cpu_4threads", {"--num_threads=4"}}},
          {4, {"xnnpack_1thread", {"--use_xnnpack=true", "--num_threads=1"}}},
          {5, {"xnnpack_2threads", {"--use_xnnpack=true", "--num_threads=2"}}},
          {6, {"xnnpack_4threads", {"--use_xnnpack=true", "--num_threads=4"}}},
          {7,
           {"gpu_default",
            {"--use_gpu=true", "--gpu_precision_loss_allowed=false"}}},
          {8,
           {"gpu_fp16",
            {"--use_gpu=true", "--gpu_precision_loss_allowed=true"}}},
          {9, {"dsp_hexagon", {"--use_hexagon=true"}}},
          {10, {"nnapi", {"--use_nnapi=true"}}},
          {11,
           {"gpu_default_with_serialization",
            {"--use_gpu=true", "--gpu_precision_loss_allowed=false",
             "--delegate_serialize_token=dummy_token"}}},
          {12,
           {"gpu_fp16_with_serialization",
            {"--use_gpu=true", "--gpu_precision_loss_allowed=true",
             "--delegate_serialize_token=dummy_token"}}},
      };

  std::string tag;
  args.emplace_back("(BenchmarkModelAndroid)");
  args.emplace_back("--graph=/data/local/tmp/graph");

  auto it = all_scenarios.find(scenario);
  if (it != all_scenarios.end()) {
    const auto& scenario_info = it->second;
    tag = scenario_info.first;
    for (const auto& arg : scenario_info.second) {
      args.push_back(arg);
    }
  }
  if (scenario == 9) {
    std::stringstream hexagon_lib_path;
    hexagon_lib_path << "--hexagon_lib_path=" << library_dir;
    args.push_back(hexagon_lib_path.str());
  }

  if (scenario == 11 || scenario == 12) {
    if (CreateDir(kSerializeDir)) {
      std::stringstream serialize_dir;
      serialize_dir << "--delegate_serialize_dir=" << kSerializeDir;
      args.push_back(serialize_dir.str());
    } else {
#ifdef __ANDROID__
      __android_log_print(ANDROID_LOG_ERROR, "tflite",
                          "Failed to create serialize directory %s.",
                          kSerializeDir);
#else
      fprintf(stderr, "Failed to create serialize directory %s.",
              kSerializeDir);
#endif
    }
  }
  return tag;
}

void RunScenario(const std::string& library_dir, int scenario, int report_fd) {
  std::vector<std::string> args;
  std::string tag = GetScenarioConfig(library_dir, scenario, args);
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.data()));
  }

  BenchmarkTfLiteModel benchmark;
  FirebaseReportingListener firebaseReporting(tag, report_fd);
  benchmark.AddListener(&firebaseReporting);
  CsvExportingListener csvExporting(tag);
  benchmark.AddListener(&csvExporting);
  auto status = benchmark.Run(static_cast<int>(argv.size()), argv.data());
  if (status != kTfLiteOk) {
    firebaseReporting.ReportFailure(status);
  }
}

}  // namespace
}  // namespace benchmark
}  // namespace tflite

extern "C" {

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_benchmark_firebase_BenchmarkModel_nativeRun(
    JNIEnv* env, jclass clazz, jstring library_dir, jint scenario,
    jint report_fd) {
  const char* lib_dir = env->GetStringUTFChars(library_dir, nullptr);

  tflite::benchmark::RunScenario(lib_dir, static_cast<int>(scenario),
                                 static_cast<int>(report_fd));

  env->ReleaseStringUTFChars(library_dir, lib_dir);
}

}  // extern "C"
