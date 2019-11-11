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

#include <jni.h>

#include <sstream>
#include <string>

#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"
#include "tensorflow/lite/tools/benchmark/logging.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace tflite {
namespace benchmark {
namespace {

class AndroidBenchmarkLoggingListener : public BenchmarkListener {
  void OnBenchmarkEnd(const BenchmarkResults& results) override {
    auto inference_us = results.inference_time_us();
    auto init_us = results.startup_latency_us();
    auto warmup_us = results.warmup_time_us();
    std::stringstream results_output;
    results_output << "Average inference timings in us: "
                   << "Warmup: " << warmup_us.avg() << ", "
                   << "Init: " << init_us << ", "
                   << "Inference: " << inference_us.avg();
    results_output << "Overall " << results.overall_mem_usage();

#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_ERROR, "tflite", "%s",
                        results_output.str().c_str());
#else
    fprintf(stderr, "%s", results_output.str().c_str());
#endif
  }
};

void Run(int argc, char** argv) {
  BenchmarkTfLiteModel benchmark;
  AndroidBenchmarkLoggingListener listener;
  benchmark.AddListener(&listener);
  benchmark.Run(argc, argv);
}

}  // namespace
}  // namespace benchmark
}  // namespace tflite

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_benchmark_BenchmarkModel_nativeRun(JNIEnv* env,
                                                            jclass clazz,
                                                            jstring args_obj) {
  const char* args_chars = env->GetStringUTFChars(args_obj, nullptr);

  // Split the args string into individual arg tokens.
  std::istringstream iss(args_chars);
  std::vector<std::string> args_split{std::istream_iterator<std::string>(iss),
                                      {}};

  // Construct a fake argv command-line object for the benchmark.
  std::vector<char*> argv;
  std::string arg0 = "(BenchmarkModelAndroid)";
  argv.push_back(const_cast<char*>(arg0.data()));
  for (auto& arg : args_split) {
    argv.push_back(const_cast<char*>(arg.data()));
  }

  tflite::benchmark::Run(static_cast<int>(argv.size()), argv.data());

  env->ReleaseStringUTFChars(args_obj, args_chars);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
