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

#include <jni.h>

#include <string>
#include <vector>

#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/jni/latency_benchmark.h"

extern "C" {

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_benchmark_delegateperformance_DelegatePerformanceBenchmark_latencyBenchmarkNativeRun(
    JNIEnv* env, jclass clazz, jobjectArray args_obj, jstring result_path_obj) {
  const char* result_path_chars =
      env->GetStringUTFChars(result_path_obj, nullptr);
  size_t args_len = env->GetArrayLength(args_obj);
  std::vector<std::string> args(args_len);
  for (size_t i = 0; i < args_len; ++i) {
    jstring arg = static_cast<jstring>(env->GetObjectArrayElement(args_obj, i));
    const char* arg_chars = env->GetStringUTFChars(arg, nullptr);
    args[i] = std::string(arg_chars);
    env->ReleaseStringUTFChars(arg, arg_chars);
    env->DeleteLocalRef(arg);
  }

  tflite::benchmark::latency::Benchmark(args, result_path_chars);

  env->ReleaseStringUTFChars(result_path_obj, result_path_chars);
}

}  // extern "C"
