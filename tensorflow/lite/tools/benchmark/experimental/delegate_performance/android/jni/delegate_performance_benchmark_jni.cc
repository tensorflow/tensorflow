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

#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/jni/accuracy_benchmark.h"
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/jni/latency_benchmark.h"

namespace {

// A helper method that converts an array of strings passed from Java to a
// vector of strings in C++.
std::vector<std::string> toStringVector(JNIEnv* env,
                                        jobjectArray string_array) {
  int len = env->GetArrayLength(string_array);
  std::vector<std::string> vec(len);
  for (int i = 0; i < len; ++i) {
    jstring str =
        static_cast<jstring>(env->GetObjectArrayElement(string_array, i));
    const char* chars = env->GetStringUTFChars(str, nullptr);
    vec[i] = std::string(chars);
    env->ReleaseStringUTFChars(str, chars);
    env->DeleteLocalRef(str);
  }
  return vec;
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_benchmark_delegateperformance_DelegatePerformanceBenchmark_latencyBenchmarkNativeRun(
    JNIEnv* env, jclass clazz, jobjectArray args_obj, jstring result_path_obj) {
  std::vector<std::string> args = toStringVector(env, args_obj);
  const char* result_path_chars =
      env->GetStringUTFChars(result_path_obj, nullptr);

  tflite::benchmark::latency::Benchmark(args, result_path_chars);

  env->ReleaseStringUTFChars(result_path_obj, result_path_chars);
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_benchmark_delegateperformance_DelegatePerformanceBenchmark_accuracyBenchmarkNativeRun(
    JNIEnv* env, jclass clazz, jobjectArray args_obj, jint model_fd,
    jlong model_offset, jlong model_size, jstring result_path_obj) {
  std::vector<std::string> args = toStringVector(env, args_obj);
  const char* result_path_chars =
      env->GetStringUTFChars(result_path_obj, nullptr);

  int status = tflite::benchmark::accuracy::Benchmark(
      args, static_cast<int>(model_fd), static_cast<size_t>(model_offset),
      static_cast<size_t>(model_size), result_path_chars);

  env->ReleaseStringUTFChars(result_path_obj, result_path_chars);
  return status;
}

}  // extern "C"
