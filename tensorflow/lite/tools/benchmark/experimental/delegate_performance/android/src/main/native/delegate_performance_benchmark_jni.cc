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

#include <cstdint>
#include <string>
#include <vector>

#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/proto/delegate_performance.pb.h"
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/native/accuracy_benchmark.h"
#include "tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/src/main/native/latency_benchmark.h"

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

// Serializes the proto message into jbyteArray.
jbyteArray CppProtoToBytes(JNIEnv* env, const google::protobuf::MessageLite& proto) {
  jbyteArray array = nullptr;
  const int byte_size = proto.ByteSizeLong();
  if (byte_size) {
    array = env->NewByteArray(byte_size);
    void* ptr = env->GetPrimitiveArrayCritical(array, nullptr);
    proto.SerializeWithCachedSizesToArray(static_cast<uint8_t*>(ptr));
    env->ReleasePrimitiveArrayCritical(array, ptr, 0);
  }
  return array;
}

}  // namespace

extern "C" {

JNIEXPORT jbyteArray JNICALL
Java_org_tensorflow_lite_benchmark_delegateperformance_DelegatePerformanceBenchmark_latencyBenchmarkNativeRun(
    JNIEnv* env, jclass clazz, jobjectArray args_obj,
    jbyteArray tflite_settings_byte_array, jstring tflite_settings_path_obj,
    jint model_fd, jlong model_offset, jlong model_size) {
  std::vector<std::string> args = toStringVector(env, args_obj);
  const char* tflite_settings_path_chars =
      env->GetStringUTFChars(tflite_settings_path_obj, nullptr);
  jbyte* tflite_settings_bytes =
      env->GetByteArrayElements(tflite_settings_byte_array, nullptr);
  const tflite::TFLiteSettings* tflite_settings =
      flatbuffers::GetRoot<tflite::TFLiteSettings>(
          reinterpret_cast<const char*>(tflite_settings_bytes));

  tflite::proto::benchmark::LatencyResults results =
      tflite::benchmark::latency::Benchmark(
          *tflite_settings, tflite_settings_path_chars,
          static_cast<int>(model_fd), static_cast<size_t>(model_offset),
          static_cast<size_t>(model_size), args);

  env->ReleaseByteArrayElements(tflite_settings_byte_array,
                                tflite_settings_bytes, JNI_ABORT);
  env->ReleaseStringUTFChars(tflite_settings_path_obj,
                             tflite_settings_path_chars);
  return CppProtoToBytes(env, results);
}

// TODO(b/262411020): Consider returning jobject directly.
JNIEXPORT jbyteArray JNICALL
Java_org_tensorflow_lite_benchmark_delegateperformance_DelegatePerformanceBenchmark_accuracyBenchmarkNativeRun(
    JNIEnv* env, jclass clazz, jbyteArray tflite_settings_byte_array,
    jint model_fd, jlong model_offset, jlong model_size,
    jstring result_path_obj) {
  const char* result_path_chars =
      env->GetStringUTFChars(result_path_obj, nullptr);
  jbyte* tflite_settings_bytes =
      env->GetByteArrayElements(tflite_settings_byte_array, nullptr);
  const tflite::TFLiteSettings* tflite_settings =
      flatbuffers::GetRoot<tflite::TFLiteSettings>(
          reinterpret_cast<const char*>(tflite_settings_bytes));
  flatbuffers::FlatBufferBuilder fbb;

  flatbuffers::Offset<tflite::BenchmarkEvent> benchmark_event =
      tflite::benchmark::accuracy::Benchmark(
          fbb, *tflite_settings, static_cast<int>(model_fd),
          static_cast<size_t>(model_offset), static_cast<size_t>(model_size),
          result_path_chars);
  fbb.Finish(benchmark_event);

  env->ReleaseByteArrayElements(tflite_settings_byte_array,
                                tflite_settings_bytes, JNI_ABORT);
  env->ReleaseStringUTFChars(result_path_obj, result_path_chars);

  jbyteArray byte_array = nullptr;
  if (fbb.GetSize() > 0) {
    byte_array = env->NewByteArray(fbb.GetSize());
    env->SetByteArrayRegion(
        byte_array, 0, fbb.GetSize(),
        reinterpret_cast<const jbyte*>(fbb.GetBufferPointer()));
  }
  return byte_array;
}

JNIEXPORT jbyteArray JNICALL
Java_org_tensorflow_lite_benchmark_delegateperformance_DelegatePerformanceBenchmark_loadTfLiteSettingsJsonNative(
    JNIEnv* env, jclass clazz, jstring json_file_path_obj) {
  const char* json_file_path_chars =
      env->GetStringUTFChars(json_file_path_obj, nullptr);

  tflite::delegates::utils::TfLiteSettingsJsonParser parser;
  parser.Parse(json_file_path_chars);

  jbyteArray tflite_settings_byte_array = nullptr;
  flatbuffers::uoffset_t tflite_settings_size = parser.GetBufferSize();
  if (tflite_settings_size > 0) {
    tflite_settings_byte_array = env->NewByteArray(tflite_settings_size);
    env->SetByteArrayRegion(
        tflite_settings_byte_array, 0, tflite_settings_size,
        reinterpret_cast<const jbyte*>(parser.GetBufferPointer()));
  }
  env->ReleaseStringUTFChars(json_file_path_obj, json_file_path_chars);
  return tflite_settings_byte_array;
}

}  // extern "C"
