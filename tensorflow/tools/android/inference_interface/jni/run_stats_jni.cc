/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tools/android/inference_interface/jni/run_stats_jni.h"

#include <jni.h>

#include <sstream>

#include "tensorflow/core/protobuf/config.pb.h"

using tensorflow::RunMetadata;
using tensorflow::StatSummarizer;

namespace {
StatSummarizer* requireHandle(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                  "close() has been called on the RunStats object");
    return nullptr;
  }
  return reinterpret_cast<StatSummarizer*>(handle);
}
}  // namespace

#define RUN_STATS_METHOD(name) \
  JNICALL Java_org_tensorflow_contrib_android_RunStats_##name

JNIEXPORT jlong RUN_STATS_METHOD(allocate)(JNIEnv* env, jclass clazz) {
  static_assert(sizeof(jlong) >= sizeof(StatSummarizer*),
                "Cannot package C++ object pointers as a Java long");
  tsl::StatSummarizerOptions opts;
  return reinterpret_cast<jlong>(new StatSummarizer(opts));
}

JNIEXPORT void RUN_STATS_METHOD(delete)(JNIEnv* env, jclass clazz,
                                        jlong handle) {
  if (handle == 0) return;
  delete reinterpret_cast<StatSummarizer*>(handle);
}

JNIEXPORT void RUN_STATS_METHOD(add)(JNIEnv* env, jclass clazz, jlong handle,
                                     jbyteArray run_metadata) {
  StatSummarizer* s = requireHandle(env, handle);
  if (s == nullptr) return;
  jbyte* data = env->GetByteArrayElements(run_metadata, nullptr);
  int size = static_cast<int>(env->GetArrayLength(run_metadata));
  tensorflow::RunMetadata proto;
  if (!proto.ParseFromArray(data, size)) {
    env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                  "runMetadata does not seem to be a serialized RunMetadata "
                  "protocol message");
  } else if (proto.has_step_stats()) {
    s->ProcessStepStats(proto.step_stats());
  }
  env->ReleaseByteArrayElements(run_metadata, data, JNI_ABORT);
}

JNIEXPORT jstring RUN_STATS_METHOD(summary)(JNIEnv* env, jclass clazz,
                                            jlong handle) {
  StatSummarizer* s = requireHandle(env, handle);
  if (s == nullptr) return nullptr;
  std::stringstream ret;
  ret << s->GetStatsByMetric("Top 10 CPU", tsl::StatsCalculator::BY_TIME, 10)
      << s->GetStatsByNodeType() << s->ShortSummary();
  return env->NewStringUTF(ret.str().c_str());
}

#undef RUN_STATS_METHOD
