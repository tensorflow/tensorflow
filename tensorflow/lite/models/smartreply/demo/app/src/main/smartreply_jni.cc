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

#include <jni.h>
#include <utility>
#include <vector>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/models/smartreply/predictor.h"

const char kIllegalStateException[] = "java/lang/IllegalStateException";

using tflite::custom::smartreply::GetSegmentPredictions;
using tflite::custom::smartreply::PredictorResponse;

template <typename T>
T CheckNotNull(JNIEnv* env, T&& t) {
  if (t == nullptr) {
    env->ThrowNew(env->FindClass(kIllegalStateException), "");
    return nullptr;
  }
  return std::forward<T>(t);
}

std::vector<std::string> jniStringArrayToVector(JNIEnv* env,
                                                jobjectArray string_array) {
  int count = env->GetArrayLength(string_array);
  std::vector<std::string> result;
  for (int i = 0; i < count; i++) {
    auto jstr =
        reinterpret_cast<jstring>(env->GetObjectArrayElement(string_array, i));
    const char* raw_str = env->GetStringUTFChars(jstr, JNI_FALSE);
    result.emplace_back(std::string(raw_str));
    env->ReleaseStringUTFChars(jstr, raw_str);
  }
  return result;
}

struct JNIStorage {
  std::vector<std::string> backoff_list;
  std::unique_ptr<::tflite::FlatBufferModel> model;
};

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_android_smartreply_SmartReplyClient_loadJNI(
    JNIEnv* env, jobject thiz, jobject model_buffer,
    jobjectArray backoff_list) {
  const char* buf =
      static_cast<char*>(env->GetDirectBufferAddress(model_buffer));
  jlong capacity = env->GetDirectBufferCapacity(model_buffer);

  JNIStorage* storage = new JNIStorage;
  storage->model = tflite::FlatBufferModel::BuildFromBuffer(
      buf, static_cast<size_t>(capacity));
  storage->backoff_list = jniStringArrayToVector(env, backoff_list);

  if (!storage->model) {
    delete storage;
    env->ThrowNew(env->FindClass(kIllegalStateException), "");
    return 0;
  }
  return reinterpret_cast<jlong>(storage);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_example_android_smartreply_SmartReplyClient_predictJNI(
    JNIEnv* env, jobject /*thiz*/, jlong storage_ptr, jobjectArray input_text) {
  // Predict
  if (storage_ptr == 0) {
    return nullptr;
  }
  JNIStorage* storage = reinterpret_cast<JNIStorage*>(storage_ptr);
  if (storage == nullptr) {
    return nullptr;
  }
  std::vector<PredictorResponse> responses;
  GetSegmentPredictions(jniStringArrayToVector(env, input_text),
                        *storage->model, {storage->backoff_list}, &responses);

  // Create a SmartReply[] to return back to Java
  jclass smart_reply_class = CheckNotNull(
      env, env->FindClass("com/example/android/smartreply/SmartReply"));
  if (env->ExceptionCheck()) {
    return nullptr;
  }
  jmethodID smart_reply_ctor = CheckNotNull(
      env,
      env->GetMethodID(smart_reply_class, "<init>", "(Ljava/lang/String;F)V"));
  if (env->ExceptionCheck()) {
    return nullptr;
  }
  jobjectArray array = CheckNotNull(
      env, env->NewObjectArray(responses.size(), smart_reply_class, nullptr));
  if (env->ExceptionCheck()) {
    return nullptr;
  }
  for (int i = 0; i < responses.size(); i++) {
    jstring text =
        CheckNotNull(env, env->NewStringUTF(responses[i].GetText().data()));
    if (env->ExceptionCheck()) {
      return nullptr;
    }
    jobject reply = env->NewObject(smart_reply_class, smart_reply_ctor, text,
                                   responses[i].GetScore());
    env->SetObjectArrayElement(array, i, reply);
  }
  return array;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_android_smartreply_SmartReplyClient_unloadJNI(
    JNIEnv* env, jobject thiz, jlong storage_ptr) {
  if (storage_ptr != 0) {
    JNIStorage* storage = reinterpret_cast<JNIStorage*>(storage_ptr);
    delete storage;
  }
}
