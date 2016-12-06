/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/java/src/main/native/operation_jni.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"

namespace {
TF_Operation* requireHandle(JNIEnv* env, jlong handle) {
  static_assert(sizeof(jlong) >= sizeof(TF_Operation*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throwException(
        env, kNullPointerException,
        "close() has been called on the Graph this Operation was a part of");
    return nullptr;
  }
  return reinterpret_cast<TF_Operation*>(handle);
}
}  // namespace

JNIEXPORT jstring JNICALL Java_org_tensorflow_Operation_name(JNIEnv* env,
                                                             jclass clazz,
                                                             jlong handle) {
  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return nullptr;
  return env->NewStringUTF(TF_OperationName(op));
}

JNIEXPORT jstring JNICALL Java_org_tensorflow_Operation_type(JNIEnv* env,
                                                             jclass clazz,
                                                             jlong handle) {
  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return nullptr;
  return env->NewStringUTF(TF_OperationOpType(op));
}

JNIEXPORT jint JNICALL Java_org_tensorflow_Operation_numOutputs(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle) {
  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return 0;
  return TF_OperationNumOutputs(op);
}
