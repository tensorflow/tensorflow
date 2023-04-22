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

#include "tensorflow/java/src/main/native/tensorflow_jni.h"

#include <limits>
#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"

JNIEXPORT jstring JNICALL Java_org_tensorflow_TensorFlow_version(JNIEnv* env,
                                                                 jclass clazz) {
  return env->NewStringUTF(TF_Version());
}

JNIEXPORT jbyteArray JNICALL
Java_org_tensorflow_TensorFlow_registeredOpList(JNIEnv* env, jclass clazz) {
  TF_Buffer* buf = TF_GetAllOpList();
  jint length = static_cast<int>(buf->length);
  jbyteArray ret = env->NewByteArray(length);
  env->SetByteArrayRegion(ret, 0, length, static_cast<const jbyte*>(buf->data));
  TF_DeleteBuffer(buf);
  return ret;
}

JNIEXPORT jlong JNICALL Java_org_tensorflow_TensorFlow_libraryLoad(
    JNIEnv* env, jclass clazz, jstring filename) {
  TF_Status* status = TF_NewStatus();
  const char* cname = env->GetStringUTFChars(filename, nullptr);
  TF_Library* h = TF_LoadLibrary(cname, status);
  throwExceptionIfNotOK(env, status);
  env->ReleaseStringUTFChars(filename, cname);
  TF_DeleteStatus(status);
  return reinterpret_cast<jlong>(h);
}

JNIEXPORT void JNICALL Java_org_tensorflow_TensorFlow_libraryDelete(
    JNIEnv* env, jclass clazz, jlong handle) {
  if (handle != 0) {
    TF_DeleteLibraryHandle(reinterpret_cast<TF_Library*>(handle));
  }
}

JNIEXPORT jbyteArray JNICALL Java_org_tensorflow_TensorFlow_libraryOpList(
    JNIEnv* env, jclass clazz, jlong handle) {
  TF_Buffer buf = TF_GetOpList(reinterpret_cast<TF_Library*>(handle));
  if (buf.length > std::numeric_limits<jint>::max()) {
    throwException(env, kIndexOutOfBoundsException,
                   "Serialized OpList is too large for a byte[] array");
    return nullptr;
  }
  auto ret_len = static_cast<jint>(buf.length);
  jbyteArray ret = env->NewByteArray(ret_len);
  env->SetByteArrayRegion(ret, 0, ret_len, static_cast<const jbyte*>(buf.data));
  return ret;
}
