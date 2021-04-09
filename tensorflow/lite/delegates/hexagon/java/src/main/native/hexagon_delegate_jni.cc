/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_HexagonDelegate_createDelegate(
    JNIEnv* env, jclass clazz) {
  // Auto-choosing the best performing config for closed release.
  TfLiteHexagonDelegateOptions options = {0};
  TfLiteHexagonInit();
  return reinterpret_cast<jlong>(TfLiteHexagonDelegateCreate(&options));
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_HexagonDelegate_deleteDelegate(
    JNIEnv* env, jclass clazz, jlong delegate) {
  TfLiteHexagonDelegateDelete(reinterpret_cast<TfLiteDelegate*>(delegate));
  TfLiteHexagonTearDown();
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_HexagonDelegate_setAdspLibraryPath(
    JNIEnv* env, jclass clazz, jstring native_lib_path) {
  const char* lib_dir_path = env->GetStringUTFChars(native_lib_path, nullptr);
  std::stringstream path;
  path << lib_dir_path
       << ";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
  env->ReleaseStringUTFChars(native_lib_path, lib_dir_path);
  return setenv("ADSP_LIBRARY_PATH", path.str().c_str(), 1 /*override*/) == 0
             ? JNI_TRUE
             : JNI_FALSE;
}

}  // extern "C"
