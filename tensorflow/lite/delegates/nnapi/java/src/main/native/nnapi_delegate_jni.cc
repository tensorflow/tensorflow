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

#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

using namespace tflite;

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegate_createDelegate(
    JNIEnv* env, jclass clazz, jint preference, jstring accelerator_name,
    jstring cache_dir, jstring model_token) {
  StatefulNnApiDelegate::Options options = StatefulNnApiDelegate::Options();
  options.execution_preference =
      (StatefulNnApiDelegate::Options::ExecutionPreference)preference;
  if (accelerator_name) {
    options.accelerator_name = env->GetStringUTFChars(accelerator_name, NULL);
  }
  if (cache_dir) {
    options.cache_dir = env->GetStringUTFChars(cache_dir, NULL);
  }
  if (model_token) {
    options.model_token = env->GetStringUTFChars(model_token, NULL);
  }

  auto delegate = new StatefulNnApiDelegate(options);

  if (options.accelerator_name) {
    env->ReleaseStringUTFChars(accelerator_name, options.accelerator_name);
  }

  if (options.cache_dir) {
    env->ReleaseStringUTFChars(cache_dir, options.accelerator_name);
  }

  if (options.model_token) {
    env->ReleaseStringUTFChars(model_token, options.accelerator_name);
  }

  return reinterpret_cast<jlong>(delegate);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegate_deleteDelegate(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong delegate) {
  delete reinterpret_cast<TfLiteDelegate*>(delegate);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
