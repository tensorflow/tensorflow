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

using namespace tflite;

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegate_createDelegate(
    JNIEnv* env, jclass clazz, jint preference, jstring accelerator_name,
    jstring cache_dir, jstring model_token, jint max_delegated_partitions,
    jboolean override_disallow_cpu, jboolean disallow_cpu_value,
    jboolean allow_fp16, jlong nnapi_support_library_handle) {
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

  if (max_delegated_partitions >= 0) {
    options.max_number_delegated_partitions = max_delegated_partitions;
  }

  if (override_disallow_cpu) {
    options.disallow_nnapi_cpu = disallow_cpu_value;
  }

  if (allow_fp16) {
    options.allow_fp16 = allow_fp16;
  }

  auto delegate =
      nnapi_support_library_handle
          ? new StatefulNnApiDelegate(reinterpret_cast<NnApiSLDriverImplFL5*>(
                                          nnapi_support_library_handle),
                                      options)
          : new StatefulNnApiDelegate(options);

  if (options.accelerator_name) {
    env->ReleaseStringUTFChars(accelerator_name, options.accelerator_name);
  }

  if (options.cache_dir) {
    env->ReleaseStringUTFChars(cache_dir, options.cache_dir);
  }

  if (options.model_token) {
    env->ReleaseStringUTFChars(model_token, options.model_token);
  }

  return reinterpret_cast<jlong>(delegate);
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegate_getNnapiErrno(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong delegate) {
  StatefulNnApiDelegate* nnapi_delegate =
      reinterpret_cast<StatefulNnApiDelegate*>(delegate);
  return nnapi_delegate->GetNnApiErrno();
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegate_deleteDelegate(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong delegate) {
  delete reinterpret_cast<StatefulNnApiDelegate*>(delegate);
}

}  // extern "C"
