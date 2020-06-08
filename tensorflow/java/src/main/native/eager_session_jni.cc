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

#include "tensorflow/java/src/main/native/eager_session_jni.h"

#include <cstring>
#include <memory>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"

JNIEXPORT jlong JNICALL Java_org_tensorflow_EagerSession_allocate(
    JNIEnv* env, jclass clazz, jboolean async, jint dpp, jbyteArray config) {
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  jbyte* cconfig = nullptr;
  TF_Status* status = TF_NewStatus();
  if (config != nullptr) {
    cconfig = env->GetByteArrayElements(config, nullptr);
    TFE_ContextOptionsSetConfig(
        opts, cconfig, static_cast<size_t>(env->GetArrayLength(config)),
        status);
    if (!throwExceptionIfNotOK(env, status)) {
      env->ReleaseByteArrayElements(config, cconfig, JNI_ABORT);
      TFE_DeleteContextOptions(opts);
      TF_DeleteStatus(status);
      return 0;
    }
  }
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_ContextOptionsSetDevicePlacementPolicy(
      opts, static_cast<TFE_ContextDevicePlacementPolicy>(dpp));
  TFE_Context* context = TFE_NewContext(opts, status);
  TFE_DeleteContextOptions(opts);
  if (config != nullptr) {
    env->ReleaseByteArrayElements(config, cconfig, JNI_ABORT);
  }
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return 0;
  }
  TF_DeleteStatus(status);
  static_assert(sizeof(jlong) >= sizeof(TFE_Context*),
                "Cannot represent a C TFE_Op as a Java long");
  return reinterpret_cast<jlong>(context);
}

JNIEXPORT void JNICALL Java_org_tensorflow_EagerSession_delete(JNIEnv* env,
                                                               jclass clazz,
                                                               jlong handle) {
  if (handle == 0) return;
  TFE_DeleteContext(reinterpret_cast<TFE_Context*>(handle));
}
