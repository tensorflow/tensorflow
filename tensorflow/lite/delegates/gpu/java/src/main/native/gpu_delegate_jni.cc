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

#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL Java_org_tensorflow_lite_gpu_GpuDelegate_createDelegate(
    JNIEnv* env, jclass clazz, jboolean precision_loss_allowed,
    jboolean dynamic_batch_enabled, jint preferred_gl_object_type) {
  TfLiteGpuDelegateOptions options;
  options.metadata = nullptr;
  options.compile_options.precision_loss_allowed =
      precision_loss_allowed == JNI_TRUE ? 1 : 0;
  options.compile_options.preferred_gl_object_type =
      static_cast<int32_t>(preferred_gl_object_type);
  options.compile_options.dynamic_batch_enabled =
      dynamic_batch_enabled == JNI_TRUE ? 1 : 0;
  return reinterpret_cast<jlong>(TfLiteGpuDelegateCreate(&options));
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_gpu_GpuDelegate_deleteDelegate(
    JNIEnv* env, jclass clazz, jlong delegate) {
  TfLiteGpuDelegateDelete(reinterpret_cast<TfLiteDelegate*>(delegate));
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_gpu_GpuDelegate_bindGlBufferToTensor(
    JNIEnv* env, jclass clazz, jlong delegate, jint tensor_index, jint ssbo) {
  return TfLiteGpuDelegateBindBufferToTensor(
             reinterpret_cast<TfLiteDelegate*>(delegate),
             static_cast<GLuint>(ssbo), static_cast<int>(tensor_index))
             ? JNI_TRUE
             : JNI_FALSE;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
