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

#include "tensorflow/lite/delegates/gpu/java/src/main/native/gpu_delegate_jni.h"

#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_experimental_GpuDelegate_createDelegate(JNIEnv* env,
                                                                 jclass clazz) {
  // Auto-choosing the best performing config for closed release.
  TfLiteGpuDelegateOptions options;
  options.metadata = nullptr;
  options.compile_options.precision_loss_allowed = 1;
  options.compile_options.preferred_gl_object_type =
      TFLITE_GL_OBJECT_TYPE_FASTEST;
  options.compile_options.dynamic_batch_enabled = 0;
  return reinterpret_cast<jlong>(TfLiteGpuDelegateCreate(&options));
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_experimental_GpuDelegate_deleteDelegate(
    JNIEnv* env, jclass clazz, jlong delegate) {
  TfLiteGpuDelegateDelete(reinterpret_cast<TfLiteDelegate*>(delegate));
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_experimental_GpuDelegate_bindGlBufferToTensor(
    JNIEnv* env, jclass clazz, jlong delegate, jint tensor_index, jint ssbo) {
  return TfLiteGpuDelegateBindBufferToTensor(
             reinterpret_cast<TfLiteDelegate*>(delegate),
             static_cast<GLuint>(ssbo), static_cast<int>(tensor_index))
             ? JNI_TRUE
             : JNI_FALSE;
}
