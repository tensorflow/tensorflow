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

#include "tensorflow/lite/delegates/gpu/delegate.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL Java_org_tensorflow_lite_gpu_GpuDelegate_createDelegate(
    JNIEnv* env, jclass clazz, jboolean precision_loss_allowed,
    jint inference_preference) {
  TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
  if (precision_loss_allowed == JNI_TRUE) {
    options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    options.inference_priority2 =
        TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
    options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
  }
  options.inference_preference = static_cast<int32_t>(inference_preference);
  return reinterpret_cast<jlong>(TfLiteGpuDelegateV2Create(&options));
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_gpu_GpuDelegate_deleteDelegate(
    JNIEnv* env, jclass clazz, jlong delegate) {
  TfLiteGpuDelegateV2Delete(reinterpret_cast<TfLiteDelegate*>(delegate));
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
