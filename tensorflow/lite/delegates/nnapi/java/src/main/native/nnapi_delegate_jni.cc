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

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_nnapi_NnApiDelegate_createDelegate(JNIEnv* env,
                                                            jclass clazz) {
  return reinterpret_cast<jlong>(tflite::NnApiDelegate());
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
