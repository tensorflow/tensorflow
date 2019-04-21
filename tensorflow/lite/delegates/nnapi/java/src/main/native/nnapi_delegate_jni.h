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

#ifndef TENSORFLOW_LITE_DELEGATES_NNPAI_JAVA_SRC_MAIN_NATIVE_NNAPI_DELEGATE_JNI_H_
#define TENSORFLOW_LITE_DELEGATES_NNAPI_JAVA_SRC_MAIN_NATIVE_NNAPI_DELEGATE_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/*
 * Class:     org_tensorflow_lite_nnapi_NnApiDelegate
 * Method:    createDelegate
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_org_tensorflow_lite_nnapi_NnApiDelegate_createDelegate(
    JNIEnv* env, jclass clazz);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_NNPAI_JAVA_SRC_MAIN_NATIVE_NNAPI_DELEGATE_JNI_H_
