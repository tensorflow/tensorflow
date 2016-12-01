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

// The methods are exposed to Java to allow for interaction with the native
// TensorFlow code. See
// tensorflow/examples/android/src/org/tensorflow/TensorFlowClassifier.java
// for the Java counterparts.

#ifndef ORG_TENSORFLOW_JNI_TENSORFLOW_JNI_H_  // NOLINT
#define ORG_TENSORFLOW_JNI_TENSORFLOW_JNI_H_  // NOLINT

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define TENSORFLOW_METHOD(METHOD_NAME) \
  Java_org_tensorflow_contrib_android_TensorFlowInferenceInterface_##METHOD_NAME  // NOLINT

#define FILL_NODE_SIGNATURE(DTYPE, JAVA_DTYPE)                       \
  JNIEXPORT void TENSORFLOW_METHOD(fillNode##DTYPE)(                 \
      JNIEnv * env, jobject thiz, jstring node_name, jintArray dims, \
      j##JAVA_DTYPE##Array arr)

#define READ_NODE_SIGNATURE(DTYPE, JAVA_DTYPE)               \
  JNIEXPORT jint TENSORFLOW_METHOD(readNode##DTYPE)(         \
      JNIEnv * env, jobject thiz, jstring node_name_jstring, \
      j##JAVA_DTYPE##Array arr)

JNIEXPORT void JNICALL TENSORFLOW_METHOD(testLoaded)(JNIEnv* env, jobject thiz);

JNIEXPORT jint JNICALL TENSORFLOW_METHOD(initializeTensorFlow)(
    JNIEnv* env, jobject thiz, jobject java_asset_manager, jstring model);

JNIEXPORT jint JNICALL TENSORFLOW_METHOD(runInference)(
    JNIEnv* env, jobject thiz, jobjectArray output_name_strings);

JNIEXPORT jint JNICALL TENSORFLOW_METHOD(close)(JNIEnv* env, jobject thiz);

FILL_NODE_SIGNATURE(Float, float);
FILL_NODE_SIGNATURE(Int, int);
FILL_NODE_SIGNATURE(Double, double);

READ_NODE_SIGNATURE(Float, float);
READ_NODE_SIGNATURE(Int, int);
READ_NODE_SIGNATURE(Double, double);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ORG_TENSORFLOW_JNI_TENSORFLOW_JNI_H_  // NOLINT
