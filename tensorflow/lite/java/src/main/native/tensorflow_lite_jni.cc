/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <stdio.h>

#include "tensorflow/lite/java/src/main/native/tensorflow_lite_jni.h"
#include "tensorflow/lite/version.h"

JNIEXPORT jstring JNICALL
Java_org_tensorflow_lite_TensorFlowLite_runtimeVersion(JNIEnv* env,
                                                       jclass /*clazz*/) {
  const char* kTfLiteVersionString = TFLITE_VERSION_STRING;
  return env->NewStringUTF(kTfLiteVersionString);
}

JNIEXPORT jstring JNICALL Java_org_tensorflow_lite_TensorFlowLite_schemaVersion(
    JNIEnv* env, jclass /*clazz*/) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%d", TFLITE_SCHEMA_VERSION);
  return env->NewStringUTF(buf);
}
