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

#ifndef ORG_TENSORFLOW_JNI_RUN_STATS_JNI_H_
#define ORG_TENSORFLOW_JNI_RUN_STATS_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define RUN_STATS_METHOD(name) \
  Java_org_tensorflow_contrib_android_RunStats_##name

JNIEXPORT JNICALL jlong RUN_STATS_METHOD(allocate)(JNIEnv*, jclass);
JNIEXPORT JNICALL void RUN_STATS_METHOD(delete)(JNIEnv*, jclass, jlong);
JNIEXPORT JNICALL void RUN_STATS_METHOD(add)(JNIEnv*, jclass, jlong,
                                             jbyteArray);
JNIEXPORT JNICALL jstring RUN_STATS_METHOD(summary)(JNIEnv*, jclass, jlong);

#undef RUN_STATS_METHOD

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ORG_TENSORFLOW_JNI_RUN_STATS_JNI_H_
