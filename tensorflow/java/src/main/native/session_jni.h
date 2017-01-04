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

#ifndef TENSORFLOW_JAVA_SESSION_JNI_H_
#define TENSORFLOW_JAVA_SESSION_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_tensorflow_Session
 * Method:    allocate
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_org_tensorflow_Session_allocate(JNIEnv *, jclass,
                                                             jlong);

/*
 * Class:     org_tensorflow_Session
 * Method:    delete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_Session_delete(JNIEnv *, jclass,
                                                          jlong);

/*
 * Class:     org_tensorflow_Session
 * Method:    run
 * Signature: (J[B[J[J[I[J[I[JZ[J)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_tensorflow_Session_run(
    JNIEnv *, jclass, jlong, jbyteArray, jlongArray, jlongArray, jintArray,
    jlongArray, jintArray, jlongArray, jboolean, jlongArray);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_JAVA_SESSION_JNI_H_
