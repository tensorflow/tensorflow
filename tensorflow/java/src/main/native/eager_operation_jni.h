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

#ifndef TENSORFLOW_JAVA_SRC_MAIN_NATIVE_EAGER_OPERATION_JNI_H_
#define TENSORFLOW_JAVA_SRC_MAIN_NATIVE_EAGER_OPERATION_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_tensorflow_EagerOperation
 * Method:    delete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperation_delete(JNIEnv *,
                                                                 jclass, jlong);

/*
 * Class:     org_tensorflow_EagerOperation
 * Method:    deleteTensorHandle
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_EagerOperation_deleteTensorHandle(JNIEnv *, jclass, jlong);

/**
 * Class:     org_tensorflow_EagerOperation
 * Method:    resolveTensorHandle
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL
Java_org_tensorflow_EagerOperation_resolveTensorHandle(JNIEnv *, jclass, jlong);

/**
 * Class:     org_tensorflow_EagerOperation
 * Method:    outputListLength
 * Signature: (JLjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_EagerOperation_outputListLength(
    JNIEnv *, jclass, jlong, jstring);

/**
 * Class:     org_tensorflow_EagerOperation
 * Method:    inputListLength
 * Signature: (JLjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_EagerOperation_inputListLength(
    JNIEnv *, jclass, jlong, jstring);

/**
 * Class:     org_tensorflow_EagerOperation
 * Method:    dataType
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_EagerOperation_dataType(JNIEnv *,
                                                                   jclass,
                                                                   jlong);

/**
 * Class:     org_tensorflow_EagerOperation
 * Method:    numDims
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_EagerOperation_numDims(JNIEnv *,
                                                                  jclass,
                                                                  jlong);

/**
 * Class:     org_tensorflow_EagerOperation
 * Method:    dim
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_org_tensorflow_EagerOperation_dim(JNIEnv *, jclass,
                                                               jlong, jint);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_JAVA_SRC_MAIN_NATIVE_EAGER_OPERATION_JNI_H_
