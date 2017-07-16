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

#ifndef TENSORFLOW_JAVA_OPERATION_JNI_H_
#define TENSORFLOW_JAVA_OPERATION_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_tensorflow_Operation
 * Method:    name
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_org_tensorflow_Operation_name(JNIEnv *, jclass,
                                                             jlong);

/*
 * Class:     org_tensorflow_Operation
 * Method:    type
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_org_tensorflow_Operation_type(JNIEnv *, jclass,
                                                             jlong);

/*
 * Class:     org_tensorflow_Operation
 * Method:    numOutputs
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_Operation_numOutputs(JNIEnv *,
                                                                jclass, jlong);

/*
 * Class:     org_tensorflow_Operation
 * Method:    outputListLength
 * Signature: (JLjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_Operation_outputListLength(JNIEnv *,
                                                                      jclass,
                                                                      jlong,
                                                                      jstring);

/*
 * Class:     org_tensorflow_Operation
 * Method:    shape
 * Signature: (JJI)[J
 */
JNIEXPORT jlongArray JNICALL Java_org_tensorflow_Operation_shape(JNIEnv *,
                                                                 jclass, jlong,
                                                                 jlong, jint);

/*
 * Class:     org_tensorflow_Operation
 * Method:    dtype
 * Signature: (JJI)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_Operation_dtype(JNIEnv *, jclass,
                                                           jlong, jlong, jint);


/*
 * Class:     org_tensorflow_Operation
 * Method:    inputListLength
 * Signature: (JLjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_Operation_inputListLength(JNIEnv *,
                                                                      jclass,
                                                                      jlong,
                                                                      jstring);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_JAVA_OPERATION_JNI_H_
