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

#ifndef TENSORFLOW_JAVA_SRC_MAIN_NATIVE_GRAPH_NODE_JNI_H_
#define TENSORFLOW_JAVA_SRC_MAIN_NATIVE_GRAPH_NODE_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_tensorflow_GraphNode
 * Method:    name
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_org_tensorflow_GraphNode_name(
    JNIEnv *, jclass, jlong);

/*
 * Class:     org_tensorflow_GraphNode
 * Method:    type
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_org_tensorflow_GraphNode_type(
    JNIEnv *, jclass, jlong);

/*
 * Class:     org_tensorflow_GraphNode
 * Method:    numOutputs
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_GraphNode_numOutputs(
    JNIEnv *, jclass, jlong);

/*
 * Class:     org_tensorflow_GraphNode
 * Method:    outputListLength
 * Signature: (JLjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_GraphNode_outputListLength(
    JNIEnv *, jclass, jlong, jstring);

/*
 * Class:     org_tensorflow_GraphNode
 * Method:    shape
 * Signature: (JJI)[J
 */
JNIEXPORT jlongArray JNICALL Java_org_tensorflow_GraphNode_shape(
    JNIEnv *, jclass, jlong, jlong, jint);

/*
 * Class:     org_tensorflow_GraphNode
 * Method:    dtype
 * Signature: (JJI)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_GraphNode_dtype(
    JNIEnv *, jclass, jlong, jlong, jint);


/*
 * Class:     org_tensorflow_GraphNode
 * Method:    inputListLength
 * Signature: (JLjava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_GraphNode_inputListLength(
    JNIEnv *, jclass, jlong, jstring);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_JAVA_SRC_MAIN_NATIVE_GRAPH_NODE_JNI_H_
