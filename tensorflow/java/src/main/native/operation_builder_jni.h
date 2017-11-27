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

#ifndef TENSORFLOW_JAVA_OPERATION_BUILDER_JNI_H_
#define TENSORFLOW_JAVA_OPERATION_BUILDER_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    allocate
 * Signature: (JLjava/lang/String;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_org_tensorflow_OperationBuilder_allocate(
    JNIEnv *, jclass, jlong, jstring, jstring);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    finish
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_org_tensorflow_OperationBuilder_finish(JNIEnv *,
                                                                    jclass,
                                                                    jlong);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    addInput
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_addInput(
    JNIEnv *, jclass, jlong, jlong, jint);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    addInputList
 * Signature: (J[J[I)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_addInputList(
    JNIEnv *, jclass, jlong, jlongArray, jintArray);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    addControlInput
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_addControlInput(
    JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setDevice
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setDevice(JNIEnv *,
                                                                      jclass,
                                                                      jlong,
                                                                      jstring);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrString
 * Signature: (JLjava/lang/String;[B)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrString(
    JNIEnv *, jclass, jlong, jstring, jbyteArray);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrInt
 * Signature: (JLjava/lang/String;J)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrInt(
    JNIEnv *, jclass, jlong, jstring, jlong);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrIntList
 * Signature: (JLjava/lang/String;[J)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrIntList(
    JNIEnv *, jclass, jlong, jstring, jlongArray);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrFloat
 * Signature: (JLjava/lang/String;F)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrFloat(
    JNIEnv *, jclass, jlong, jstring, jfloat);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrFloatList
 * Signature: (JLjava/lang/String;[F)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrFloatList(
    JNIEnv *, jclass, jlong, jstring, jfloatArray);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrBool
 * Signature: (JLjava/lang/String;Z)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrBool(
    JNIEnv *, jclass, jlong, jstring, jboolean);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrBoolList
 * Signature: (JLjava/lang/String;[Z)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrBoolList(
    JNIEnv *, jclass, jlong, jstring, jbooleanArray);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrType
 * Signature: (JLjava/lang/String;I)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrType(
    JNIEnv *, jclass, jlong, jstring, jint);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrTypeList
 * Signature: (JLjava/lang/String;[I)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrTypeList(
    JNIEnv *, jclass, jlong, jstring, jintArray);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrTensor
 * Signature: (JLjava/lang/String;J)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrTensor(
    JNIEnv *, jclass, jlong, jstring, jlong);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrTensorList
 * Signature: (JLjava/lang/String;[J)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrTensorList(
    JNIEnv *, jclass, jlong, jstring, jlongArray);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrShape
 * Signature: (JLjava/lang/String;[JI)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrShape(
    JNIEnv *, jclass, jlong, jstring, jlongArray, jint);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrShapeList
 * Signature: (JLjava/lang/String;[J[I)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrShapeList(
    JNIEnv *, jclass, jlong, jstring, jlongArray, jintArray);

/*
 * Class:     org_tensorflow_OperationBuilder
 * Method:    setAttrStringList
 * Signature: (JLjava/lang/String;[L)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_OperationBuilder_setAttrStringList(
    JNIEnv *, jclass, jlong, jstring, jobjectArray);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_JAVA_OPERATION_BUILDER_JNI_H_
