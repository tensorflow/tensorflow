/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_JAVA_SRC_MAIN_NATIVE_EAGER_OPERATION_BUILDER_JNI_H_
#define TENSORFLOW_JAVA_SRC_MAIN_NATIVE_EAGER_OPERATION_BUILDER_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    allocate
 * Signature: (JLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_org_tensorflow_EagerOperationBuilder_allocate(
    JNIEnv *, jclass, jlong, jstring);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    delete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_EagerOperationBuilder_delete(JNIEnv *, jclass, jlong);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    execute
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL
Java_org_tensorflow_EagerOperationBuilder_execute(JNIEnv *, jclass, jlong);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    addInput
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_addInput(
    JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    addInputList
 * Signature: (J[J)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_addInputList(
    JNIEnv *, jclass, jlong, jlongArray);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setDevice
 * Signature: (JLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setDevice(
    JNIEnv *, jclass, jlong, jstring);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrString
 * Signature: (JLjava/lang/String;[B)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setAttrString(
    JNIEnv *, jclass, jlong, jstring, jbyteArray);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrStringList
 * Signature: (JLjava/lang/String;[L)V
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_EagerOperationBuilder_setAttrStringList(JNIEnv *, jclass,
                                                            jlong, jstring,
                                                            jobjectArray);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrInt
 * Signature: (JLjava/lang/String;J)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setAttrInt(
    JNIEnv *, jclass, jlong, jstring, jlong);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrIntList
 * Signature: (JLjava/lang/String;[J)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setAttrIntList(
    JNIEnv *, jclass, jlong, jstring, jlongArray);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrFloat
 * Signature: (JLjava/lang/String;F)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setAttrFloat(
    JNIEnv *, jclass, jlong, jstring, jfloat);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrFloatList
 * Signature: (JLjava/lang/String;[F)V
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_EagerOperationBuilder_setAttrFloatList(JNIEnv *, jclass,
                                                           jlong, jstring,
                                                           jfloatArray);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrBool
 * Signature: (JLjava/lang/String;Z)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setAttrBool(
    JNIEnv *, jclass, jlong, jstring, jboolean);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrBoolList
 * Signature: (JLjava/lang/String;[Z)V
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_EagerOperationBuilder_setAttrBoolList(JNIEnv *, jclass,
                                                          jlong, jstring,
                                                          jbooleanArray);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrType
 * Signature: (JLjava/lang/String;I)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setAttrType(
    JNIEnv *, jclass, jlong, jstring, jint);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrTypeList
 * Signature: (JLjava/lang/String;[I)V
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_EagerOperationBuilder_setAttrTypeList(JNIEnv *, jclass,
                                                          jlong, jstring,
                                                          jintArray);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrTensor
 * Signature: (JLjava/lang/String;J)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setAttrTensor(
    JNIEnv *, jclass, jlong, jstring, jlong);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrShape
 * Signature: (JLjava/lang/String;[JI)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setAttrShape(
    JNIEnv *, jclass, jlong, jstring, jlongArray, jint);

/*
 * Class:     org_tensorflow_EagerOperationBuilder
 * Method:    setAttrShapeList
 * Signature: (JLjava/lang/String;[J[I)V
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_EagerOperationBuilder_setAttrShapeList(JNIEnv *, jclass,
                                                           jlong, jstring,
                                                           jlongArray,
                                                           jintArray);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_JAVA_SRC_MAIN_NATIVE_EAGER_OPERATION_BUILDER_JNI_H_
