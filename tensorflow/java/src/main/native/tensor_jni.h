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

#ifndef TENSORFLOW_JAVA_TENSOR_JNI_H_
#define TENSORFLOW_JAVA_TENSOR_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Class:     org_tensorflow_Tensor
 * Method:    allocate
 * Signature: (I[JJ)J
 */
JNIEXPORT jlong JNICALL Java_org_tensorflow_Tensor_allocate(JNIEnv *, jclass,
                                                            jint, jlongArray,
                                                            jlong);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    allocateScalarBytes
 * Signature: ([B)J
 */
JNIEXPORT jlong JNICALL
Java_org_tensorflow_Tensor_allocateScalarBytes(JNIEnv *, jclass, jbyteArray);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    allocateNonScalarBytes
 * Signature: ([J[Ljava/lang/Object;)J
 */
JNIEXPORT jlong JNICALL Java_org_tensorflow_Tensor_allocateNonScalarBytes(
    JNIEnv *, jclass, jlongArray, jobjectArray);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    delete
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_Tensor_delete(JNIEnv *, jclass,
                                                         jlong);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    buffer
 * Signature: (J)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_org_tensorflow_Tensor_buffer(JNIEnv *, jclass,
                                                            jlong);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    dtype
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_Tensor_dtype(JNIEnv *, jclass,
                                                        jlong);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    shape
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_org_tensorflow_Tensor_shape(JNIEnv *, jclass,
                                                              jlong);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    setValue
 * Signature: (JLjava/lang/Object;)V
 *
 * REQUIRES: The jobject's type and shape are compatible the with the DataType
 * and shape of the Tensor referred to by the jlong handle.
 */
JNIEXPORT void JNICALL Java_org_tensorflow_Tensor_setValue(JNIEnv *, jclass,
                                                           jlong, jobject);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    scalarFloat
 * Signature: (J)F
 *
 */
JNIEXPORT jfloat JNICALL Java_org_tensorflow_Tensor_scalarFloat(JNIEnv *,
                                                                jclass, jlong);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    scalarDouble
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_org_tensorflow_Tensor_scalarDouble(JNIEnv *,
                                                                  jclass,
                                                                  jlong);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    scalarInt
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_org_tensorflow_Tensor_scalarInt(JNIEnv *, jclass,
                                                            jlong);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    scalarLong
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_org_tensorflow_Tensor_scalarLong(JNIEnv *, jclass,
                                                              jlong);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    scalarBoolean
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_org_tensorflow_Tensor_scalarBoolean(JNIEnv *,
                                                                    jclass,
                                                                    jlong);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    scalarBytes
 * Signature: (J)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_tensorflow_Tensor_scalarBytes(JNIEnv *,
                                                                    jclass,
                                                                    jlong);

/*
 * Class:     org_tensorflow_Tensor
 * Method:    readNDArray
 * Signature: (JLjava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_Tensor_readNDArray(JNIEnv *, jclass,
                                                              jlong, jobject);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_JAVA_TENSOR_JNI_H_
