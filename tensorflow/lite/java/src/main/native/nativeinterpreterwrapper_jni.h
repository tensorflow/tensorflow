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

#ifndef TENSORFLOW_LITE_JAVA_SRC_MAIN_NATIVE_NATIVEINTERPRETERWRAPPER_JNI_H_
#define TENSORFLOW_LITE_JAVA_SRC_MAIN_NATIVE_NATIVEINTERPRETERWRAPPER_JNI_H_

#include <jni.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/java/src/main/native/exception_jni.h"
#include "tensorflow/lite/java/src/main/native/tensor_jni.h"
#include "tensorflow/lite/model.h"

namespace tflite {
// This is to be provided at link-time by a library.
extern std::unique_ptr<OpResolver> CreateOpResolver();
}  // namespace tflite

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:    allocateTensors
 *  Signature: (JJ)V
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_allocateTensors(
    JNIEnv* env, jclass clazz, jlong handle, jlong error_handle);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:    getInputTensorIndex
 *  Signature: (JI)I
 */
JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputTensorIndex(
    JNIEnv* env, jclass clazz, jlong handle, jint input_index);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:    getOutputTensorIndex
 *  Signature: (JI)I
 */
JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputTensorIndex(
    JNIEnv* env, jclass clazz, jlong handle, jint output_index);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:    getInputCount
 *  Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputCount(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:    getOutputCount
 *  Signature: (J)I
 */
JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputCount(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (J)[Ljava/lang/Object;
 */
JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputNames(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (J)[Ljava/lang/Object;
 */
JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputNames(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (JZ)V
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_useNNAPI(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle,
                                                           jboolean state);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (JZ)V
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_allowFp16PrecisionForFp32(
    JNIEnv* env, jclass clazz, jlong handle, jboolean allow);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (JI)V
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_numThreads(JNIEnv* env,
                                                             jclass clazz,
                                                             jlong handle,
                                                             jint num_threads);
/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (I)J
 */
JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createErrorReporter(
    JNIEnv* env, jclass clazz, jint size);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (Ljava/lang/String;J)J
 */
JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createModel(
    JNIEnv* env, jclass clazz, jstring model_file, jlong error_handle);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (Ljava/lang/Object;J)J
 */
JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createModelWithBuffer(
    JNIEnv* env, jclass clazz, jobject model_buffer, jlong error_handle);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (JJI)J
 */
JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createInterpreter(
    JNIEnv* env, jclass clazz, jlong model_handle, jlong error_handle,
    jint num_threads);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:    run
 *  Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_org_tensorflow_lite_NativeInterpreterWrapper_run(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (JI)I
 *
 * Gets output dimensions.
 */
JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputDataType(
    JNIEnv* env, jclass clazz, jlong handle, jint output_idx);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (JI)I
 *
 * Gets output quantization zero point.
 */
JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputQuantizationZeroPoint(
    JNIEnv* env, jclass clazz, jlong handle, jint output_idx);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (JI)F
 *
 * Gets output quantization scale.
 */
JNIEXPORT jfloat JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputQuantizationScale(
    JNIEnv* env, jclass clazz, jlong handle, jint output_idx);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (JJI[I)Z
 *
 * It returns true if resizing input tensor to different dimensions, else return
 * false.
 */
JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_resizeInput(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle,
    jint input_idx, jintArray dims);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:    applyDelegate
 *  Signature: (JJJ)V
 */
JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_applyDelegate(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle,
    jlong delegate_handle);

/*
 *  Class:     org_tensorflow_lite_NativeInterpreterWrapper
 *  Method:
 *  Signature: (JJJ)
 */
JNIEXPORT void JNICALL Java_org_tensorflow_lite_NativeInterpreterWrapper_delete(
    JNIEnv* env, jclass clazz, jlong error_handle, jlong model_handle,
    jlong interpreter_handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // TENSORFLOW_LITE_JAVA_SRC_MAIN_NATIVE_NATIVEINTERPRETERWRAPPER_JNI_H_
