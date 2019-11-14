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

#include "tensorflow/java/src/main/native/eager_operation_jni.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <memory>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"

namespace {

TFE_Op* requireOp(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kIllegalStateException,
                   "Eager session has been closed");
    return nullptr;
  }
  return reinterpret_cast<TFE_Op*>(handle);
}

TFE_TensorHandle* requireTensorHandle(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kIllegalStateException, "EagerSession has been closed");
    return nullptr;
  }
  return reinterpret_cast<TFE_TensorHandle*>(handle);
}

}  // namespace

JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperation_delete(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle) {
  if (handle == 0) return;
  TFE_DeleteOp(reinterpret_cast<TFE_Op*>(handle));
}

JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperation_deleteTensorHandle(
    JNIEnv* env, jclass clazz, jlong handle) {
  if (handle == 0) return;
  TFE_DeleteTensorHandle(reinterpret_cast<TFE_TensorHandle*>(handle));
}

JNIEXPORT jlong JNICALL Java_org_tensorflow_EagerOperation_resolveTensorHandle(
    JNIEnv* env, jclass clazz, jlong handle) {
  TFE_TensorHandle* tensor_handle = requireTensorHandle(env, handle);
  if (tensor_handle == nullptr) return 0;
  TF_Status* status = TF_NewStatus();
  TF_Tensor* tensor = TFE_TensorHandleResolve(tensor_handle, status);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return 0;
  }
  TF_DeleteStatus(status);
  static_assert(sizeof(jlong) >= sizeof(TF_Tensor*),
                "Cannot represent a C TF_Tensor as a Java long");
  return reinterpret_cast<jlong>(tensor);
}

JNIEXPORT jint JNICALL Java_org_tensorflow_EagerOperation_outputListLength(
    JNIEnv* env, jclass clazz, jlong handle, jstring name) {
  TFE_Op* op = requireOp(env, handle);
  if (op == nullptr) return 0;
  TF_Status* status = TF_NewStatus();
  const char* cname = env->GetStringUTFChars(name, nullptr);
  int length = TFE_OpGetOutputLength(op, cname, status);
  env->ReleaseStringUTFChars(name, cname);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return 0;
  }
  TF_DeleteStatus(status);
  return static_cast<jint>(length);
}

JNIEXPORT jint JNICALL Java_org_tensorflow_EagerOperation_inputListLength(
    JNIEnv* env, jclass clazz, jlong handle, jstring name) {
  TFE_Op* op = requireOp(env, handle);
  if (op == nullptr) return 0;
  TF_Status* status = TF_NewStatus();
  const char* cname = env->GetStringUTFChars(name, nullptr);
  int length = TFE_OpGetInputLength(op, cname, status);
  env->ReleaseStringUTFChars(name, cname);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return 0;
  }
  TF_DeleteStatus(status);
  return static_cast<jint>(length);
}

JNIEXPORT jint JNICALL Java_org_tensorflow_EagerOperation_dataType(
    JNIEnv* env, jclass clazz, jlong handle) {
  TFE_TensorHandle* tensor_handle = requireTensorHandle(env, handle);
  if (tensor_handle == nullptr) return 0;
  TF_DataType data_type = TFE_TensorHandleDataType(tensor_handle);
  return static_cast<jint>(data_type);
}

JNIEXPORT jint JNICALL Java_org_tensorflow_EagerOperation_numDims(
    JNIEnv* env, jclass clazz, jlong handle) {
  TFE_TensorHandle* tensor_handle = requireTensorHandle(env, handle);
  if (tensor_handle == nullptr) return 0;
  TF_Status* status = TF_NewStatus();
  int num_dims = TFE_TensorHandleNumDims(tensor_handle, status);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return 0;
  }
  TF_DeleteStatus(status);
  return static_cast<jint>(num_dims);
}

JNIEXPORT jlong JNICALL Java_org_tensorflow_EagerOperation_dim(JNIEnv* env,
                                                               jclass clazz,
                                                               jlong handle,
                                                               jint dim_index) {
  TFE_TensorHandle* tensor_handle = requireTensorHandle(env, handle);
  if (tensor_handle == nullptr) return 0;
  TF_Status* status = TF_NewStatus();
  int64_t dim = TFE_TensorHandleDim(tensor_handle, dim_index, status);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return 0;
  }
  TF_DeleteStatus(status);
  return static_cast<jlong>(dim);
}
