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

#include "tensorflow/java/src/main/native/graph_operation_jni.h"
#include <memory>
#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"

namespace {
template <class T>
T* requireHandleImpl(JNIEnv* env, jlong handle) {
  static_assert(sizeof(jlong) >= sizeof(T*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throwException(
        env, kNullPointerException,
        "close() has been called on the Graph this Operation was a part of");
    return nullptr;
  }
  return reinterpret_cast<T*>(handle);
}

TF_Operation* requireHandle(JNIEnv* env, jlong handle) {
  return requireHandleImpl<TF_Operation>(env, handle);
}

TF_Graph* requireGraphHandle(JNIEnv* env, jlong handle) {
  return requireHandleImpl<TF_Graph>(env, handle);
}
}  // namespace

JNIEXPORT jstring JNICALL Java_org_tensorflow_GraphOperation_name(
    JNIEnv* env, jclass clazz, jlong handle) {
  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return nullptr;
  return env->NewStringUTF(TF_OperationName(op));
}

JNIEXPORT jstring JNICALL Java_org_tensorflow_GraphOperation_type(
    JNIEnv* env, jclass clazz, jlong handle) {
  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return nullptr;
  return env->NewStringUTF(TF_OperationOpType(op));
}

JNIEXPORT jint JNICALL Java_org_tensorflow_GraphOperation_numOutputs(
    JNIEnv* env, jclass clazz, jlong handle) {
  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return 0;
  return TF_OperationNumOutputs(op);
}

JNIEXPORT jint JNICALL Java_org_tensorflow_GraphOperation_outputListLength(
    JNIEnv* env, jclass clazz, jlong handle, jstring name) {
  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return 0;

  TF_Status* status = TF_NewStatus();

  const char* cname = env->GetStringUTFChars(name, nullptr);
  int result = TF_OperationOutputListLength(op, cname, status);
  env->ReleaseStringUTFChars(name, cname);

  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
  return result;
}

JNIEXPORT jlongArray JNICALL Java_org_tensorflow_GraphOperation_shape(
    JNIEnv* env, jclass clazz, jlong graph_handle, jlong op_handle,
    jint output_index) {
  TF_Graph* graph = requireGraphHandle(env, graph_handle);
  if (graph == nullptr) return nullptr;
  TF_Operation* op = requireHandle(env, op_handle);
  if (op == nullptr) return nullptr;

  int num_outputs = TF_OperationNumOutputs(op);
  if (output_index < 0 || output_index >= num_outputs) {
    throwException(
        env, kIndexOutOfBoundsException,
        "invalid output index (%d) for an operation that has %d outputs",
        output_index, num_outputs);
    return nullptr;
  }

  TF_Output output{op, output_index};
  TF_Status* status = TF_NewStatus();
  jsize num_dims = TF_GraphGetTensorNumDims(graph, output, status);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  if (num_dims < 0) return nullptr;
  static_assert(sizeof(jlong) == sizeof(int64_t),
                "Java long is not compatible with the TensorFlow C API");
  // One might have trivially wanted to do:
  // TF_GraphGetTensorShape(graph, output, static_cast<int64_t*>(dims), ...)
  // but on some platforms this fails with:
  // static_cast from 'jlong *' (aka 'long *') to 'int64_t *' (aka 'long long
  // *') is not allowed
  // For now, do the expensive but safe thing of copying.
  std::unique_ptr<int64_t[]> cdims(new int64_t[num_dims]);
  TF_GraphGetTensorShape(graph, output, cdims.get(), static_cast<int>(num_dims),
                         status);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  jlongArray ret = env->NewLongArray(num_dims);
  jlong* dims = env->GetLongArrayElements(ret, nullptr);
  for (int i = 0; i < num_dims; ++i) {
    dims[i] = static_cast<jlong>(cdims[i]);
  }
  env->ReleaseLongArrayElements(ret, dims, 0);
  return ret;
}

JNIEXPORT jint JNICALL Java_org_tensorflow_GraphOperation_dtype(
    JNIEnv* env, jclass clazz, jlong graph_handle, jlong op_handle,
    jint output_index) {
  TF_Graph* graph = requireGraphHandle(env, graph_handle);
  if (graph == nullptr) return 0;
  TF_Operation* op = requireHandle(env, op_handle);
  if (op == nullptr) return 0;

  int num_outputs = TF_OperationNumOutputs(op);
  if (output_index < 0 || output_index >= num_outputs) {
    throwException(
        env, kIndexOutOfBoundsException,
        "invalid output index (%d) for an operation that has %d outputs",
        output_index, num_outputs);
    return 0;
  }

  return static_cast<jint>(TF_OperationOutputType(TF_Output{op, output_index}));
}

JNIEXPORT jint JNICALL Java_org_tensorflow_GraphOperation_inputListLength(
    JNIEnv* env, jclass clazz, jlong handle, jstring name) {
  TF_Operation* op = requireHandle(env, handle);
  if (op == nullptr) return 0;

  TF_Status* status = TF_NewStatus();

  const char* cname = env->GetStringUTFChars(name, nullptr);
  int result = TF_OperationInputListLength(op, cname, status);
  env->ReleaseStringUTFChars(name, cname);

  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
  return result;
}
