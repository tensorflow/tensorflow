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

#include <string.h>
#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"
#include "tensorflow/java/src/main/native/session_jni.h"

namespace {
TF_Session* requireHandle(JNIEnv* env, jlong handle) {
  static_assert(sizeof(jlong) >= sizeof(TF_Session*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throwException(env, kNullPointerException,
                   "close() has been called on the Session");
    return nullptr;
  }
  return reinterpret_cast<TF_Session*>(handle);
}

template <class T>
void resolveHandles(JNIEnv* env, const char* type, jlongArray src_array,
                    T** dst, jint n) {
  if (env->ExceptionCheck()) return;
  jint len = env->GetArrayLength(src_array);
  if (len != n) {
    throwException(env, kIllegalArgumentException, "expected %d, got %d %s", n,
                   len, type);
    return;
  }
  jlong* src_start = env->GetLongArrayElements(src_array, nullptr);
  jlong* src = src_start;
  for (int i = 0; i < n; ++i, ++src, ++dst) {
    if (*src == 0) {
      throwException(env, kNullPointerException, "invalid %s (#%d of %d)", type,
                     i, n);
      break;
    }
    *dst = reinterpret_cast<T*>(*src);
  }
  env->ReleaseLongArrayElements(src_array, src_start, JNI_ABORT);
}

void resolveOutputs(JNIEnv* env, const char* type, jlongArray src_op,
                    jintArray src_index, TF_Output* dst, jint n) {
  if (env->ExceptionCheck()) return;
  jint len = env->GetArrayLength(src_op);
  if (len != n) {
    throwException(env, kIllegalArgumentException,
                   "expected %d, got %d %s Operations", n, len, type);
    return;
  }
  len = env->GetArrayLength(src_index);
  if (len != n) {
    throwException(env, kIllegalArgumentException,
                   "expected %d, got %d %s Operation output indices", n, len,
                   type);
    return;
  }
  jlong* op_handles = env->GetLongArrayElements(src_op, nullptr);
  jint* indices = env->GetIntArrayElements(src_index, nullptr);
  for (int i = 0; i < n; ++i) {
    if (op_handles[i] == 0) {
      throwException(env, kNullPointerException, "invalid %s (#%d of %d)", type,
                     i, n);
      break;
    }
    dst[i] = TF_Output{reinterpret_cast<TF_Operation*>(op_handles[i]),
                       static_cast<int>(indices[i])};
  }
  env->ReleaseIntArrayElements(src_index, indices, JNI_ABORT);
  env->ReleaseLongArrayElements(src_op, op_handles, JNI_ABORT);
}

void TF_MaybeDeleteBuffer(TF_Buffer* buf) {
  if (buf == nullptr) return;
  TF_DeleteBuffer(buf);
}

typedef std::unique_ptr<TF_Buffer, decltype(&TF_MaybeDeleteBuffer)>
    unique_tf_buffer;

unique_tf_buffer MakeUniqueBuffer(TF_Buffer* buf) {
  return unique_tf_buffer(buf, TF_MaybeDeleteBuffer);
}
}  // namespace

JNIEXPORT jlong JNICALL Java_org_tensorflow_Session_allocate(
    JNIEnv* env, jclass clazz, jlong graph_handle) {
  if (graph_handle == 0) {
    throwException(env, kNullPointerException, "Graph has been close()d");
    return 0;
  }
  TF_Graph* graph = reinterpret_cast<TF_Graph*>(graph_handle);
  TF_Status* status = TF_NewStatus();
  TF_SessionOptions* opts = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graph, opts, status);
  TF_DeleteSessionOptions(opts);
  bool ok = throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);

  return ok ? reinterpret_cast<jlong>(session) : 0;
}

JNIEXPORT void JNICALL Java_org_tensorflow_Session_delete(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle) {
  TF_Session* session = requireHandle(env, handle);
  if (session == nullptr) return;
  TF_Status* status = TF_NewStatus();
  TF_CloseSession(session, status);
  // Result of close is ignored, delete anyway.
  TF_DeleteSession(session, status);
  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
}

JNIEXPORT jbyteArray JNICALL Java_org_tensorflow_Session_run(
    JNIEnv* env, jclass clazz, jlong handle, jbyteArray jrun_options,
    jlongArray input_tensor_handles, jlongArray input_op_handles,
    jintArray input_op_indices, jlongArray output_op_handles,
    jintArray output_op_indices, jlongArray target_op_handles,
    jboolean want_run_metadata, jlongArray output_tensor_handles) {
  TF_Session* session = requireHandle(env, handle);
  if (session == nullptr) return nullptr;

  const jint ninputs = env->GetArrayLength(input_tensor_handles);
  const jint noutputs = env->GetArrayLength(output_tensor_handles);
  const jint ntargets = env->GetArrayLength(target_op_handles);

  std::unique_ptr<TF_Output[]> inputs(new TF_Output[ninputs]);
  std::unique_ptr<TF_Tensor* []> input_values(new TF_Tensor*[ninputs]);
  std::unique_ptr<TF_Output[]> outputs(new TF_Output[noutputs]);
  std::unique_ptr<TF_Tensor* []> output_values(new TF_Tensor*[noutputs]);
  std::unique_ptr<TF_Operation* []> targets(new TF_Operation*[ntargets]);
  unique_tf_buffer run_metadata(
      MakeUniqueBuffer(want_run_metadata ? TF_NewBuffer() : nullptr));

  resolveHandles(env, "input Tensors", input_tensor_handles, input_values.get(),
                 ninputs);
  resolveOutputs(env, "input", input_op_handles, input_op_indices, inputs.get(),
                 ninputs);
  resolveOutputs(env, "output", output_op_handles, output_op_indices,
                 outputs.get(), noutputs);
  resolveHandles(env, "target Operations", target_op_handles, targets.get(),
                 ntargets);
  if (env->ExceptionCheck()) return nullptr;

  TF_Status* status = TF_NewStatus();

  unique_tf_buffer run_options(MakeUniqueBuffer(nullptr));
  jbyte* jrun_options_data = nullptr;
  if (jrun_options != nullptr) {
    size_t sz = env->GetArrayLength(jrun_options);
    if (sz > 0) {
      jrun_options_data = env->GetByteArrayElements(jrun_options, nullptr);
      run_options.reset(
          TF_NewBufferFromString(static_cast<void*>(jrun_options_data), sz));
    }
  }

  TF_SessionRun(session, run_options.get(), inputs.get(), input_values.get(),
                static_cast<int>(ninputs), outputs.get(), output_values.get(),
                static_cast<int>(noutputs), targets.get(),
                static_cast<int>(ntargets), run_metadata.get(), status);

  if (jrun_options_data != nullptr) {
    env->ReleaseByteArrayElements(jrun_options, jrun_options_data, JNI_ABORT);
  }

  if (!throwExceptionIfNotOK(env, status)) {
    return nullptr;
  }
  jlong* t = env->GetLongArrayElements(output_tensor_handles, nullptr);
  for (int i = 0; i < noutputs; ++i) {
    t[i] = reinterpret_cast<jlong>(output_values[i]);
  }
  env->ReleaseLongArrayElements(output_tensor_handles, t, 0);

  jbyteArray ret = nullptr;
  if (run_metadata != nullptr) {
    ret = env->NewByteArray(run_metadata->length);
    jbyte* elems = env->GetByteArrayElements(ret, nullptr);
    memcpy(elems, run_metadata->data, run_metadata->length);
    env->ReleaseByteArrayElements(ret, elems, JNI_COMMIT);
  }
  return ret;
}
