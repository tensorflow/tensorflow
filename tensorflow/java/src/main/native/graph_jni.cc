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

#include "tensorflow/java/src/main/native/graph_jni.h"

#include <limits>
#include <memory>
#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/utils_jni.h"
#include "tensorflow/java/src/main/native/exception_jni.h"

namespace {
TF_Graph* requireHandle(JNIEnv* env, jlong handle) {
  static_assert(sizeof(jlong) >= sizeof(TF_Graph*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throwException(env, kIllegalStateException,
                   "close() has been called on the Graph");
    return nullptr;
  }
  return reinterpret_cast<TF_Graph*>(handle);
}
}  // namespace

JNIEXPORT jlong JNICALL Java_org_tensorflow_Graph_allocate(JNIEnv*, jclass) {
  return reinterpret_cast<jlong>(TF_NewGraph());
}

JNIEXPORT void JNICALL Java_org_tensorflow_Graph_delete(JNIEnv*, jclass,
                                                        jlong handle) {
  if (handle == 0) return;
  TF_DeleteGraph(reinterpret_cast<TF_Graph*>(handle));
}

JNIEXPORT jlong JNICALL Java_org_tensorflow_Graph_operation(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong handle,
                                                            jstring name) {
  TF_Graph* g = requireHandle(env, handle);
  if (g == nullptr) return 0;
  const char* cname = env->GetStringUTFChars(name, nullptr);
  TF_Operation* op = TF_GraphOperationByName(g, cname);
  env->ReleaseStringUTFChars(name, cname);
  return reinterpret_cast<jlong>(op);
}

JNIEXPORT jlongArray JNICALL Java_org_tensorflow_Graph_nextOperation(JNIEnv* env,
                                                                     jclass clazz,
                                                                     jlong handle,
                                                                     jint position) {
  TF_Graph* g = requireHandle(env, handle);
  if (g == nullptr) return nullptr;

  size_t pos = static_cast<size_t>(position);
  TF_Operation* operation = TF_GraphNextOperation(g, &pos);
  if (operation == nullptr) return nullptr;

  jlong handle_and_position[2];
  handle_and_position[0] = reinterpret_cast<jlong>(operation);
  handle_and_position[1] = static_cast<jlong>(pos);

  jlongArray rhett = env->NewLongArray(2);
  env->SetLongArrayRegion(rhett, 0, 2, handle_and_position);
  return rhett;
}

JNIEXPORT void JNICALL Java_org_tensorflow_Graph_importGraphDef(
    JNIEnv* env, jclass clazz, jlong handle, jbyteArray graph_def,
    jstring prefix) {
  TF_Graph* g = requireHandle(env, handle);
  if (g == nullptr) return;

  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();

  jboolean is_copy;
  const char* cprefix = env->GetStringUTFChars(prefix, &is_copy);
  TF_ImportGraphDefOptionsSetPrefix(opts, cprefix);
  env->ReleaseStringUTFChars(prefix, cprefix);

  static_assert(sizeof(jbyte) == 1, "unexpected size of the jbyte type");
  jbyte* bytes = env->GetByteArrayElements(graph_def, &is_copy);
  TF_Buffer* buf =
      TF_NewBufferFromString(bytes, env->GetArrayLength(graph_def));
  TF_Status* status = TF_NewStatus();

  TF_GraphImportGraphDef(g, buf, opts, status);
  throwExceptionIfNotOK(env, status);
  // Continue cleaning up resources even if an exception was thrown.

  TF_DeleteStatus(status);
  TF_DeleteBuffer(buf);
  env->ReleaseByteArrayElements(graph_def, bytes, JNI_ABORT);

  TF_DeleteImportGraphDefOptions(opts);
}

JNIEXPORT jbyteArray JNICALL
Java_org_tensorflow_Graph_toGraphDef(JNIEnv* env, jclass clazz, jlong handle) {
  jbyteArray ret = nullptr;
  TF_Graph* g = requireHandle(env, handle);
  if (g == nullptr) return ret;

  TF_Buffer* buf = TF_NewBuffer();
  TF_Status* status = TF_NewStatus();
  TF_GraphToGraphDef(g, buf, status);
  if (throwExceptionIfNotOK(env, status)) {
    // sizeof(jsize) is less than sizeof(size_t) on some platforms.
    if (buf->length > std::numeric_limits<jint>::max()) {
      throwException(env, kIndexOutOfBoundsException,
                     "GraphDef is too large to serialize into a byte[] array");
    } else {
      static_assert(sizeof(jbyte) == 1, "unexpected size of the jbyte type");
      jint ret_len = static_cast<jint>(buf->length);
      ret = env->NewByteArray(ret_len);
      env->SetByteArrayRegion(ret, 0, ret_len,
                              static_cast<const jbyte*>(buf->data));
    }
  }
  TF_DeleteStatus(status);
  TF_DeleteBuffer(buf);
  return ret;
}

JNIEXPORT jlongArray JNICALL Java_org_tensorflow_Graph_addGradients(
    JNIEnv* env, jclass clazz, jlong handle, jstring prefix,
    jlongArray y_handles, jintArray y_indices, jlongArray x_handles,
    jintArray x_indices, jlongArray dx_handles, jintArray dx_indices) {
  TF_Graph* g = requireHandle(env, handle);
  if (g == nullptr) return nullptr;

  const jint ny = env->GetArrayLength(y_handles);
  const jint nx = env->GetArrayLength(x_handles);

  std::unique_ptr<TF_Output[]> y(new TF_Output[ny]);
  std::unique_ptr<TF_Output[]> x(new TF_Output[nx]);
  std::unique_ptr<TF_Output[]> dx(nullptr);
  std::unique_ptr<TF_Output[]> dy(new TF_Output[nx]);

  resolveOutputs(env, "y", y_handles, y_indices, y.get(), ny);
  resolveOutputs(env, "x", x_handles, x_indices, x.get(), nx);
  if (dx_handles != nullptr) {
    if (env->GetArrayLength(dx_handles) != ny) {
      throwException(env, kIllegalArgumentException,
                     "expected %d, got %d dx handles", ny,
                     env->GetArrayLength(dx_handles));
    }
    dx.reset(new TF_Output[ny]);
    resolveOutputs(env, "dx", dx_handles, dx_indices, dx.get(), ny);
  }
  if (env->ExceptionCheck()) return nullptr;

  const char* cprefix = nullptr;
  if (prefix != nullptr) {
    cprefix = env->GetStringUTFChars(prefix, nullptr);
  }
  TF_Status* status = TF_NewStatus();
  TF_AddGradientsWithPrefix(g, cprefix, y.get(), ny, x.get(), nx, dx.get(),
                            status, dy.get());
  if (prefix != nullptr) {
    env->ReleaseStringUTFChars(prefix, cprefix);
  }
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);

  // returned array contains both op handles and output indices, in pair
  jlongArray dy_handles_and_indices = env->NewLongArray(nx << 1);
  jlong* dy_elems = env->GetLongArrayElements(dy_handles_and_indices, nullptr);
  for (int i = 0, j = nx; i < nx; ++i, ++j) {
    TF_Output dy_output = dy.get()[i];
    dy_elems[i] = reinterpret_cast<jlong>(dy_output.oper);
    dy_elems[j] = static_cast<jlong>(dy_output.index);
  }
  env->ReleaseLongArrayElements(dy_handles_and_indices, dy_elems, 0);

  return dy_handles_and_indices;
}
