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

#include "tensorflow/java/src/main/native/utils_jni.h"

#include "tensorflow/java/src/main/native/exception_jni.h"

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




