/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <jni.h>

#include "tensorflow/lite/core/shims/c/common.h"
#include "tensorflow/lite/java/src/main/native/jni_utils.h"

using tflite::jni::CastLongToPointer;

namespace {

using DeleteFunction = void(TfLiteOpaqueDelegate*);

TfLiteOpaqueDelegate* convertLongToDelegate(JNIEnv* env,
                                            jlong delegate_handle) {
  return CastLongToPointer<TfLiteOpaqueDelegate>(env, delegate_handle);
}

DeleteFunction* convertLongToDeleteFunction(JNIEnv* env,
                                            jlong delete_function) {
  return CastLongToPointer<DeleteFunction>(env, delete_function);
}

}  // anonymous namespace.

extern "C" {

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_XnnpackDelegate_applyDeleteFunction(
    JNIEnv* env, jclass clazz, jlong delete_function_handle,
    jlong delegate_handle) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return;

  TfLiteOpaqueDelegate* delegate = convertLongToDelegate(env, delegate_handle);
  if (delegate == nullptr) return;
  DeleteFunction* delete_function =
      convertLongToDeleteFunction(env, delete_function_handle);
  if (delete_function == nullptr) return;
  delete_function(delegate);
}

}  // extern "C"
