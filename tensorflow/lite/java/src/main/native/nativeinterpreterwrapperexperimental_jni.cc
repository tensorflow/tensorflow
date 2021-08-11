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

#include <dlfcn.h>
#include <jni.h>
#include <stdio.h>
#include <time.h>

#include <atomic>
#include <map>
#include <utility>
#include <vector>

#include "tensorflow/lite/core/shims/cc/interpreter.h"
#include "tensorflow/lite/java/src/main/native/jni_utils.h"

using tflite::jni::BufferErrorReporter;
using tflite::jni::ThrowException;
using tflite_shims::Interpreter;

namespace {

Interpreter* convertLongToInterpreter(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Invalid handle to Interpreter.");
    return nullptr;
  }
  return reinterpret_cast<Interpreter*>(handle);
}

BufferErrorReporter* convertLongToErrorReporter(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Invalid handle to ErrorReporter.");
    return nullptr;
  }
  return reinterpret_cast<BufferErrorReporter*>(handle);
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapperExperimental_resetVariableTensors(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return;

  Interpreter* interpreter = convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return;

  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return;

  TfLiteStatus status = interpreter->ResetVariableTensors();
  if (status != kTfLiteOk) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Failed to reset variable tensors: %s",
                   error_reporter->CachedErrorMessage());
  }
}

}  // extern "C"
