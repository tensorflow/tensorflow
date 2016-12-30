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

#include <stdarg.h>

#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"

const char kIllegalArgumentException[] = "java/lang/IllegalArgumentException";
const char kIllegalStateException[] = "java/lang/IllegalStateException";
const char kNullPointerException[] = "java/lang/NullPointerException";
const char kIndexOutOfBoundsException[] = "java/lang/IndexOutOfBoundsException";
const char kUnsupportedOperationException[] =
    "java/lang/UnsupportedOperationException";

void throwException(JNIEnv* env, const char* clazz, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  char* message = nullptr;
  if (vasprintf(&message, fmt, args) >= 0) {
    env->ThrowNew(env->FindClass(clazz), message);
  } else {
    env->ThrowNew(env->FindClass(clazz), "");
  }
  va_end(args);
}

namespace {
// Map TF_Codes to unchecked exceptions.
const char* exceptionClassName(TF_Code code) {
  switch (code) {
    case TF_OK:
      return nullptr;
    case TF_INVALID_ARGUMENT:
      return kIllegalArgumentException;
    case TF_UNAUTHENTICATED:
    case TF_PERMISSION_DENIED:
      return "java/lang/SecurityException";
    case TF_RESOURCE_EXHAUSTED:
    case TF_FAILED_PRECONDITION:
      return kIllegalStateException;
    case TF_OUT_OF_RANGE:
      return kIndexOutOfBoundsException;
    case TF_UNIMPLEMENTED:
      return kUnsupportedOperationException;
    default:
      return "org/tensorflow/TensorFlowException";
  }
}
}  // namespace

bool throwExceptionIfNotOK(JNIEnv* env, const TF_Status* status) {
  const char* clazz = exceptionClassName(TF_GetCode(status));
  if (clazz == nullptr) return true;
  env->ThrowNew(env->FindClass(clazz), TF_Message(status));
  return false;
}
