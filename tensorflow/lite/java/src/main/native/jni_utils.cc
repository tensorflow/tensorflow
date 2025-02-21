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

#include "tensorflow/lite/java/src/main/native/jni_utils.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "tensorflow/lite/c/jni/jni_utils.h"

namespace tflite {
namespace jni {

const char kIllegalArgumentException[] = "java/lang/IllegalArgumentException";
const char kIllegalStateException[] = "java/lang/IllegalStateException";
const char kNullPointerException[] = "java/lang/NullPointerException";
const char kUnsupportedOperationException[] =
    "java/lang/UnsupportedOperationException";

void ThrowException(JNIEnv* env, const char* clazz, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  const size_t max_msg_len = 512;
  auto* message = static_cast<char*>(malloc(max_msg_len));
  if (message && (vsnprintf(message, max_msg_len, fmt, args) >= 0)) {
    env->ThrowNew(env->FindClass(clazz), message);
  } else {
    env->ThrowNew(env->FindClass(clazz), "");
  }
  if (message) {
    free(message);
  }
  va_end(args);
}

bool CheckJniInitializedOrThrow(JNIEnv* env) {
  return TfLiteCheckInitializedOrThrow(env);
}

BufferErrorReporter::BufferErrorReporter(JNIEnv* env, int limit) {
  buffer_ = new char[limit];
  if (!buffer_) {
    ThrowException(env, kNullPointerException,
                   "Internal error: Malloc of BufferErrorReporter to hold %d "
                   "char failed.",
                   limit);
    return;
  }
  buffer_[0] = '\0';
  start_idx_ = 0;
  end_idx_ = limit - 1;
}

BufferErrorReporter::~BufferErrorReporter() { delete[] buffer_; }

int BufferErrorReporter::Report(const char* format, va_list args) {
  int size = 0;
  // If an error has already been logged, insert a newline.
  if (start_idx_ > 0 && start_idx_ < end_idx_) {
    buffer_[start_idx_++] = '\n';
    ++size;
  }
  if (start_idx_ < end_idx_) {
    size = vsnprintf(buffer_ + start_idx_, end_idx_ - start_idx_, format, args);
  }
  start_idx_ += size;
  return size;
}

const char* BufferErrorReporter::CachedErrorMessage() { return buffer_; }

jobjectArray CreateStringArray(const std::vector<const char*>& values,
                               JNIEnv* env) {
  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) {
    ThrowException(env, tflite::jni::kUnsupportedOperationException,
                   "Internal error: Can not find java/lang/String class.");
    return nullptr;
  }

  jobjectArray results =
      env->NewObjectArray(values.size(), string_class, env->NewStringUTF(""));
  int i = 0;
  for (const char* value : values) {
    env->SetObjectArrayElement(results, i++, env->NewStringUTF(value));
  }
  return results;
}

bool AreDimsDifferent(JNIEnv* env, TfLiteTensor* tensor, const jintArray dims) {
  int num_dims = static_cast<int>(env->GetArrayLength(dims));
  jint* ptr = env->GetIntArrayElements(dims, nullptr);
  if (ptr == nullptr) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Empty dimensions of input array.");
    return true;
  }
  bool is_different = false;
  if (tensor->dims->size != num_dims) {
    is_different = true;
  } else {
    for (int i = 0; i < num_dims; ++i) {
      if (ptr[i] != tensor->dims->data[i]) {
        is_different = true;
        break;
      }
    }
  }
  env->ReleaseIntArrayElements(dims, ptr, JNI_ABORT);
  return is_different;
}

std::vector<int> ConvertJIntArrayToVector(JNIEnv* env, const jintArray inputs) {
  int size = static_cast<int>(env->GetArrayLength(inputs));
  std::vector<int> outputs(size, 0);
  jint* ptr = env->GetIntArrayElements(inputs, nullptr);
  if (ptr == nullptr) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Array has empty dimensions.");
    return {};
  }
  for (int i = 0; i < size; ++i) {
    outputs[i] = ptr[i];
  }
  env->ReleaseIntArrayElements(inputs, ptr, JNI_ABORT);
  return outputs;
}

}  // namespace jni
}  // namespace tflite
