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

#ifndef TENSORFLOW_LITE_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_
#define TENSORFLOW_LITE_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_

#include <jni.h>
#include <stdarg.h>

#include <vector>

#include "tensorflow/lite/error_reporter.h"

namespace tflite {
namespace jni {

extern const char kIllegalArgumentException[];
extern const char kIllegalStateException[];
extern const char kNullPointerException[];
extern const char kUnsupportedOperationException[];

/**
 * Thin wrapper around env->ThrowNew(...) that constructs the message using
 * printf-style formatting.
 *
 * Beware that if there is an exception already pending, then throwing
 * another exception may result in program termination, so it is good
 * practice to ensure that there is no pending exception before calling
 * this function.
 */
void ThrowException(JNIEnv* env, const char* clazz, const char* fmt, ...);

/**
 * Checks whether the necessary JNI infra has been initialized, throwing a Java
 * exception otherwise.
 *
 * @param env The JNIEnv for the current thread (which has to be attached to the
 *     JVM).
 * @return Whether or not the JNI infra has been initialized. If this method
 *     returns false, no other JNI method should be called until the pending
 *     exception has been handled (typically by returning to Java).
 */
bool CheckJniInitializedOrThrow(JNIEnv* env);

class BufferErrorReporter : public ErrorReporter {
 public:
  BufferErrorReporter(JNIEnv* env, int limit);
  ~BufferErrorReporter() override;
  int Report(const char* format, va_list args) override;
  const char* CachedErrorMessage();
  using ErrorReporter::Report;

 private:
  char* buffer_;
  int start_idx_ = 0;
  int end_idx_ = 0;
};

// Creates a Java string array from a C++ string vector.
jobjectArray CreateStringArray(const std::vector<const char*>& values,
                               JNIEnv* env);

// Checks the difference between tensor dimensions and given dimensions. Returns
// true if there is a difference, else false.
bool AreDimsDifferent(JNIEnv* env, TfLiteTensor* tensor, jintArray dims);

// Creates a C++ integer vector from a jintArray.
std::vector<int> ConvertJIntArrayToVector(JNIEnv* env, jintArray inputs);
// Converts a handle to a pointer of expected type.
template <typename T>
T* CastLongToPointer(JNIEnv* env, jlong handle) {
  if (handle == 0 || handle == -1) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Found invalid handle");
    return nullptr;
  }
  return reinterpret_cast<T*>(handle);
}

}  // namespace jni
}  // namespace tflite

#endif  // TENSORFLOW_LITE_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_
