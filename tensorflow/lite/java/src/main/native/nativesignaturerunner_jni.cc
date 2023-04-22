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

#include "tensorflow/lite/core/shims/cc/interpreter.h"
#include "tensorflow/lite/java/src/main/native/jni_utils.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/util.h"

using tflite::jni::ThrowException;
using tflite_shims::Interpreter;

#ifndef TFLITE_DISABLE_SELECT_JAVA_APIS
namespace tflite {
// A helper class to access private information of SignatureRunner class.
class SignatureRunnerJNIHelper {
 public:
  explicit SignatureRunnerJNIHelper(SignatureRunner* runner)
      : signature_runner_(runner) {}

  // Gets the subgraph index associated with this SignatureRunner.
  int GetSubgraphIndex() {
    if (!signature_runner_) return -1;

    return signature_runner_->signature_def_->subgraph_index;
  }

  // Gets the tensor index of a given input.
  int GetInputTensorIndex(const char* input_name) {
    const auto& it = signature_runner_->signature_def_->inputs.find(input_name);
    if (it == signature_runner_->signature_def_->inputs.end()) {
      return -1;
    }
    return it->second;
  }

  // Gets the tensor index of a given output.
  int GetOutputTensorIndex(const char* output_name) {
    const auto& it =
        signature_runner_->signature_def_->outputs.find(output_name);
    if (it == signature_runner_->signature_def_->outputs.end()) {
      return -1;
    }
    return it->second;
  }

  // Gets the index of the input specified by `input_name`.
  int GetInputIndex(const char* input_name) {
    int input_tensor_index = GetInputTensorIndex(input_name);
    if (input_tensor_index == -1) return -1;

    int count = 0;
    for (int tensor_idx : signature_runner_->subgraph_->inputs()) {
      if (input_tensor_index == tensor_idx) return count;
      ++count;
    }
    return -1;
  }

  // Gets the index of the output specified by `output_name`.
  int GetOutputIndex(const char* output_name) {
    int output_tensor_index = GetOutputTensorIndex(output_name);
    if (output_tensor_index == -1) return -1;

    int count = 0;
    for (int tensor_idx : signature_runner_->subgraph_->outputs()) {
      if (output_tensor_index == tensor_idx) return count;
      ++count;
    }
    return -1;
  }

 private:
  SignatureRunner* signature_runner_;
};
}  // namespace tflite

using tflite::SignatureRunner;
using tflite::SignatureRunnerJNIHelper;
using tflite::jni::BufferErrorReporter;
using tflite::jni::CastLongToPointer;
using tflite_shims::Interpreter;
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS

extern "C" {
JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetSignatureRunner(
    JNIEnv* env, jclass clazz, jlong handle, jstring signature_key) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeGetSignatureRunner");
  return -1;
#else
  Interpreter* interpreter = CastLongToPointer<Interpreter>(env, handle);
  if (interpreter == nullptr) return -1;
  const char* signature_key_ptr =
      env->GetStringUTFChars(signature_key, nullptr);

  SignatureRunner* runner = interpreter->GetSignatureRunner(signature_key_ptr);
  if (runner == nullptr) {
    // Release the memory before returning.
    env->ReleaseStringUTFChars(signature_key, signature_key_ptr);
    return -1;
  }

  // Release the memory before returning.
  env->ReleaseStringUTFChars(signature_key, signature_key_ptr);
  return reinterpret_cast<jlong>(runner);
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetSubgraphIndex(
    JNIEnv* env, jclass clazz, jlong handle) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  TFLITE_LOG(tflite::TFLITE_LOG_WARNING,
             "Not supported: nativeGetSubgraphIndex");
  return -1;
#else
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return -1;
  return SignatureRunnerJNIHelper(runner).GetSubgraphIndex();
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeInputNames(
    JNIEnv* env, jclass clazz, jlong handle) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeInputNames");
  return nullptr;
#else
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return nullptr;
  return tflite::jni::CreateStringArray(runner->input_names(), env);
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeOutputNames(
    JNIEnv* env, jclass clazz, jlong handle) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeOutputNames");
  return nullptr;
#else
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return nullptr;
  return tflite::jni::CreateStringArray(runner->output_names(), env);
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetInputIndex(
    JNIEnv* env, jclass clazz, jlong handle, jstring input_name) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeGetInputIndex");
  return -1;
#else
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return -1;
  const char* input_name_ptr = env->GetStringUTFChars(input_name, nullptr);
  int index = SignatureRunnerJNIHelper(runner).GetInputIndex(input_name_ptr);
  // Release the memory before returning.
  env->ReleaseStringUTFChars(input_name, input_name_ptr);
  return index;
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetOutputIndex(
    JNIEnv* env, jclass clazz, jlong handle, jstring output_name) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeGetOutputIndex");
  return -1;
#else
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return -1;
  const char* output_name_ptr = env->GetStringUTFChars(output_name, nullptr);
  int index = SignatureRunnerJNIHelper(runner).GetOutputIndex(output_name_ptr);
  // Release the memory before returning.
  env->ReleaseStringUTFChars(output_name, output_name_ptr);
  return index;
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeResizeInput(
    JNIEnv* env, jclass clazz, jlong handle, jlong error_handle,
    jstring input_name, jintArray dims) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeResizeInput");
  return -1;
#else
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  BufferErrorReporter* error_reporter =
      CastLongToPointer<BufferErrorReporter>(env, error_handle);
  if (runner == nullptr || error_reporter == nullptr) return JNI_FALSE;
  // Check whether it is resizing with the same dimensions.
  const char* input_name_ptr = env->GetStringUTFChars(input_name, nullptr);
  TfLiteTensor* target = runner->input_tensor(input_name_ptr);
  if (target == nullptr) {
    // Release the memory before returning.
    env->ReleaseStringUTFChars(input_name, input_name_ptr);
    return JNI_FALSE;
  }
  bool is_changed = tflite::jni::AreDimsDifferent(env, target, dims);
  if (is_changed) {
    TfLiteStatus status = runner->ResizeInputTensor(
        input_name_ptr, tflite::jni::ConvertJIntArrayToVector(env, dims));
    if (status != kTfLiteOk) {
      ThrowException(env, tflite::jni::kIllegalArgumentException,
                     "Internal error: Failed to resize input %s: %s",
                     input_name_ptr, error_reporter->CachedErrorMessage());
      // Release the memory before returning.
      env->ReleaseStringUTFChars(input_name, input_name_ptr);
      return JNI_FALSE;
    }
  }
  // Release the memory before returning.
  env->ReleaseStringUTFChars(input_name, input_name_ptr);
  return is_changed ? JNI_TRUE : JNI_FALSE;
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeAllocateTensors(
    JNIEnv* env, jclass clazz, jlong handle, jlong error_handle) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeAllocateTensors");
#else

  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  BufferErrorReporter* error_reporter =
      CastLongToPointer<BufferErrorReporter>(env, error_handle);
  if (runner == nullptr || error_reporter == nullptr) return;

  if (runner->AllocateTensors() != kTfLiteOk) {
    ThrowException(
        env, tflite::jni::kIllegalStateException,
        "Internal error: Unexpected failure when preparing tensor allocations:"
        " %s",
        error_reporter->CachedErrorMessage());
  }
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeInvoke(
    JNIEnv* env, jclass clazz, jlong handle, jlong error_handle) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: nativeInvoke");
#else

  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  BufferErrorReporter* error_reporter =
      CastLongToPointer<BufferErrorReporter>(env, error_handle);
  if (runner == nullptr || error_reporter == nullptr) return;

  if (runner->Invoke() != kTfLiteOk) {
    ThrowException(env, tflite::jni::kIllegalStateException,
                   "Internal error: Failed to run on the given Interpreter: %s",
                   error_reporter->CachedErrorMessage());
  }
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}
}  // extern "C"
