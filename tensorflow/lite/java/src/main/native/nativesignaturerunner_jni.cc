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

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/java/src/main/native/jni_utils.h"
#include "tensorflow/lite/signature_runner.h"

#ifdef TFLITE_DISABLE_SELECT_JAVA_APIS
#include <algorithm>

#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#endif

using tflite::Interpreter;
using tflite::jni::ThrowException;

namespace tflite {
// A helper class to access private information of SignatureRunner class.
class SignatureRunnerJNIHelper {
 public:
  explicit SignatureRunnerJNIHelper(SignatureRunner* runner)
      : signature_runner_(runner) {}

#ifndef TFLITE_DISABLE_SELECT_JAVA_APIS
  // Attempts to get the subgraph index associated with this SignatureRunner.
  // Returns the subgraph index, if available, or -1 on error.
  int GetSubgraphIndex() {
    if (!signature_runner_) return -1;

    return signature_runner_->signature_def_->subgraph_index;
  }
#endif

 public:
  // Gets the index of the input specified by `input_name`.
  int GetInputIndex(const char* input_name) {
#ifndef TFLITE_DISABLE_SELECT_JAVA_APIS
    int input_tensor_index = GetInputTensorIndex(input_name);
    if (input_tensor_index == -1) return -1;

    int count = 0;
    for (int tensor_idx : signature_runner_->subgraph_->inputs()) {
      if (input_tensor_index == tensor_idx) return count;
      ++count;
    }
    return -1;
#else
    const auto& inputs = signature_runner_->input_names();
    const auto& it = std::find(inputs.begin(), inputs.end(), input_name);
    if (it == inputs.end()) {
      return -1;
    }
    return it - inputs.begin();
#endif
  }

  // Gets the index of the output specified by `output_name`.
  int GetOutputIndex(const char* output_name) {
#ifndef TFLITE_DISABLE_SELECT_JAVA_APIS
    int output_tensor_index = GetOutputTensorIndex(output_name);
    if (output_tensor_index == -1) return -1;

    int count = 0;
    for (int tensor_idx : signature_runner_->subgraph_->outputs()) {
      if (output_tensor_index == tensor_idx) return count;
      ++count;
    }
    return -1;
#else
    const auto& outputs = signature_runner_->output_names();
    const auto& it = std::find(outputs.begin(), outputs.end(), output_name);
    if (it == outputs.end()) {
      return -1;
    }
    return it - outputs.begin();
#endif
  }

 private:
#ifndef TFLITE_DISABLE_SELECT_JAVA_APIS
  // Gets the tensor index of a given input.
  int GetInputTensorIndex(const char* input_name) {
    const auto& inputs = signature_runner_->signature_def_->inputs;
    const auto& it = inputs.find(input_name);
    if (it == inputs.end()) {
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
#endif  // ndef TFLITE_DISABLE_SELECT_JAVA_APIS

  SignatureRunner* signature_runner_;
};
}  // namespace tflite

using tflite::Interpreter;
using tflite::SignatureRunner;
using tflite::SignatureRunnerJNIHelper;
using tflite::jni::BufferErrorReporter;
using tflite::jni::CastLongToPointer;

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetSignatureRunner(
    JNIEnv* env, jclass clazz, jlong handle, jstring signature_key) {
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
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return nullptr;
  return tflite::jni::CreateStringArray(runner->input_names(), env);
}

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeOutputNames(
    JNIEnv* env, jclass clazz, jlong handle) {
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return nullptr;
  return tflite::jni::CreateStringArray(runner->output_names(), env);
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetInputIndex(
    JNIEnv* env, jclass clazz, jlong handle, jstring input_name) {
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return -1;
  const char* input_name_ptr = env->GetStringUTFChars(input_name, nullptr);
  int index = SignatureRunnerJNIHelper(runner).GetInputIndex(input_name_ptr);
  // Release the memory before returning.
  env->ReleaseStringUTFChars(input_name, input_name_ptr);
  return index;
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeGetOutputIndex(
    JNIEnv* env, jclass clazz, jlong handle, jstring output_name) {
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  if (runner == nullptr) return -1;
  const char* output_name_ptr = env->GetStringUTFChars(output_name, nullptr);
  int index = SignatureRunnerJNIHelper(runner).GetOutputIndex(output_name_ptr);
  // Release the memory before returning.
  env->ReleaseStringUTFChars(output_name, output_name_ptr);
  return index;
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeResizeInput(
    JNIEnv* env, jclass clazz, jlong handle, jlong error_handle,
    jstring input_name, jintArray dims) {
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
#if TFLITE_DISABLE_SELECT_JAVA_APIS
    // In this case, only unknown dimensions (those with size = -1)
    // can be resized.
    TfLiteStatus status = runner->ResizeInputTensorStrict(
        input_name_ptr, tflite::jni::ConvertJIntArrayToVector(env, dims));
#else
    TfLiteStatus status = runner->ResizeInputTensor(
        input_name_ptr, tflite::jni::ConvertJIntArrayToVector(env, dims));
#endif
    if (status != kTfLiteOk) {
      ThrowException(env, tflite::jni::kIllegalArgumentException,
                     "Error: Failed to resize input %s: %s", input_name_ptr,
                     error_reporter->CachedErrorMessage());
      // Release the memory before returning.
      env->ReleaseStringUTFChars(input_name, input_name_ptr);
      return JNI_FALSE;
    }
  }
  // Release the memory before returning.
  env->ReleaseStringUTFChars(input_name, input_name_ptr);
  return is_changed ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeAllocateTensors(
    JNIEnv* env, jclass clazz, jlong handle, jlong error_handle) {
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
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeSignatureRunnerWrapper_nativeInvoke(
    JNIEnv* env, jclass clazz, jlong handle, jlong error_handle) {
  SignatureRunner* runner = CastLongToPointer<SignatureRunner>(env, handle);
  BufferErrorReporter* error_reporter =
      CastLongToPointer<BufferErrorReporter>(env, error_handle);
  if (runner == nullptr || error_reporter == nullptr) return;

  if (runner->Invoke() != kTfLiteOk) {
    ThrowException(env, tflite::jni::kIllegalStateException,
                   "Internal error: Failed to run on the given Interpreter: %s",
                   error_reporter->CachedErrorMessage());
  }
}

}  // extern "C"
