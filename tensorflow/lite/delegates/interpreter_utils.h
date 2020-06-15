/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_INTERPRETER_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_INTERPRETER_UTILS_H_

#include "tensorflow/lite/interpreter.h"

// Utility functions and classes for using delegates.

namespace tflite {
namespace delegates {
#if !TFLITE_EXPERIMENTAL_RUNTIME_EAGER
class InterpreterUtils {
 public:
  /// Invokes an interpreter with automatic fallback from delegation to CPU.
  ///
  /// If using the delegate fails, the delegate is automatically undone and an
  /// attempt made to return the interpreter to an invokable state.
  ///
  /// Allowing the fallback is suitable only if both of the following hold:
  /// - The caller is known not to cache pointers to tensor data across Invoke()
  ///   calls.
  /// - The model is not stateful (no variables, no LSTMs) or the state isn't
  ///   needed between batches.
  ///
  /// Returns one of the following three status codes:
  /// 1. kTfLiteOk: Success. Output is valid.
  /// 2. kTfLiteDelegateError: Delegate error but fallback succeeded. Output is
  /// valid.
  /// NOTE: This undoes all delegates previously applied to the Interpreter.
  /// 3. kTfLiteError: Unexpected/runtime failure. Output is invalid.
  /// WARNING: This is an experimental API and subject to change.
  static TfLiteStatus InvokeWithCPUFallback(Interpreter* interpreter);
};
#endif  // !TFLITE_EXPERIMENTAL_RUNTIME_EAGER
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_INTERPRETER_UTILS_H_
