/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
/// \file
/// Provides options to an interpreter.
///
#ifndef TENSORFLOW_LITE_INTERPRETER_OPTIONS_H_
#define TENSORFLOW_LITE_INTERPRETER_OPTIONS_H_

namespace tflite {

/// Options class for `Interpreter`.
/// WARNING: This is an experimental API and subject to change.
class InterpreterOptions {
 public:
  InterpreterOptions()
      : experimental_preserve_all_tensors_(false),
        experimental_ensure_dynamic_tensors_are_released_(false),
        experimental_optimize_memory_for_large_tensors_(0) {}

  /// Preserving all intermediates tensors for debugging.
  /// WARNING: This is an experimental API and subject to change.
  void SetPreserveAllTensors(bool value = true) {
    experimental_preserve_all_tensors_ = value;
  }

  /// Returns if the `experimental_preserve_all_tensors_` feature is enabled.
  /// WARNING: This is an experimental API and subject to change.
  bool GetPreserveAllTensors() { return experimental_preserve_all_tensors_; }

  /// Force all intermediate dynamic tensors to be released once they are not
  /// used by the model. Please use this configuration with caution, since it
  /// might reduce the peak memory usage of the model at the cost of a slower
  /// inference speed.
  /// WARNING: This is an experimental API and subject to change.
  void SetEnsureDynamicTensorsAreReleased(bool value = true) {
    experimental_ensure_dynamic_tensors_are_released_ = value;
  }

  /// Returns if the `experimental_ensure_dynamic_tensors_are_released_` feature
  /// is enabled.
  /// WARNING: This is an experimental API and subject to change.
  bool GetEnsureDynamicTensorsAreReleased() {
    return experimental_ensure_dynamic_tensors_are_released_;
  }

  /// Use dynamic tensor allocation and deallocation method for large tensors
  /// instead of static memory planner. Dynamic tensors are allocated just
  /// before when they're needed and released when they're not needed anymore.
  /// It improves peak memory usage but there could be some latency impact. The
  /// value (in bytes, and default is 1024 * 1024) is used to determine large
  /// tensors.
  /// WARNING: This is an experimental API and subject to change.
  void OptimizeMemoryForLargeTensors(int value = 1 << 20) {
    if (value > 0) {
      experimental_optimize_memory_for_large_tensors_ = value;
      experimental_ensure_dynamic_tensors_are_released_ = true;
    }
  }

  /// Returns the size (in bytes) threshold for dynamic tensor allocation
  /// method. It returns zero if the feature is not enabled.
  /// WARNING: This is an experimental API and subject to change.
  int GetDynamicAllocationForLargeTensors() {
    return experimental_optimize_memory_for_large_tensors_;
  }

 private:
  bool experimental_preserve_all_tensors_;
  bool experimental_ensure_dynamic_tensors_are_released_;
  int experimental_optimize_memory_for_large_tensors_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_INTERPRETER_OPTIONS_H_
