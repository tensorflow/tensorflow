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
#ifndef TENSORFLOW_CORE_TPU_TPU_COMPILE_INTERFACE_H_
#define TENSORFLOW_CORE_TPU_TPU_COMPILE_INTERFACE_H_

#include "absl/strings/string_view.h"

// Some legacy code requires different implementations for operations like
// fingerprint/hashing during compilation and/or graph rewriting. These
// alternate implementations can be registered (via a module initializer) to
// change the default behavior.
class TpuCompileInterface {
 public:
  virtual ~TpuCompileInterface() {}
  static TpuCompileInterface* Get();
  static bool RegisterImplementation(TpuCompileInterface* impl);

  virtual uint64_t FingerprintString(absl::string_view str) = 0;

  // Proto: tensorflow::tpu::CompilationResultProto
  // Location: tensorflow/core/protobuf/tpu/compilation_result.proto
  static inline constexpr char kTpuCompileErrorPayloadKey[] =
      "type.googleapis.com/tensorflow.tpu.CompilationResultProto";

  // Unique string added to the error message for permanent errors during
  // XLA:TPU compilation. This can be used by TensorFlow models to distinguish
  // compilation errors from transient errors created by TPU worker preemptions
  // and restarts.
  static inline constexpr char kTpuCompileErrorMessage[] =
      "XLA:TPU compile permanent error";
};

#endif  // TENSORFLOW_CORE_TPU_TPU_COMPILE_INTERFACE_H_
