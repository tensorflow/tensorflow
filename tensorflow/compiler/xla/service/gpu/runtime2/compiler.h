/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_COMPILER_H_

#include <stddef.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "third_party/iree/compiler/bindings/c/iree/compiler/embedding_api.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla::gpu {

// Forward declare.
class RuntimeCompiler;

// Returns a new instance of the XLA:GPU runtime compiler loading it from a
// library. Every instance of the compiler creates a unique IREE compiler
// session.
std::unique_ptr<RuntimeCompiler> CreateRuntimeCompiler();

// Updates XLA:GPU input module with device kernels compiled by XLA.
StatusOr<std::string> BindXlaDeviceKernels(const DebugOptions& debug_options,
                                           mlir::ModuleOp,
                                           std::string_view asm_text,
                                           const std::vector<uint8_t>& binary);

// Wrapper around IREE compiler + bundled XLA:GPU compiler plugins to
// orchestrate compilation from XLA:GPU input dialects for IREE VM
// flatbuffer.
//
// TODO(ezhulenev): Instead of returning `bool` return helpful Status
// errors.
class RuntimeCompiler {
 public:
  // RAII wrapper around the compiler output (IREE VM bytecode).
  class Bytecode {
   public:
    Bytecode(iree_compiler_output_t* output, void* data, size_t length);
    ~Bytecode();

    void* data() { return data_; }
    size_t lenth() { return length_; }

   private:
    iree_compiler_output_t* output_;
    void* data_;
    size_t length_;
  };

  RuntimeCompiler(iree_compiler_session_t* session,
                  iree_compiler_invocation_t* inv);

  ~RuntimeCompiler();

  bool ParseSourceBuffer(std::string_view buffer);

  bool SetFlag(const char* flag);

  std::unique_ptr<Bytecode> CompileStandardPipeline();

 private:
  void SetError(iree_compiler_error_t* error) {
    LOG(ERROR) << "XLA:GPU runtime compiler error: "
               << ireeCompilerErrorGetMessage(error);
    if (error_) {
      ireeCompilerErrorDestroy(error_);
    }
    error_ = error;
  }

  iree_compiler_session_t* session_ = nullptr;
  iree_compiler_invocation_t* inv_ = nullptr;

  iree_compiler_error_t* error_ = nullptr;
  iree_compiler_output_t* output_ = nullptr;
};

}  // namespace xla::gpu

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_RUNTIME2_COMPILER_H_
