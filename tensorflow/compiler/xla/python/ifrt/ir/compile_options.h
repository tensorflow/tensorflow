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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_COMPILE_OPTIONS_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_COMPILE_OPTIONS_H_

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "tensorflow/compiler/xla/python/ifrt/compiler.h"
#include "tensorflow/compiler/xla/python/ifrt/executable.h"

namespace xla {
namespace ifrt {

// CompileOptions for an IFRT IR program.
struct IfrtIRCompileOptions
    : llvm::RTTIExtends<IfrtIRCompileOptions, CompileOptions> {
  IfrtIRCompileOptions() = default;
  IfrtIRCompileOptions(
      xla::CompileOptions xla_options,
      absl::flat_hash_map<std::string, LoadedExecutable*> loaded_exec_binding)
      : llvm::RTTIExtends<IfrtIRCompileOptions, CompileOptions>(
            std::move(xla_options)),
        loaded_exec_binding(std::move(loaded_exec_binding)) {}

  // Map from `getSymName()` of declared LoadedExecutableOp in the `mlir_module`
  // to pre-compiled LoadedExecutable instance. The LoadedExecutables must
  // outlive the LoadedExecutable to be compiled.
  absl::flat_hash_map<std::string, LoadedExecutable*> loaded_exec_binding;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_COMPILE_OPTIONS_H_
