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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_LLVM_IR_RUNTIME_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_LLVM_IR_RUNTIME_H_

#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace cpu {
namespace runtime {

extern const char* const kTanhV4F32SymbolName;
extern const char* const kTanhV8F32SymbolName;
extern const char* const kExpV4F32SymbolName;
extern const char* const kExpV8F32SymbolName;
extern const char* const kLogV4F32SymbolName;
extern const char* const kLogV8F32SymbolName;

// The following CPU runtime functions have LLVM-IR only implementations:
//
//  - __xla_cpu_runtime_TanhV4F32
//  - __xla_cpu_runtime_TanhV8F32
//
// |LinkIRRuntimeFunctions| rewrites calls to these functions into generic LLVM
// IR.

void RewriteIRRuntimeFunctions(llvm::Module* module, bool enable_fast_math);

}  // namespace runtime
}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_LLVM_IR_RUNTIME_H_
