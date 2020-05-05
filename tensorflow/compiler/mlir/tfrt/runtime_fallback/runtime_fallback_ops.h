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

// This file defines the operations used in the Runtime Fallback dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_RUNTIME_FALLBACK_RUNTIME_FALLBACK_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_RUNTIME_FALLBACK_RUNTIME_FALLBACK_OPS_H_

#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffects.h"  // from @llvm-project

namespace mlir {
namespace tfd {

// Dialect for TFRT delegate operations.
class RuntimeFallbackDialect : public Dialect {
 public:
  explicit RuntimeFallbackDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "tfd"; }
};

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/runtime_fallback_ops.h.inc"

}  // namespace tfd
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_RUNTIME_FALLBACK_RUNTIME_FALLBACK_OPS_H_
