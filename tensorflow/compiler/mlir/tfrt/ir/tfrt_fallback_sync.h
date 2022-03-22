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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_SYNC_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_SYNC_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "tfrt/core_runtime/opdefs/traits.h"  // from @tf_runtime
#include "tfrt/tensor/opdefs/tensor.h"  // from @tf_runtime

using namespace mlir;  // NOLINT

namespace tfrt {
namespace fallback_sync {

// Dialect for fallback operations.
class FallbackSyncDialect : public Dialect {
 public:
  explicit FallbackSyncDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "tfrt_fallback_sync"; }
};

}  // namespace fallback_sync
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_SYNC_H_
