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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TPU_EMBEDDING_OPS_REGISTRY_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TPU_EMBEDDING_OPS_REGISTRY_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/TypeID.h"  // from @llvm-project

namespace mlir {
namespace TF {

// A global ops registry that is used to hold TPU embedding ops.
// Example:
//    TPUEmbeddingOpsRegistry::Global().Add<TF::FooOp>();
//    for (auto op_name : TPUEmbeddingOpsRegistry::Global().GetOpsNames()) {
//      ...
//    }
class TPUEmbeddingOpsRegistry {
 public:
  // Add the op to the registry.
  template <typename OpType>
  void Add() {
    ops_names_.push_back(OpType::getOperationName());
  }

  // Get all the names of the ops in the registry.
  const llvm::SmallVector<llvm::StringLiteral>& GetOpsNames();

  // Returns the global registry.
  static TPUEmbeddingOpsRegistry& Global();

 private:
  llvm::SmallVector<llvm::StringLiteral> ops_names_{};
};
}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TPU_EMBEDDING_OPS_REGISTRY_H_
