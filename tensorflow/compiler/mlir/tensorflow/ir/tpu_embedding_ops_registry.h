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

#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project

namespace mlir {
namespace TF {

// A global ops registry that is used to hold TPU embedding ops.
//
// Example:
//    TPUEmbeddingOpsRegistry::Global().Add<TF::FooOp>();
//    for (auto op_type_id : TPUEmbeddingOpsRegistry::Global().GetOpsTypeIds())
//    {
//      ...
//    }
class TPUEmbeddingOpsRegistry {
 public:
  // Add the op to the registry.
  //
  // Adding an op here will allow use old bridge legalization from the MLIR
  // bridge with the use of fallback mechanism. Therefore, addition of any op
  // here must have a python test with MLIR bridge enabled to verify that the
  // fallback works correctly.
  template <typename OpType>
  void Add() {
    ops_type_ids_.insert(TypeID::get<OpType>());
  }

  // Returns the type id of the ops in the TPUEmbeddingOpRegistry.
  const llvm::SmallDenseSet<mlir::TypeID>& GetOpsTypeIds();

  // Returns the global registry.
  static TPUEmbeddingOpsRegistry& Global();

 private:
  llvm::SmallDenseSet<mlir::TypeID> ops_type_ids_{};
};
}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TPU_EMBEDDING_OPS_REGISTRY_H_
