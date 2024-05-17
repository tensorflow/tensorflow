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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_EXECUTE_OP_REGISTRY_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_EXECUTE_OP_REGISTRY_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace tensorflow {
namespace mlrt_compiler {

class ExecuteOpRegistry {
 public:
  mlir::LogicalResult RegisterExecuteOp(mlir::Operation* op, uint32_t op_key) {
    if (op_key >= execute_ops_.size()) {
      execute_ops_.resize(op_key + 1);
    }
    if (auto* register_op = execute_ops_[op_key]) {
      if (register_op->getName() != op->getName() ||
          register_op->getAttrs() != op->getAttrs()) {
        return op->emitError() << "Key " << op_key << " already registered.";
      }
      return mlir::success();
    }
    execute_ops_[op_key] = op;
    return mlir::success();
  }

  void ReplaceExecuteOp(int64_t key, mlir::Operation* op) {
    execute_ops_[key] = op;
  }

  llvm::ArrayRef<mlir::Operation*> GetExecuteOps() const {
    return execute_ops_;
  }

 private:
  // Using a vector to keep fallback ops in order, and the key for a fallback op
  // is its corresponding index here.
  llvm::SmallVector<mlir::Operation*, 8> execute_ops_;
};

}  // namespace mlrt_compiler
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_MLRT_EXECUTE_OP_REGISTRY_H_
