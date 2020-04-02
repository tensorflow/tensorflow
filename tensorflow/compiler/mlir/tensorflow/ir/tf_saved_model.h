/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_SAVED_MODEL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_SAVED_MODEL_H_

#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project

namespace mlir {
namespace tf_saved_model {

class TensorFlowSavedModelDialect : public Dialect {
 public:
  explicit TensorFlowSavedModelDialect(MLIRContext *context);
  LogicalResult verifyRegionArgAttribute(Operation *op, unsigned region_index,
                                         unsigned arg_index,
                                         NamedAttribute named_attr) override;
  LogicalResult verifyRegionResultAttribute(Operation *op,
                                            unsigned region_index,
                                            unsigned result_index,
                                            NamedAttribute named_attr) override;
  LogicalResult verifyOperationAttribute(Operation *op,
                                         NamedAttribute named_attr) override;
};

// Declares the operations for this dialect using the generated header.
#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h.inc"

// Returns the list of exported names for `op`.
// An empty list means `op` is not exported.
SmallVector<StringRef, 2> GetExportedNames(Operation *op);

// Returns true if `op` is exported.
bool IsExported(Operation *op);

// Returns true if `module` has tf_saved_model linkage semantics.
bool HasTfSavedModelSemantics(ModuleOp module);

// Returns the tf_saved_model.global_tensor op that func's arg_index'th argument
// refers to as a bound input, or null.
GlobalTensorOp LookupBoundInput(FuncOp func, int arg_index,
                                const SymbolTable &symbol_table);

// Gets the type that an exported function arg that is bound to `global_tensor`
// should have.
Type GetBoundInputArgTypeFor(GlobalTensorOp global_tensor);

}  // namespace tf_saved_model
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_SAVED_MODEL_H_
