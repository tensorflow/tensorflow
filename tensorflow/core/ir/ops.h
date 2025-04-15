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

#ifndef TENSORFLOW_CORE_IR_OPS_H_
#define TENSORFLOW_CORE_IR_OPS_H_

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/RegionKindInterface.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/interfaces.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"

// Get the C++ declaration for all the ops defined in ODS for the dialect.

#define GET_OP_CLASSES
#include "tensorflow/core/ir/ops.h.inc"

namespace mlir {
namespace tfg {

// Analysis that keeps track of all function names in a module.
struct FunctionTable {
  explicit FunctionTable(ModuleOp module);

  // Returns whether there are no functions.
  bool empty() const { return functions.empty(); }

  // Returns whether `op` may be a function call.
  bool MayBeCall(Operation* op) const;

  // Returns whether `op` is a legacy function call. A "legacy" function call
  // is when the operation name is the name of a function in the library.
  bool IsLegacyCall(Operation* op) const;

 private:
  // All the functions in the graph.
  DenseSet<StringRef> functions;
};

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_OPS_H_
