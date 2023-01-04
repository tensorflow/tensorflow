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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_COMMON_OUTLINE_OPERATIONS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_COMMON_OUTLINE_OPERATIONS_H_

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_os_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace common {

// Returns true if the `op` is a constant-like op or produces none type.
bool IsConstantOrNone(Operation* op);

// Computes the list of Value(s) referenced by Subgraph Operations that are
// not defined within the Subgraph. Any such Value(s)
// are validly in-scope for the initial Operation. They must be either
// defined above the subgraph or appear as an argument to the containing func.
// These Value(s) are taken to be the arguments of the new raised func.
// An operand dependency is a Value referenced anywhere in an Op
// that is defined above the Op. All SSA Values are assigned/defined in a
// BlockArg or as a result of an Operation.
llvm::SmallVector<Value> AccumulateOperandsDefinedAbove(
    const llvm::SetVector<Operation*>& partition_ops);

// Similar to `AccumulateOperandsDefinedAbove()`, computes the Value(s) that are
// defined within a Subgraph and referenced in a descendant Operation. These
// Values(s) are to be returned by the new raised function.
llvm::SmallVector<Value> AccumulateResultsDefinedWithin(
    const llvm::SetVector<Operation*>& partition_ops);

// Represents a view of a set of mlir Operations that form a subgraph of the
// entire Module's DAG. `Subgraph` can be thought of as segment of sequential
// Operations within a func definition. Additional facts:
//    1. Subgraphs are restricted to a single Block. They do not span
//        branching instructions. Thus the subgraph is a simple 1-degree path.
//    2. All Operations in a subgraph belong to the same block in a
//        funtion body.
//    3. Function bodies are assumed to have only one block in some places.
class Subgraph {
  // Set vector preserves insertion order, must insert Ops in topological order.
 public:
  const llvm::SetVector<Operation*> partition_ops_;

  // Subgraphs are given a unique incremented integer id based on when
  // they were encountered in this pass.
  const int subgraph_id_;

  const llvm::StringRef dialect_namespace_;

  Subgraph(const llvm::SetVector<Operation*> partition_ops, int num_subgraphs)
      : partition_ops_(partition_ops),
        subgraph_id_(num_subgraphs),
        func_arguments_(AccumulateOperandsDefinedAbove(partition_ops)),
        func_outputs_(AccumulateResultsDefinedWithin(partition_ops)) {}

  const llvm::SmallVector<Value>& FuncArguments() const {
    // `Value`s in MLIR library are implemented as having "value semantics"
    // see "llvm/llvm-project/mlir/include/mlir/IR/Value.h" so copying is fine.
    return func_arguments_;
  }
  const llvm::SmallVector<Value>& FuncOutputs() const { return func_outputs_; }

 private:
  // Compute once at construction and save as field.
  const llvm::SmallVector<Value> func_arguments_;
  const llvm::SmallVector<Value> func_outputs_;
};

// Helper data structure for output parameters to `ExtractSubgraphToFunc`.
// `ExtractSubgraphToFunc` adds exactly two "new" `Operations`, a FuncOp and
// a CallOp. Pass these back to the caller for setting more specific attributes
// after graph mutation has taken place.
struct OpsAdded {
  mlir::func::FuncOp func_op;
  mlir::func::CallOp call_op;
};

// Given a `Subgraph` containing a sequence of adjacent `Operations` from
// the `module`, raise these `Operations` (and any ops contained nested within)
// to the body of a new seperate root level function. Replace in their current
// location with a `CallOp` which invokes said `FuncOp`. The inputs to
// this new functions are taken to be the `Values` that appear as operands
// to ops in the subgraph, which are not self-contained within the subgraph.
// The outputs of this function are taken to be the results of ops in the
// subgraph which are referenced as operands outside of the subgraph.
// Also refer to documention of `AccumulateOperandsDefinedAbove` &
// `AccumulateResultsDefinedWithin`.
void ExtractSubgraphToFunc(const Subgraph& subgraph, OpBuilder& builder,
                           ModuleOp& module, OpsAdded& ops_added);

}  // namespace common
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_COMMON_OUTLINE_OPERATIONS_H_
