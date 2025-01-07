/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_CODEGEN_IR_XLA_OPS_H_
#define XLA_CODEGEN_IR_XLA_OPS_H_

#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/Attributes.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"  // IWYU pragma: keep
#include "mlir/IR/Dialect.h"  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"  // IWYU pragma: keep
#include "mlir/Interfaces/CallInterfaces.h"  // IWYU pragma: keep
#include "mlir/Interfaces/InferTypeOpInterface.h"  // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h"  // IWYU pragma: keep
#include "xla/codegen/ir/xla_dialect.h.inc"
#include "xla/hlo/analysis/indexing_map.h"  // IWYU pragma: keep
#define GET_ATTRDEF_CLASSES
#include "xla/codegen/ir/xla_attrs.h.inc"
#define GET_OP_CLASSES
#include "xla/codegen/ir/xla_ops.h.inc"

namespace xla {

struct VariableConstraints {
  llvm::SmallVector<llvm::SmallDenseMap<mlir::AffineExpr, Interval>>
      constraints_for_dims;
  llvm::SmallVector<llvm::SmallDenseMap<mlir::AffineExpr, Interval>>
      constraints_for_symbols;
};
VariableConstraints GetConstraintsForVariables(const IndexingMap& map);

// Parses a comma-separated list of operands, ex: %d1, %d2.
mlir::ParseResult parseOperands(
    mlir::OpAsmParser& parser,
    mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4>* operands);

// Parses a chain of string attributes into an indexing map.
// Example:
// "()[s0, s1] -> (1 + s0 + s1 mod 3 - s1, s0 mod 2),"
//   " domain: s0 in [-10, 10], s1 in [0, 2]"
// will be parsed as 3 StringAttrs, concatenated into a single string, and then
// parsed into an IndexingMap.
std::optional<IndexingMap> parseChainOfStringsAsIndexingMap(
    mlir::AsmParser& parser);

}  // namespace xla

#endif  // XLA_CODEGEN_IR_XLA_OPS_H_
