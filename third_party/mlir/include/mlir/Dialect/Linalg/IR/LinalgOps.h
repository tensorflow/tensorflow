//===- LinalgOps.h - Linalg Operations --------------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef MLIR_DIALECT_LINALG_LINALGOPS_H_
#define MLIR_DIALECT_LINALG_LINALGOPS_H_

#include "mlir/Dialect/Linalg/IR/LinalgTraits.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace linalg {

/// Returns the name mangled library call name to disambiguate between different
/// overloads at the C level. The name mangling scheme is basic and uses MLIR
/// type names:
///   1. form a string which is the concatenation of the linalg op name with all
///      the operand type names, separate by underscores;
///   2. drop the `linalg.` prefix, and the `<`, `>`, `?` symbols from the type.
/// Assumes `op` is a LinalgOp.
///
/// Examples:
///
/// 1. linalg.fill(%A, %f) : memref<f32>, f32
///   name mangles into `linalg_fill_viewf32_f32_impl`
///
/// 2. linalg.dot(%A, %B, %C) :
///      memref<?xf32, stride_specification>,
///      memref<?xf32, stride_specification>, memref<f32>
///   name mangles into `linalg_dot_viewxf32_viewxf32_viewf32_impl`
///
/// 3. linalg.matmul(...) :
///      memref<?x?xf32, stride_specification>,
///      memref<?x?xf32, stride_specification>,
///      memref<?x?xf32, stride_specification>
///   name mangles into `linalg_matmul_viewxxf32_viewxxf32_viewxxf32_impl`
std::string generateLibraryCallName(Operation *op);

/// Returns the list of maps that map loops to operands of a Linalg op.
/// The i-th affine map identifies loop indices to subscripts that are used when
/// accessing the i-th operand.
/// For instance, a matmul that can be written in index notation as:
/// `A(i, k) * B(k, j) -> C(i, j)` will have the following, ordered, list of
/// affine maps:
///
/// ```mlir
///    (
///      (i, j, k) -> (i, k),
///      (i, j, k) -> (k, j),
///      (i, j, k) -> (i, j)
///    )
/// ```
///
/// Only permutation maps are currently supported.
SmallVector<AffineMap, 4> loopToOperandRangesMaps(Operation *op);

#include "mlir/Dialect/Linalg/IR/LinalgLibraryOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOps.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgLibraryOps.h.inc"

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_LINALGOPS_H_
