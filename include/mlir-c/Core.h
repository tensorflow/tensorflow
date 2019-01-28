/*===-- mlir-c/Core.h - Core Library C Interface ------------------*- C -*-===*\
|*                                                                            *|
|* Copyright 2019 The MLIR Authors.                                           *|
|*                                                                            *|
|* Licensed under the Apache License, Version 2.0 (the "License");            *|
|* you may not use this file except in compliance with the License.           *|
|* You may obtain a copy of the License at                                    *|
|*                                                                            *|
|*   http://www.apache.org/licenses/LICENSE-2.0                               *|
|*                                                                            *|
|* Unless required by applicable law or agreed to in writing, software        *|
|* distributed under the License is distributed on an "AS IS" BASIS,          *|
|* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *|
|* See the License for the specific language governing permissions and        *|
|* limitations under the License.                                             *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to MLIR.                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/
#ifndef MLIR_C_CORE_H
#define MLIR_C_CORE_H

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

/// Opaque MLIR types.
/// Opaque C type for mlir::MLIRContext*.
typedef void *mlir_context_t;
/// Opaque C type for mlir::Type.
typedef const void *mlir_type_t;
/// Opaque C type for mlir::Function*.
typedef void *mlir_func_t;
/// Opaque C type for mlir::edsc::MLIREmiter.
typedef void *edsc_mlir_emitter_t;
/// Opaque C type for mlir::edsc::Expr.
typedef void *edsc_expr_t;
/// Opaque C type for mlir::edsc::Stmt.
typedef void *edsc_stmt_t;

/// Simple C lists for non-owning mlir Opaque C types.
/// Recommended usage is construction from the `data()` and `size()` of a scoped
/// owning SmallVectorImpl<...> and passing to one of the C functions declared
/// later in this file.
/// Once the function returns and the proper EDSC has been constructed,
/// resources are freed by exiting the scope.
typedef struct {
  int64_t *values;
  uint64_t n;
} int64_list_t;

typedef struct {
  mlir_type_t *types;
  uint64_t n;
} mlir_type_list_t;

typedef struct {
  edsc_expr_t *exprs;
  uint64_t n;
} edsc_expr_list_t;

typedef struct {
  edsc_stmt_t *stmts;
  uint64_t n;
} edsc_stmt_list_t;

typedef struct {
  edsc_expr_t base;
  edsc_expr_list_t indices;
} edsc_indexed_t;

typedef struct {
  edsc_indexed_t *list;
  uint64_t n;
} edsc_indexed_list_t;

/// Minimal C API for exposing EDSCs to Swift, Python and other languages.

/// Returns a simple scalar mlir::Type using the following convention:
///   - makeScalarType(c, "bf16") return an `mlir::Type::getBF16`
///   - makeScalarType(c, "f16") return an `mlir::Type::getF16`
///   - makeScalarType(c, "f32") return an `mlir::Type::getF32`
///   - makeScalarType(c, "f64") return an `mlir::Type::getF64`
///   - makeScalarType(c, "index") return an `mlir::Type::getIndex`
///   - makeScalarType(c, "i", bitwidth) return an
///     `mlir::Type::getInteger(bitwidth)`
///
/// No other combinations are currently supported.
mlir_type_t makeScalarType(mlir_context_t context, const char *name,
                           unsigned bitwidth);

/// Returns an `mlir::MemRefType` of the element type `elemType` and shape
/// `sizes`.
mlir_type_t makeMemRefType(mlir_context_t context, mlir_type_t elemType,
                           int64_list_t sizes);

/// Returns an `mlir::FunctionType` of the element type `elemType` and shape
/// `sizes`.
mlir_type_t makeFunctionType(mlir_context_t context, mlir_type_list_t inputs,
                             mlir_type_list_t outputs);

/// Returns the arity of `function`.
unsigned getFunctionArity(mlir_func_t function);

/// Returns a new opaque mlir::edsc::Expr that is bound into `emitter` with a
/// constant of the specified type.
edsc_expr_t bindConstantBF16(edsc_mlir_emitter_t emitter, double value);
edsc_expr_t bindConstantF16(edsc_mlir_emitter_t emitter, float value);
edsc_expr_t bindConstantF32(edsc_mlir_emitter_t emitter, float value);
edsc_expr_t bindConstantF64(edsc_mlir_emitter_t emitter, double value);
edsc_expr_t bindConstantInt(edsc_mlir_emitter_t emitter, int64_t value,
                            unsigned bitwidth);
edsc_expr_t bindConstantIndex(edsc_mlir_emitter_t emitter, int64_t value);

/// Returns the rank of the `function` argument at position `pos`.
/// If the argument is of MemRefType, this returns the rank of the MemRef.
/// Otherwise returns `0`.
/// TODO(ntv): support more than MemRefType and scalar Type.
unsigned getRankOfFunctionArgument(mlir_func_t function, unsigned pos);

/// Returns an opaque mlir::Type of the `function` argument at position `pos`.
mlir_type_t getTypeOfFunctionArgument(mlir_func_t function, unsigned pos);

/// Returns an opaque mlir::edsc::Expr that has been bound to the `pos` argument
/// of `function`.
edsc_expr_t bindFunctionArgument(edsc_mlir_emitter_t emitter,
                                 mlir_func_t function, unsigned pos);

/// Fills the preallocated list `result` with opaque mlir::edsc::Expr that have
/// been bound to each argument of `function`.
///
/// Prerequisites:
///   - `result` must have been preallocated with space for exactly the number
///     of arguments of `function`.
void bindFunctionArguments(edsc_mlir_emitter_t emitter, mlir_func_t function,
                           edsc_expr_list_t *result);

/// Returns the rank of `boundMemRef`. This API function is provided to more
/// easily compose with `bindFunctionArgument`. A similar function could be
/// provided for an mlir_type_t of type MemRefType but it is expected that users
/// of this API either:
///   1. construct the MemRefType explicitly, in which case they already have
///      access to the rank and shape of the MemRefType;
///   2. access MemRefs via mlir_function_t *values* in which case they would
///      pass edsc_expr_t bound to an edsc_emitter_t.
///
/// Prerequisites:
///   - `boundMemRef` must be an opaque edsc_expr_t that has alreay been bound
///     in `emitter`.
unsigned getBoundMemRefRank(edsc_mlir_emitter_t emitter,
                            edsc_expr_t boundMemRef);

/// Fills the preallocated list `result` with opaque mlir::edsc::Expr that have
/// been bound to each dimension of `boundMemRef`.
///
/// Prerequisites:
///   - `result` must have been preallocated with space for exactly the rank of
///     `boundMemRef`;
///   - `boundMemRef` must be an opaque edsc_expr_t that has alreay been bound
///     in `emitter`. This is because symbolic MemRef shapes require an SSAValue
///     that can only be recovered from `emitter`.
void bindMemRefShape(edsc_mlir_emitter_t emitter, edsc_expr_t boundMemRef,
                     edsc_expr_list_t *result);

/// Fills the preallocated lists `resultLbs`, `resultUbs` and `resultSteps` with
/// opaque mlir::edsc::Expr that have been bound to proper values to traverse
/// each dimension of `memRefType`.
/// At the moment:
///   - `resultsLbs` are always bound to the constant index `0`;
///   - `resultsUbs` are always bound to the shape of `memRefType`;
///   - `resultsSteps` are always bound to the constant index `1`.
/// In the future, this will allow non-contiguous MemRef views.
///
/// Prerequisites:
///   - `resultLbs`, `resultUbs` and `resultSteps` must have each been
///     preallocated with space for exactly the rank of `boundMemRef`;
///   - `boundMemRef` must be an opaque edsc_expr_t that has alreay been bound
///     in `emitter`. This is because symbolic MemRef shapes require an SSAValue
///     that can only be recovered from `emitter`.
void bindMemRefView(edsc_mlir_emitter_t emitter, edsc_expr_t boundMemRef,
                    edsc_expr_list_t *resultLbs, edsc_expr_list_t *resultUbs,
                    edsc_expr_list_t *resultSteps);

/// Returns an opaque expression for an mlir::edsc::Expr.
edsc_expr_t makeBindable();

/// Returns an opaque expression for an mlir::edsc::Stmt.
edsc_stmt_t makeStmt(edsc_expr_t e);

/// Returns an opaque expression for an mlir::edsc::Indexed.
edsc_indexed_t makeIndexed(edsc_expr_t expr);

/// Returns an indexed opaque expression with indices bound in the structure
/// given an `indexed` and `indices`.
/// Prerequisite:
///   - `indexed` must not have been indexed previously.
edsc_indexed_t index(edsc_indexed_t indexed, edsc_expr_list_t indices);

/// Returns an opaque expression that will emit an mlir::LoadOp.
edsc_expr_t Load(edsc_indexed_t indexed, edsc_expr_list_t indices);

/// Returns an opaque statement for an mlir::StoreOp.
edsc_stmt_t Store(edsc_expr_t value, edsc_indexed_t indexed,
                  edsc_expr_list_t indices);

/// Returns an opaque statement for an mlir::SelectOp.
edsc_expr_t Select(edsc_expr_t cond, edsc_expr_t lhs, edsc_expr_t rhs);

/// Returns an opaque statement for an mlir::ReturnOp.
edsc_stmt_t Return(edsc_expr_list_t values);

/// Returns a single opaque statement that acts as an mlir block. At the moment
/// this is pure syntactic sugar to allow lists of mlir::edsc::Stmt to be
/// specified and emitted. In particular, block arguments are not currently
/// supported.
edsc_stmt_t Block(edsc_stmt_list_t enclosedStmts);

/// Returns an opaque statement for an mlir::ForInst with `enclosedStmts` nested
/// below it.
edsc_stmt_t For(edsc_expr_t iv, edsc_expr_t lb, edsc_expr_t ub,
                edsc_expr_t step, edsc_stmt_list_t enclosedStmts);

/// Returns an opaque statement for a perfectly nested set of mlir::ForInst with
/// `enclosedStmts` nested below it.
edsc_stmt_t ForNest(edsc_expr_list_t iv, edsc_expr_list_t lb,
                    edsc_expr_list_t ub, edsc_expr_list_t step,
                    edsc_stmt_list_t enclosedStmts);

/// Returns an opaque expression for the corresponding Binary operation.
edsc_expr_t Add(edsc_expr_t e1, edsc_expr_t e2);
edsc_expr_t Sub(edsc_expr_t e1, edsc_expr_t e2);
edsc_expr_t Mul(edsc_expr_t e1, edsc_expr_t e2);
// edsc_expr_t Div(edsc_expr_t e1, edsc_expr_t e2);
edsc_expr_t LT(edsc_expr_t e1, edsc_expr_t e2);
edsc_expr_t LE(edsc_expr_t e1, edsc_expr_t e2);
edsc_expr_t GT(edsc_expr_t e1, edsc_expr_t e2);
edsc_expr_t GE(edsc_expr_t e1, edsc_expr_t e2);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // MLIR_C_CORE_H
