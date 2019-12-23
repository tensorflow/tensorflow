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
/// Opaque C type for mlir::FuncOp.
typedef void *mlir_func_t;
/// Opaque C type for mlir::Attribute.
typedef const void *mlir_attr_t;

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
  const char *name;
  mlir_attr_t value;
} mlir_named_attr_t;

typedef struct {
  mlir_named_attr_t *list;
  uint64_t n;
} mlir_named_attr_list_t;

/// Minimal C API for exposing EDSCs to Swift, Python and other languages.

/// Returns an `mlir::MemRefType` of the element type `elemType` and shape
/// `sizes`.
mlir_type_t makeMemRefType(mlir_context_t context, mlir_type_t elemType,
                           int64_list_t sizes);

/// Returns an `mlir::FunctionType` of the element type `elemType` and shape
/// `sizes`.
mlir_type_t makeFunctionType(mlir_context_t context, mlir_type_list_t inputs,
                             mlir_type_list_t outputs);

/// Returns an `mlir::IndexType`.
mlir_type_t makeIndexType(mlir_context_t context);

/// Returns an `mlir::IntegerAttr` of the specified type that contains the given
/// value.
mlir_attr_t makeIntegerAttr(mlir_type_t type, int64_t value);

/// Returns an `mlir::BoolAttr` with the given value.
mlir_attr_t makeBoolAttr(mlir_context_t context, bool value);

/// Returns an `mlir::FloatAttr` with the given value.
mlir_attr_t makeFloatAttr(mlir_context_t context, float value);

/// Returns an `mlir::StringAttr` with the given value.
mlir_attr_t makeStringAttr(mlir_context_t context, const char *value);

/// Parses an MLIR type from the string `type` in the given context. Returns a
/// NULL type on error. If non-NULL, `charsRead` will contain the number of
/// characters that were processed by the parser.
mlir_type_t mlirParseType(const char *type, mlir_context_t context,
                          uint64_t *charsRead);

/// Returns the arity of `function`.
unsigned getFunctionArity(mlir_func_t function);

/// Returns the rank of the `function` argument at position `pos`.
/// If the argument is of MemRefType, this returns the rank of the MemRef.
/// Otherwise returns `0`.
/// TODO(ntv): support more than MemRefType and scalar Type.
unsigned getRankOfFunctionArgument(mlir_func_t function, unsigned pos);

/// Returns an opaque mlir::Type of the `function` argument at position `pos`.
mlir_type_t getTypeOfFunctionArgument(mlir_func_t function, unsigned pos);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // MLIR_C_CORE_H
