//===- LLVMLowering.h - Lowering to the LLVM IR dialect ---------*- C++ -*-===//
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
//
// Provides a dialect conversion targeting the LLVM IR dialect.  By default, it
// converts Standard ops and types and provides hooks for dialect-specific
// extensions to the conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LLVMIR_LLVMLOWERING_H
#define MLIR_LLVMIR_LLVMLOWERING_H

#include "mlir/Transforms/DialectConversion.h"

namespace llvm {
class IntegerType;
class LLVMContext;
class Module;
class Type;
}

namespace mlir {
namespace LLVM {
class LLVMDialect;
}

/// Conversion from the Standard dialect to the LLVM IR dialect.  Provides hooks
/// for derived classes to extend the conversion.
class LLVMLowering : public DialectConversion {
public:
  /// Convert types to LLVM IR.  This calls `convertAdditionalType` to convert
  /// non-standard or non-builtin types.
  Type convertType(Type t) override final;

  /// Convert a non-empty list of types to be returned from a function into a
  /// supported LLVM IR type.  In particular, if more than one values is
  /// returned, create an LLVM IR structure type with elements that correspond
  /// to each of the MLIR types converted with `convertType`.
  Type packFunctionResults(ArrayRef<Type> types);

  /// Returns the LLVM context.
  llvm::LLVMContext &getLLVMContext();

protected:
  /// Create a set of converters that live in the pass object by passing them a
  /// reference to the LLVM IR dialect.  Store the module associated with the
  /// dialect for further type conversion.
  llvm::DenseSet<DialectOpConversion *>
  initConverters(MLIRContext *mlirContext) override final;

  /// Derived classes can override this function to initialize custom converters
  /// in addition to the existing converters from Standard operations.  It will
  /// be called after the `module` and `llvmDialect` have been made available.
  virtual llvm::DenseSet<DialectOpConversion *> initAdditionalConverters() {
    return {};
  };

  /// Derived classes can override this function to convert custom types.  It
  /// will be called by convertType if the default conversion from standard and
  /// builtin types fails.  Derived classes can thus call convertType whenever
  /// they need type conversion that supports both default and custom types.
  virtual Type convertAdditionalType(Type t) { return t; }

  /// Convert function signatures to LLVM IR.  In particular, convert functions
  /// with multiple results into functions returning LLVM IR's structure type.
  /// Use `convertType` to convert individual argument and result types.
  FunctionType convertFunctionSignatureType(
      FunctionType t, ArrayRef<NamedAttributeList> argAttrs,
      SmallVectorImpl<NamedAttributeList> &convertedArgAttrs) override final;

  /// Storage for the conversion patterns.
  llvm::BumpPtrAllocator converterStorage;
  /// LLVM IR module used to parse/create types.
  llvm::Module *module;
  LLVM::LLVMDialect *llvmDialect;

private:
  Type convertStandardType(Type type);

  // Convert a function type.  The arguments and results are converted one by
  // one.  Additionally, if the function returns more than one value, pack the
  // results into an LLVM IR structure type so that the converted function type
  // returns at most one result.
  Type convertFunctionType(FunctionType type);

  // Convert the index type.  Uses llvmModule data layout to create an integer
  // of the pointer bitwidth.
  Type convertIndexType(IndexType type);

  // Convert an integer type `i*` to `!llvm<"i*">`.
  Type convertIntegerType(IntegerType type);

  // Convert a floating point type: `f16` to `!llvm.half`, `f32` to
  // `!llvm.float` and `f64` to `!llvm.double`.  `bf16` is not supported
  // by LLVM.
  Type convertFloatType(FloatType type);

  // Convert a memref type into an LLVM type that captures the relevant data.
  // For statically-shaped memrefs, the resulting type is a pointer to the
  // (converted) memref element type. For dynamically-shaped memrefs, the
  // resulting type is an LLVM structure type that contains:
  //   1. a pointer to the (converted) memref element type
  //   2. as many index types as memref has dynamic dimensions.
  Type convertMemRefType(MemRefType type);

  // Convert a 1D vector type into an LLVM vector type.
  Type convertVectorType(VectorType type);

  // Get the LLVM representation of the index type based on the bitwidth of the
  // pointer as defined by the data layout of the module.
  llvm::IntegerType *getIndexType();

  // Wrap the given LLVM IR type into an LLVM IR dialect type.
  Type wrap(llvm::Type *llvmType);

  // Extract an LLVM IR type from the LLVM IR dialect type.
  llvm::Type *unwrap(Type type);
};

/// Base class for operation conversions targeting the LLVM IR dialect. Provides
/// conversion patterns with an access to the containing LLVMLowering for the
/// purpose of type conversions.
class LLVMOpLowering : public DialectOpConversion {
public:
  LLVMOpLowering(StringRef rootOpName, MLIRContext *context,
                 LLVMLowering &lowering);

protected:
  // Back-reference to the lowering class, used to call type and function
  // conversions accounting for potential extensions.
  LLVMLowering &lowering;
};

} // namespace mlir

#endif // MLIR_LLVMIR_LLVMLOWERING_H
