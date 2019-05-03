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
class Module;
}

namespace mlir {
namespace LLVM {
class LLVMDialect;
}

/// Conversion from the Standard dialect to the LLVM IR dialect.  Provides hooks
/// for derived classes to extend the conversion.
class LLVMLowering : public DialectConversion {
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

  /// Convert standard and builtin types to LLVM IR.
  Type convertType(Type t) override final;

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
};

} // namespace mlir

#endif // MLIR_LLVMIR_LLVMLOWERING_H
