//===- LLVMDialect.h - MLIR LLVM IR dialect ---------------------*- C++ -*-===//
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
// This file defines the LLVM IR dialect in MLIR, containing LLVM operations and
// LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVMDIALECT_H_
#define MLIR_TARGET_LLVMDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

namespace llvm {
class Type;
class LLVMContext;
} // end namespace llvm

namespace mlir {
namespace LLVM {

namespace detail {
struct LLVMTypeStorage;
}

class LLVMType : public mlir::Type::TypeBase<LLVMType, mlir::Type,
                                             detail::LLVMTypeStorage> {
public:
  enum Kind {
    LLVM_TYPE = FIRST_LLVM_TYPE,
  };

  using Base::Base;

  static bool kindof(unsigned kind) { return kind == LLVM_TYPE; }

  static LLVMType get(MLIRContext *context, llvm::Type *llvmType);

  llvm::Type *getUnderlyingType() const;
};

///// Ops /////
#define GET_OP_CLASSES
#include "mlir/LLVMIR/LLVMOps.h.inc"

class LLVMDialect : public Dialect {
public:
  explicit LLVMDialect(MLIRContext *context);
  llvm::LLVMContext &getLLVMContext() { return llvmContext; }
  llvm::Module &getLLVMModule() { return module; }

  /// Parse a type registered to this dialect.
  Type parseType(StringRef tyData, Location loc,
                 MLIRContext *context) const override;

  /// Print a type registered to this dialect.
  void printType(Type type, raw_ostream &os) const override;

  /// Verify a function argument attribute registered to this dialect.
  /// Returns true if the verification failed, false otherwise.
  bool verifyFunctionArgAttribute(const Function *func, unsigned argIdx,
                                  NamedAttribute argAttr) override;

private:
  llvm::LLVMContext llvmContext;
  llvm::Module module;
};

} // end namespace LLVM
} // end namespace mlir

#endif // MLIR_TARGET_LLVMDIALECT_H_
