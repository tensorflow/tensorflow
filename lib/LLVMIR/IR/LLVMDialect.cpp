//===- LLVMDialect.cpp - LLVM IR Ops and Dialect registration -------------===//
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
// This file defines the types and operation details for the LLVM IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace mlir {
namespace LLVM {
namespace detail {
class LLVMTypeStorage : public ::mlir::detail::TypeStorage {
public:
  // LLVM types are pointer-unique.
  using KeyTy = llvm::Type *;

  KeyTy getKey() const { return underlyingType; }

  static LLVMTypeStorage *construct(TypeStorageAllocator &allocator,
                                    llvm::Type *t) {
    auto *memory = allocator.allocate<LLVMTypeStorage>();
    auto *storage = new (memory) LLVMTypeStorage;
    storage->underlyingType = t;
    return storage;
  }

  llvm::Type *underlyingType;
};
} // end namespace detail
} // end namespace LLVM
} // end namespace mlir

char LLVMType::typeID = '\0';

LLVMType LLVMType::get(MLIRContext *context, llvm::Type *llvmType) {
  return Base::get(context, FIRST_LLVM_TYPE, llvmType);
}

static Type parseLLVMType(StringRef data, Location loc, MLIRContext *ctx) {
  llvm::SMDiagnostic errorMessage;
  auto *llvmDialect =
      static_cast<LLVMDialect *>(ctx->getRegisteredDialect("llvm"));
  assert(llvmDialect && "LLVM dialect not registered");
  llvm::Type *type =
      llvm::parseType(data, errorMessage, llvmDialect->getLLVMModule());
  if (!type) {
    ctx->emitError(loc, errorMessage.getMessage());
    return {};
  }
  return LLVMType::get(ctx, type);
}

static void printLLVMType(Type ty, raw_ostream &os) {
  auto type = ty.dyn_cast<LLVMType>();
  assert(type && "printing wrong type");
  assert(type.getUnderlyingType() && "no underlying LLVM type");
  type.getUnderlyingType()->print(os);
}

llvm::Type *LLVMType::getUnderlyingType() const {
  return static_cast<ImplType *>(type)->underlyingType;
}

/*---- LLVM IR Dialect and its registration ----------------------------- */

LLVMDialect::LLVMDialect(MLIRContext *context)
    : Dialect("llvm", context), module("LLVMDialectModule", llvmContext) {
  addTypes<LLVMType>();
#define GET_OP_LIST
  addOperations<
#include "mlir/LLVMIR/llvm_ops.inc"
      >();

  typeParseHook = parseLLVMType;
  typePrintHook = printLLVMType;
}

static DialectRegistration<LLVMDialect> llvmDialect;
