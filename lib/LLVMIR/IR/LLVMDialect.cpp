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
#include "llvm/IR/Function.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace mlir {
namespace LLVM {
namespace detail {
struct LLVMTypeStorage : public ::mlir::detail::TypeStorage {
  LLVMTypeStorage(llvm::Type *ty) : underlyingType(ty) {}

  // LLVM types are pointer-unique.
  using KeyTy = llvm::Type *;
  bool operator==(const KeyTy &key) const { return key == underlyingType; }

  static LLVMTypeStorage *construct(TypeStorageAllocator &allocator,
                                    llvm::Type *ty) {
    return new (allocator.allocate<LLVMTypeStorage>()) LLVMTypeStorage(ty);
  }

  llvm::Type *underlyingType;
};
} // end namespace detail
} // end namespace LLVM
} // end namespace mlir

LLVMType LLVMType::get(MLIRContext *context, llvm::Type *llvmType) {
  return Base::get(context, FIRST_LLVM_TYPE, llvmType);
}

llvm::Type *LLVMType::getUnderlyingType() const {
  return static_cast<ImplType *>(type)->underlyingType;
}

/*---- LLVM IR Dialect and its registration ----------------------------- */

LLVMDialect::LLVMDialect(MLIRContext *context)
    : Dialect("llvm", context), module("LLVMDialectModule", llvmContext) {
  addTypes<LLVMType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/LLVMIR/LLVMOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/LLVMIR/LLVMOps.cpp.inc"

/// Parse a type registered to this dialect.
Type LLVMDialect::parseType(StringRef tyData, Location loc,
                            MLIRContext *context) const {
  llvm::SMDiagnostic errorMessage;
  llvm::Type *type = llvm::parseType(tyData, errorMessage, module);
  if (!type)
    return (context->emitError(loc, errorMessage.getMessage()), nullptr);
  return LLVMType::get(context, type);
}

/// Print a type registered to this dialect.
void LLVMDialect::printType(Type type, raw_ostream &os) const {
  auto llvmType = type.dyn_cast<LLVMType>();
  assert(llvmType && "printing wrong type");
  assert(llvmType.getUnderlyingType() && "no underlying LLVM type");
  llvmType.getUnderlyingType()->print(os);
}

/// Verify LLVMIR function argument attributes.
bool LLVMDialect::verifyFunctionArgAttribute(Function *func, unsigned argIdx,
                                             NamedAttribute argAttr) {
  // Check that llvm.noalias is a boolean attribute.
  if (argAttr.first == "llvm.noalias" && !argAttr.second.isa<BoolAttr>())
    return func->emitError(
        "llvm.noalias argument attribute of non boolean type");
  return false;
}

static DialectRegistration<LLVMDialect> llvmDialect;
