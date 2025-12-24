/* Copyright 2024 The OpenXLA Authors.

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

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"

namespace mlir::triton::xla {
namespace {

// Extract additional attributes from an LLVM function that are not passed
// to the builder directly.
SmallVector<NamedAttribute> GetExtraAttrs(LLVM::LLVMFuncOp func) {
  llvm::StringSet<> registered_attr_names{
      func.getSymNameAttrName().getValue(),
      func.getFunctionTypeAttrName().getValue(),
      func.getLinkageAttrName().getValue(),
      func.getDsoLocalAttrName().getValue(),
      func.getCConvAttrName().getValue(),
      func.getArgAttrsAttrName().getValue(),
      func.getFunctionEntryCountAttrName().getValue()};
  return llvm::to_vector(
      llvm::make_filter_range(func->getAttrs(), [&](NamedAttribute attr) {
        return !registered_attr_names.contains(attr.getName().getValue());
      }));
}

// Strip address spaces from function parameters.
void StripParameterAddressSpaces(RewriterBase& rewriter,
                                 LLVM::LLVMFuncOp func) {
  // Figure out what the new signature should be.
  LLVM::LLVMFunctionType func_ty = func.getFunctionType();
  SmallVector<Type> generic_func_params(
      llvm::map_range(func_ty.getParams(), [](Type type) -> Type {
        auto ptr_ty = dyn_cast<LLVM::LLVMPointerType>(type);
        if (!ptr_ty) {
          return type;
        }
        if (ptr_ty.getAddressSpace() != NVVM::NVVMMemorySpace::Global) {
          return type;
        }
        return LLVM::LLVMPointerType::get(ptr_ty.getContext());
      }));
  LLVM::LLVMFunctionType generic_func_ty =
      func_ty.clone(generic_func_params, func_ty.getReturnTypes());

  // Create a function with the new signature.
  SmallVector<DictionaryAttr> arg_attrs(llvm::map_range(
      func.getArgAttrsAttr().getValue(),
      [](Attribute attr) { return cast<DictionaryAttr>(attr); }));
  auto generic_func = LLVM::LLVMFuncOp::create(
      rewriter, func.getLoc(), func.getSymName(), generic_func_ty,
      func.getLinkage(), func.getDsoLocal(), func.getCConv(),
      /*comdat=*/nullptr, GetExtraAttrs(func), arg_attrs,
      func.getFunctionEntryCount());

  // Convert generic address spaces back to original ones within the function
  // body.
  Block* entry = generic_func.addEntryBlock(rewriter);
  rewriter.setInsertionPointToEnd(entry);
  SmallVector<Value> converted_args;
  for (auto [arg, type] :
       llvm::zip(generic_func.getArguments(), func_ty.getParams())) {
    Value converted = arg;
    if (arg.getType() != type) {
      converted =
          LLVM::AddrSpaceCastOp::create(rewriter, arg.getLoc(), type, arg);
    }
    converted_args.push_back(converted);
  }

  // Move the rest of function body from the original function.
  rewriter.cloneRegionBefore(func.getBody(), generic_func.getBody(),
                             generic_func.getBody().end());
  rewriter.eraseOp(func);
  rewriter.mergeBlocks(entry->getNextNode(), entry, converted_args);
}

#define GEN_PASS_DEF_GENERALIZEKERNELSIGNATUREPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

// Rewrite signatures of kernel functions to use generic data pointers and
// cast them to global ones within the kernel.
struct GeneralizeKernelSignaturePass
    : public impl::GeneralizeKernelSignaturePassBase<
          GeneralizeKernelSignaturePass> {
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation()->walk([&](LLVM::LLVMFuncOp func) {
      if (!func->hasAttr(NVVM::NVVMDialect::getKernelFuncAttrName())) {
        return;
      }
      rewriter.setInsertionPointAfter(func);
      StripParameterAddressSpaces(rewriter, func);
    });
  }
};

}  // namespace

std::unique_ptr<Pass> CreateGeneralizeKernelSignaturePass() {
  return std::make_unique<GeneralizeKernelSignaturePass>();
}

}  // namespace mlir::triton::xla
