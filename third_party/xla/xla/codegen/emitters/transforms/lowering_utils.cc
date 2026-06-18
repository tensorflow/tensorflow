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

#include "xla/codegen/emitters/transforms/lowering_utils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "xla/tsl/platform/logging.h"

namespace xla {
namespace emitters {

void EnsureAMDGPUAllocasUseAS5(mlir::Operation* operation) {
  operation->walk([](mlir::LLVM::AllocaOp alloca) {
    auto ptr_type =
        mlir::cast<mlir::LLVM::LLVMPointerType>(alloca.getResult().getType());
    // Check if address space is 0 (default/generic)
    if (ptr_type.getAddressSpace() == 0) {
      mlir::OpBuilder builder(alloca);
      // Create new alloca in address space 5
      auto new_ptr_type =
          mlir::LLVM::LLVMPointerType::get(builder.getContext(), 5);
      auto new_alloca = builder.create<mlir::LLVM::AllocaOp>(
          alloca.getLoc(), new_ptr_type, alloca.getElemType(),
          alloca.getArraySize(), alloca.getAlignment().value_or(0));
      alloca.replaceAllUsesWith(new_alloca.getResult());
      alloca.erase();
    }
  });
  VLOG(3) << "Ensured AMDGPU allocas use address space 5";
}

}  // namespace emitters
}  // namespace xla
