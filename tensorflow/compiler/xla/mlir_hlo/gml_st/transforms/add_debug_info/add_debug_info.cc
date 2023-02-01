/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <optional>
#include <string>
#include <utility>

#include "gml_st/transforms/passes.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Path.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_ADDDEBUGINFOPASS
#include "gml_st/transforms/passes.h.inc"

struct AddDebugInfoPass : public impl::AddDebugInfoPassBase<AddDebugInfoPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();
    OpBuilder builder(context);
    std::string inputFilePath("-");

    if (auto fileLoc = module.getLoc().dyn_cast<mlir::FileLineColLoc>())
      inputFilePath = fileLoc.getFilename().getValue();

    auto fileAttr =
        LLVM::DIFileAttr::get(context, llvm::sys::path::filename(inputFilePath),
                              llvm::sys::path::parent_path(inputFilePath));

    auto producer = StringAttr::get(context, "XLA CPU");
    auto cuAttr = LLVM::DICompileUnitAttr::get(
        context, llvm::dwarf::DW_LANG_C_plus_plus_17, fileAttr, producer,
        /*isOptimized=*/false, LLVM::DIEmissionKind::LineTablesOnly);
    module.walk([&](func::FuncOp funcOp) {
      StringAttr funcName = StringAttr::get(context, funcOp.getName());
      auto bT = LLVM::DIBasicTypeAttr::get(
          context, llvm::dwarf::DW_TAG_base_type, "void", /*sizeInBits=*/0,
          /*encoding=*/1);
      auto subTypeAttr = LLVM::DISubroutineTypeAttr::get(
          context, llvm::dwarf::DW_CC_normal, {bT});
      auto spAttr = LLVM::DISubprogramAttr::get(
          context, cuAttr, fileAttr, funcName, funcName, fileAttr, /*line=*/1,
          /*scopeline=*/1, LLVM::DISubprogramFlags::Definition, subTypeAttr);
      funcOp->setLoc(builder.getFusedLoc({funcOp->getLoc()}, spAttr));
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAddDebugInfoPass() {
  return std::make_unique<AddDebugInfoPass>();
}

}  // namespace gml_st
}  // namespace mlir
