#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Target/LLVMIR/Passes.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

//===----------------------------------------------------------------------===//
// This file implements a pass to add debug info scope to LLVM operations, and
// is inspired by the DIScopeForLLVMFuncOpPass in LLVM/MLIR. Different from the
// DIScopeForLLVMFuncOpPass, this pass also handles inlined functions.
//===----------------------------------------------------------------------===//

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Target/LLVMIR/Passes.h.inc"

namespace {

/// Attempt to extract a filename for the given loc.
FileLineColLoc extractFileLoc(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc;
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return extractFileLoc(nameLoc.getChildLoc());
  if (auto opaqueLoc = dyn_cast<OpaqueLoc>(loc))
    return extractFileLoc(opaqueLoc.getFallbackLocation());
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc))
    return extractFileLoc(fusedLoc.getLocations().front());
  if (auto callerLoc = dyn_cast<CallSiteLoc>(loc))
    return extractFileLoc(callerLoc.getCaller());
  StringAttr unknownFile = mlir::StringAttr::get(loc.getContext(), "<unknown>");
  return mlir::FileLineColLoc::get(unknownFile, 0, 0);
}

/// Add a debug info scope to LLVMFuncOp that are missing it.
struct LLVMDIScopePass : public LLVMDIScopeBase<LLVMDIScopePass> {
  LLVMDIScopePass() = default;

  void setSubprogramAttr(LLVM::LLVMFuncOp funcOp) {
    Location loc = funcOp.getLoc();
    if (loc->findInstanceOf<mlir::FusedLocWith<LLVM::DISubprogramAttr>>())
      return;

    MLIRContext *context = &getContext();

    // To find a DICompileUnitAttr attached to a parent (the module for
    // example), otherwise create a default one.
    LLVM::DICompileUnitAttr compileUnitAttr;
    if (ModuleOp module = funcOp->getParentOfType<ModuleOp>()) {
      auto fusedCompileUnitAttr =
          module->getLoc()
              ->findInstanceOf<mlir::FusedLocWith<LLVM::DICompileUnitAttr>>();
      if (fusedCompileUnitAttr)
        compileUnitAttr = fusedCompileUnitAttr.getMetadata();
    }

    // Filename, line and colmun to associate to the function.
    LLVM::DIFileAttr fileAttr;
    int64_t line = 1, col = 1;
    FileLineColLoc fileLoc = extractFileLoc(loc);
    if (!fileLoc && compileUnitAttr) {
      fileAttr = compileUnitAttr.getFile();
    } else if (!fileLoc) {
      fileAttr = LLVM::DIFileAttr::get(context, "<unknown>", "");
    } else {
      line = fileLoc.getLine();
      col = fileLoc.getColumn();
      StringRef inputFilePath = fileLoc.getFilename().getValue();
      fileAttr = LLVM::DIFileAttr::get(
          context, llvm::sys::path::filename(inputFilePath),
          llvm::sys::path::parent_path(inputFilePath));
    }
    auto subroutineTypeAttr =
        LLVM::DISubroutineTypeAttr::get(context, llvm::dwarf::DW_CC_normal, {});

    // Figure out debug information (`subprogramFlags` and `compileUnitAttr`) to
    // attach to the function definition / declaration. External functions are
    // declarations only, and are defined in a different compile unit, so mark
    // them appropriately in `subprogramFlags`, and set an empty
    // `compileUnitAttr`.
    DistinctAttr distinctId;
    auto subprogramFlags = LLVM::DISubprogramFlags::Optimized;
    if (!funcOp.isExternal()) {
      distinctId = mlir::DistinctAttr::create(mlir::UnitAttr::get(context));
      if (!compileUnitAttr) {
        compileUnitAttr = LLVM::DICompileUnitAttr::get(
            distinctId, llvm::dwarf::DW_LANG_C, fileAttr,
            StringAttr::get(context, "triton"),
            /*isOptimized=*/true, LLVM::DIEmissionKind::LineTablesOnly);
      }
      subprogramFlags = subprogramFlags | LLVM::DISubprogramFlags::Definition;
    } else {
      compileUnitAttr = {};
    }

    StringAttr funcNameAttr = funcOp.getNameAttr();
    // Note that scopeline is set differently from LLVM's
    // DIScopeForLLVMFuncOpPass. I don't find reasons why scopeline should be
    // the column offset
    auto subprogramAttr = LLVM::DISubprogramAttr::get(
        context, distinctId, compileUnitAttr, fileAttr, funcNameAttr,
        funcNameAttr, fileAttr, /*line=*/line, /*scopeline=*/line,
        subprogramFlags, subroutineTypeAttr, /*retainNodes=*/{},
        /*annotations=*/{});
    funcOp->setLoc(FusedLoc::get(context, {loc}, subprogramAttr));
  }

  // Get a nested loc for inlined functions
  Location getNestedLoc(Operation *op, LLVM::DIScopeAttr scopeAttr,
                        Location calleeLoc) {
    auto calleeFileName = extractFileLoc(calleeLoc).getFilename();
    auto context = op->getContext();
    LLVM::DIFileAttr calleeFileAttr = LLVM::DIFileAttr::get(
        context, llvm::sys::path::filename(calleeFileName),
        llvm::sys::path::parent_path(calleeFileName));
    auto lexicalBlockFileAttr = LLVM::DILexicalBlockFileAttr::get(
        context, scopeAttr, calleeFileAttr, /*discriminator=*/0);
    Location loc = calleeLoc;
    if (mlir::isa<CallSiteLoc>(calleeLoc)) {
      auto nestedLoc = mlir::cast<CallSiteLoc>(calleeLoc).getCallee();
      loc = getNestedLoc(op, lexicalBlockFileAttr, nestedLoc);
    }
    return FusedLoc::get(context, {loc}, lexicalBlockFileAttr);
  }

  void setLexicalBlockFileAttr(Operation *op) {
    auto opLoc = op->getLoc();
    if (auto callSiteLoc = dyn_cast<CallSiteLoc>(opLoc)) {
      auto callerLoc = callSiteLoc.getCaller();
      auto calleeLoc = callSiteLoc.getCallee();
      LLVM::DIScopeAttr scopeAttr;
      // We assemble the full inline stack so the parent of this loc must be a
      // function
      auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
      auto funcOpLoc = mlir::cast<FusedLoc>(funcOp.getLoc());
      scopeAttr = mlir::cast<LLVM::DISubprogramAttr>(funcOpLoc.getMetadata());
      auto loc =
          CallSiteLoc::get(getNestedLoc(op, scopeAttr, calleeLoc), callerLoc);
      op->setLoc(loc);
    }
  }

  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) -> void {
      if (isa<LLVM::LLVMFuncOp>(op))
        setSubprogramAttr(cast<LLVM::LLVMFuncOp>(op));
      else
        setLexicalBlockFileAttr(op);
    });
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> mlir::createLLVMDIScopePass() {
  return std::make_unique<LLVMDIScopePass>();
}
