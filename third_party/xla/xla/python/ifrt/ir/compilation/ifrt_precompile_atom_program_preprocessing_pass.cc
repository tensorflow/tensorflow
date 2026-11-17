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

#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/compilation/passes.h"
#include "xla/python/ifrt/ir/compilation/utils.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/utils.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTPRECOMPILEATOMPROGRAMPREPROCESSINGPASS
#include "xla/python/ifrt/ir/compilation/passes.h.inc"

namespace {

class IfrtPrecompileAtomProgramPreprocessingPass
    : public impl::IfrtPrecompileAtomProgramPreprocessingPassBase<
          IfrtPrecompileAtomProgramPreprocessingPass> {
 public:
  using impl::IfrtPrecompileAtomProgramPreprocessingPassBase<
      IfrtPrecompileAtomProgramPreprocessingPass>::
      IfrtPrecompileAtomProgramPreprocessingPassBase;

  mlir::LogicalResult initialize(mlir::MLIRContext* context) override;

  void runOnOperation() override;

 private:
  std::vector<std::string> platform_names_;
};

// Determines the module type based on platform name.
mlir::FailureOr<llvm::StringRef> GetModuleType(
    CallOp call_op, const std::vector<std::string>& platform_names) {
  // All devices should have the same type. Use the first platform name to
  // determine module type.
  mlir::FailureOr<llvm::StringRef> platform_name = GetPlatformNameForDeviceId(
      call_op, call_op.getDevices().front(), platform_names);
  if (mlir::failed(platform_name)) {
    return mlir::failure();
  }
  if (IsXlaPlatform(*platform_name)) {
    return llvm::StringRef(kIfrtModuleTypeXla);
  }
  return call_op->emitOpError()
         << "Unsupported platform for call op: " << *platform_name;
}

mlir::LogicalResult IfrtPrecompileAtomProgramPreprocessingPass::initialize(
    mlir::MLIRContext* context) {
  auto platform_names_or = ExpandPlatformNames(platform_names);
  if (!platform_names_or.ok()) {
    return mlir::emitError(mlir::UnknownLoc(),
                           platform_names_or.status().message());
  }
  platform_names_ = std::move(*platform_names_or);
  return mlir::success();
}

void IfrtPrecompileAtomProgramPreprocessingPass::runOnOperation() {
  mlir::SymbolTableCollection symbol_table;
  mlir::OpBuilder builder(&getContext());
  mlir::ModuleOp module_op = getOperation();
  mlir::func::FuncOp main_func = GetMainFunction(module_op);
  // Construct a map from callee `SymbolRefAttr` to the unique `CallOps`
  // using it. This map is used to decide if an atom program module must be
  // cloned (i.e., used on different types of devices).
  llvm::DenseMap<mlir::SymbolRefAttr, llvm::DenseSet<mlir::StringAttr>>
      callee_module_type_count;
  llvm::DenseMap<std::pair<mlir::SymbolRefAttr, mlir::StringAttr>,
                 mlir::SymbolRefAttr>
      module_and_type_refs_to_callee_symbol_ref;
  auto result = main_func.walk([&](CallOp call_op) -> mlir::WalkResult {
    mlir::StringAttr module_type_attr =
        call_op->getAttrOfType<mlir::StringAttr>(kIfrtModuleTypeAttrName);
    if (module_type_attr == nullptr) {
      // Set the module type if the CallOp does not have it set.
      if (mlir::FailureOr<llvm::StringRef> module_type =
              GetModuleType(call_op, platform_names_);
          mlir::succeeded(module_type)) {
        module_type_attr = builder.getStringAttr(*module_type);
        call_op->setAttr(kIfrtModuleTypeAttrName, module_type_attr);
      } else {
        return mlir::WalkResult::interrupt();
      }
    }

    bool new_module_type = callee_module_type_count[call_op.getCallee()]
                               .insert(module_type_attr)
                               .second;
    // Clone the callee module if it is of a new type, and there's more than
    // one type.
    if (new_module_type) {
      if (callee_module_type_count[call_op.getCallee()].size() > 1) {
        mlir::FailureOr<mlir::ModuleOp> callee_module =
            GetOutlinedAtomProgramModule(call_op, symbol_table);
        if (mlir::failed(callee_module)) {
          return mlir::WalkResult::interrupt();
        }
        ClonedAtomProgramModule cloned =
            CloneOutlinedAtomProgramModule(*callee_module, symbol_table);
        module_and_type_refs_to_callee_symbol_ref[std::make_pair(
            call_op.getCallee(), module_type_attr)] = cloned.callee_ref;
      } else {
        module_and_type_refs_to_callee_symbol_ref[std::make_pair(
            call_op.getCallee(), module_type_attr)] = call_op.getCalleeAttr();
      }
    }
    call_op.setCalleeAttr(
        module_and_type_refs_to_callee_symbol_ref[std::make_pair(
            call_op.getCallee(), module_type_attr)]);
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }

  // Stores symbol references to modules that have been preprocessed.
  mlir::DenseSet<mlir::SymbolRefAttr> preprocessed_modules;
  result = main_func.walk([&](CallOp call_op) -> mlir::WalkResult {
    // Preprocess a module only if it hasn't been already preprocessed.
    if (!preprocessed_modules.insert(call_op.getCallee()).second) {
      return mlir::WalkResult::advance();
    }
    mlir::FailureOr<mlir::ModuleOp> callee_module =
        GetOutlinedAtomProgramModule(call_op, symbol_table);
    if (mlir::failed(callee_module)) {
      return mlir::WalkResult::interrupt();
    }
    mlir::StringAttr module_type_attr =
        call_op->getAttrOfType<mlir::StringAttr>(kIfrtModuleTypeAttrName);
    mlir::OpPassManager pm(mlir::ModuleOp::getOperationName());
    if (module_type_attr == kIfrtModuleTypeXla) {
      createIfrtCompileXlaPreprocessingPipeline(pm, compile_options);
    } else if (module_type_attr != kIfrtModuleTypeMpmdReshard) {
      return call_op.emitOpError()
             << "module type " << module_type_attr << " is not supported";
    }
    if (mlir::failed(runPipeline(pm, *callee_module))) {
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
