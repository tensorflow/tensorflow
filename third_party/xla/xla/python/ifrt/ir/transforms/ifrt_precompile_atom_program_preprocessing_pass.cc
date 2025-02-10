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
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"

namespace xla {
namespace ifrt {
namespace {

#define GEN_PASS_DEF_IFRTPRECOMPILEATOMPROGRAMPREPROCESSINGPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

class IfrtPrecompileAtomProgramPreprocessingPass
    : public impl::IfrtPrecompileAtomProgramPreprocessingPassBase<
          IfrtPrecompileAtomProgramPreprocessingPass> {
 public:
  using impl::IfrtPrecompileAtomProgramPreprocessingPassBase<
      IfrtPrecompileAtomProgramPreprocessingPass>::
      IfrtPrecompileAtomProgramPreprocessingPassBase;

  void runOnOperation() override;
};

// Determines the module type based on platform name.
mlir::FailureOr<llvm::StringRef> GetModuleType(
    CallOp call_op, const mlir::Pass::ListOption<std::string>& platform_names) {
  llvm::ArrayRef<int> device_ids = call_op.getDevices();
  // All devices should have the same type. Use the first platform name to
  // determine module type.
  auto first_logical_device_id = device_ids.front();
  if (first_logical_device_id >= platform_names.size()) {
    return call_op->emitOpError()
           << "cannot find mapping for logical device id "
           << first_logical_device_id
           << ". Mapping size: " << platform_names.size();
  }
  auto platform_name = platform_names[first_logical_device_id];

  // Get module type based on platform name.
  if (platform_name == xla::TpuName() || platform_name == xla::CudaName()) {
    return kIfrtModuleTypeXla;
  } else {
    return call_op->emitOpError()
           << "Unsupported platform for call op: " << platform_name;
  }
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
              GetModuleType(call_op, platform_names);
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
        mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
        auto callee_module =
            llvm::dyn_cast<mlir::ModuleOp>(callee->getParentOp());
        mlir::ModuleOp cloned_module = callee_module.clone();
        mlir::func::FuncOp cloned_callee = GetMainFunction(cloned_module);
        // Insert new cloned atom program module in the SymbolTable.
        symbol_table
            .getSymbolTable(
                callee_module->getParentWithTrait<mlir::OpTrait::SymbolTable>())
            .insert(cloned_module);
        mlir::SymbolRefAttr callee_attr = mlir::SymbolRefAttr::get(
            cloned_module.getSymNameAttr(),
            mlir::SymbolRefAttr::get(cloned_callee.getSymNameAttr()));
        module_and_type_refs_to_callee_symbol_ref[std::make_pair(
            call_op.getCallee(), module_type_attr)] = callee_attr;
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
    mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
    mlir::StringAttr module_type_attr =
        call_op->getAttrOfType<mlir::StringAttr>(kIfrtModuleTypeAttrName);
    auto callee_module = llvm::dyn_cast<mlir::ModuleOp>(callee->getParentOp());
    mlir::OpPassManager pm(mlir::ModuleOp::getOperationName());
    if (module_type_attr == kIfrtModuleTypeXla) {
      CreateIfrtCompileXlaPreprocessingPipeline(pm);
    } else if (module_type_attr != kIfrtModuleTypeMpmdReshard) {
      return call_op.emitOpError()
             << "module type " << module_type_attr << " is not supported";
    }
    if (mlir::failed(runPipeline(pm, callee_module))) {
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

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtPrecompileAtomProgramPreprocessingPass(
    IfrtPrecompileAtomProgramPreprocessingPassOptions options) {
  return std::make_unique<IfrtPrecompileAtomProgramPreprocessingPass>(options);
}

}  // namespace ifrt
}  // namespace xla
