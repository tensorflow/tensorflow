/* Copyright 2026 The OpenXLA Authors.

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
#include "xla/backends/gpu/codegen/nested_kernel_utils.h"

#include <cstddef>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/codegen/emitters/transforms/lower_to_llvm_gpu.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"

namespace xla::gpu {
namespace {

constexpr unsigned kSharedMemoryAddrSpace = 3;

}  // namespace

std::string GetNestedKernelFunctionName(absl::string_view entry_function_name,
                                        size_t index) {
  return absl::StrCat(entry_function_name, "_nested_", index);
}

absl::Status LowerLauncherModuleToLlvmDialect(
    mlir::ModuleOp launcher_module,
    const stream_executor::DeviceDescription& device_info, bool should_verify) {
  mlir::PassManager pm(launcher_module.getContext());
  pm.enableVerifier(should_verify);
  pm.addPass(emitters::createLowerTensorsPass(device_info));
  pm.addPass(emitters::createMergePointersToSameSlicePass());
  pm.addPass(createConvertIndexTypePass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(emitters::createLowerToLLVMGPUPass(device_info));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  tsl::StatusScopedDiagnosticHandler diagnostic_handler(
      launcher_module.getContext());
  return diagnostic_handler.consumeStatus(pm.run(launcher_module));
}

absl::Status SpliceNestedKernelModuleIntoLauncher(
    mlir::ModuleOp launcher_module, mlir::ModuleOp nested_module,
    absl::string_view nested_fn_name) {
  mlir::MLIRContext* ctx = launcher_module.getContext();
  mlir::OpBuilder builder(ctx);

  auto nested_entry_func =
      nested_module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(nested_fn_name);
  TF_RET_CHECK(nested_entry_func != nullptr)
      << "Nested kernel entry function " << nested_fn_name
      << " not found in lowered nested module.";

  nested_entry_func.setLinkage(mlir::LLVM::Linkage::Internal);
  nested_entry_func.setPassthroughAttr(
      builder.getArrayAttr({builder.getStringAttr("alwaysinline")}));
  // `@launcher` already carries buffer-assignment `noalias`/`align` attributes
  // on its parameters. Stripping `noalias` from the `alwaysinline` internal
  // function avoids conflicting `noalias` attributes when two nested parameters
  // bind to the same outer buffer.
  for (unsigned arg_idx = 0; arg_idx < nested_entry_func.getNumArguments();
       ++arg_idx) {
    nested_entry_func.removeArgAttr(
        arg_idx, mlir::LLVM::LLVMDialect::getNoAliasAttrName());
  }

  // Prefix internal helper functions and non-shmem globals in `nested_module`
  // so symbols from different nested kernels cannot collide in
  // `launcher_module`.
  llvm::SmallVector<mlir::Operation*> ops_to_rename;
  for (mlir::Operation& op : nested_module.getBody()->getOperations()) {
    if (&op == nested_entry_func.getOperation()) {
      continue;
    }
    if (auto func = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(op)) {
      if (!func.isExternal()) {
        ops_to_rename.push_back(&op);
      }
    } else if (auto global = mlir::dyn_cast<mlir::LLVM::GlobalOp>(op)) {
      if (global.getAddrSpace() != kSharedMemoryAddrSpace) {
        ops_to_rename.push_back(&op);
      }
    }
  }

  for (mlir::Operation* op : ops_to_rename) {
    mlir::StringAttr old_name = mlir::SymbolTable::getSymbolName(op);
    std::string new_name = absl::StrCat(nested_fn_name, "_", old_name.str());
    if (mlir::failed(mlir::SymbolTable::replaceAllSymbolUses(
            op, builder.getStringAttr(new_name), nested_module))) {
      return absl::InternalError(
          absl::StrCat("Failed to rename symbol ", old_name.str()));
    }
    mlir::SymbolTable::setSymbolName(op, new_name);
    if (auto func = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(op)) {
      func.setLinkage(mlir::LLVM::Linkage::Internal);
    } else if (auto global = mlir::dyn_cast<mlir::LLVM::GlobalOp>(op)) {
      global.setLinkage(mlir::LLVM::Linkage::Private);
    }
  }

  if (auto decl = launcher_module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
          nested_fn_name)) {
    decl.erase();
  }

  llvm::SmallVector<mlir::Operation*> ops_to_move;
  for (mlir::Operation& op : nested_module.getBody()->getOperations()) {
    if (auto global = mlir::dyn_cast<mlir::LLVM::GlobalOp>(op)) {
      if (global.getAddrSpace() == kSharedMemoryAddrSpace &&
          launcher_module.lookupSymbol(global.getSymName()) != nullptr) {
        continue;
      }
    }
    if (auto func = mlir::dyn_cast<mlir::LLVM::LLVMFuncOp>(op)) {
      if (func.isExternal() &&
          launcher_module.lookupSymbol(func.getSymName()) != nullptr) {
        continue;
      }
    }
    ops_to_move.push_back(&op);
  }

  for (mlir::Operation* op : ops_to_move) {
    op->moveBefore(launcher_module.getBody(), launcher_module.getBody()->end());
  }

  return absl::OkStatus();
}

}  // namespace xla::gpu
