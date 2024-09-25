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
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/stablehlo_custom_call.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_call_module_attrs.h"
#include "tensorflow/compiler/tf2xla/kernels/xla_call_module_loader.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep
#include "tsl/platform/statusor.h"

namespace mlir {
namespace TF {
namespace {

#define GEN_PASS_DEF_XLACALLMODULEDESERIALIZATIONPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// `tf.backend_config` is a DictionaryAttr, JAX2TF sets the value of its
// i64 attribute `called_index` to the TF function's name.
constexpr llvm::StringRef kTfBackendConfigAttrName = "tf.backend_config";
constexpr llvm::StringRef kCalledIndexAttrName = "called_index";
constexpr llvm::StringRef kCalledFuncAttrName = "called_func";

// Deserialize the StableHLO module embedded in XlaCallModuleOp's module
// attribute.
absl::StatusOr<OwningOpRef<ModuleOp>> DeserializeStablehlo(MLIRContext *context,
                                                           XlaCallModuleOp op) {
  std::vector<std::string> disabled_checks;
  for (auto attr : op.getDisabledChecks().getAsRange<StringAttr>()) {
    disabled_checks.push_back(attr.getValue().str());
  }
  std::vector<std::string> platforms;
  for (auto attr : op.getPlatforms().getAsRange<StringAttr>()) {
    platforms.push_back(attr.getValue().str());
  }
  TF_ASSIGN_OR_RETURN(
      auto loader,
      tensorflow::XlaCallModuleLoader::Create(
          context, static_cast<int>(op.getVersion()), op.getModule().str(),
          std::move(disabled_checks), std::move(platforms),
          /*num_invocation_args=*/op.getArgs().size(),
          op.getHasTokenInputOutput()));
  return std::move(*loader).module();
}

// Renames functions in the stablehlo module to avoid naming conflicts with
// existing functions in the tf module.
// Sets _from_xla_call_module attribute for each stablehlo function.
// Returns the new stablehlo main function's name or error.
//
// If we directly insert stablehlo functions into tf module, MLIR will rename
// the stablehlo functions themselves in the tf module automatically to avoid
// naming conflicts. But we need to rename the function calls inside the
// stablehlo functions as well. So we first do this renaming in the stablehlo
// module itself without inserting into the tf module.
FailureOr<StringAttr> RenameStablehloFunctions(
    MLIRContext *context, SymbolTableCollection &symbol_tables,
    ModuleOp tf_module, ModuleOp stablehlo_module) {
  SymbolTable &tf_symbol_table = symbol_tables.getSymbolTable(tf_module);
  // `stablehlo_module` is deleted right after the deserialization, so no need
  // to store its `SymbolTable` to `SymbolTableCollection`.
  SymbolTable stablehlo_symbol_table(stablehlo_module);

  Builder builder(context);
  StringAttr main_func_name;
  for (auto func : stablehlo_module.getOps<func::FuncOp>()) {
    const bool is_main_func = func.getSymName() == kStablehloMainFunctionName;
    if (tf_symbol_table.lookup(func.getSymName())) {
      if (failed(stablehlo_symbol_table.renameToUnique(
              func, {&tf_symbol_table, &stablehlo_symbol_table}))) {
        return func.emitError()
               << "failed to rename StableHLO function " << func.getSymName();
      }
    }
    if (is_main_func) {
      main_func_name = func.getSymNameAttr();
    }
    func->setAttr(kFromXlaCallModuleAttrName, builder.getUnitAttr());
  }
  if (!main_func_name) {
    return stablehlo_module.emitError()
           << "StableHLO module does not have an entry function";
  }
  return main_func_name;
}

// Moves functions from one module to another.
// The moved functions are set to private.
void MoveFunctions(SymbolTableCollection &symbol_tables, ModuleOp from,
                   ModuleOp to) {
  SymbolTable &to_symbol_table = symbol_tables.getSymbolTable(to);
  for (auto func : llvm::make_early_inc_range(from.getOps<func::FuncOp>())) {
    func->remove();
    func.setPrivate();
    to_symbol_table.insert(func);
  }
}

void CopyStablehloModuleAttrs(ModuleOp stablehlo_module, XlaCallModuleOp op) {
  op->setAttr(kStablehloModuleAttrsAttrName,
              stablehlo_module->getAttrDictionary());
}

// Symbolizes `called_index` attributes in custom all ops to `called_func`.
LogicalResult SymbolizeCustomCallCalledIndex(
    ModuleOp module, llvm::ArrayRef<SymbolRefAttr> function_list) {
  WalkResult result =
      module.walk([&](stablehlo::CustomCallOp op) {
        if (!IsTfFuncCustomCall(op)) {
          return WalkResult::advance();
        }

        auto backend_config =
            op->getAttrOfType<DictionaryAttr>(kTfBackendConfigAttrName);
        if (!backend_config) {
          op->emitOpError()
              << "is missing attribute '" << kTfBackendConfigAttrName << "'";
          return WalkResult::interrupt();
        }

        auto called_index_attr = mlir::dyn_cast_or_null<IntegerAttr>(
            backend_config.get(kCalledIndexAttrName));
        if (!called_index_attr) {
          op->emitOpError()
              << "is missing attribute '" << kCalledIndexAttrName << "'";
          return WalkResult::interrupt();
        }
        int called_index = called_index_attr.getInt();
        if (called_index < 0 || called_index >= function_list.size()) {
          op->emitOpError()
              << "references function #" << called_index
              << " but enclosing XlaCallModule has a function list of size "
              << function_list.size();
          return WalkResult::interrupt();
        }

        llvm::SmallVector<NamedAttribute> new_config;
        // Copy the attributes in the current config except `called_index`.
        for (auto attr : backend_config) {
          if (attr.getName() != kCalledIndexAttrName) {
            new_config.push_back(attr);
          }
        }

        Builder builder(op.getContext());
        // Sets the `called_index` attribute to the TF function's name.
        new_config.push_back(builder.getNamedAttr(kCalledFuncAttrName,
                                                  function_list[called_index]));

        // Sets the `tf.backend_config` attribute to the `new_config`.
        op->setAttr(kTfBackendConfigAttrName,
                    builder.getDictionaryAttr(new_config));

        return WalkResult::advance();
      });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult DeserializeXlaCallModule(MLIRContext *context,
                                       SymbolTableCollection &symbol_tables,
                                       ModuleOp module, XlaCallModuleOp op) {
  auto deserialized = DeserializeStablehlo(context, op);
  if (!deserialized.ok()) {
    return op.emitOpError()
           << "failed to deserialize StableHLO module from XlaCallModule: "
           << deserialized.status().ToString();
  }
  OwningOpRef<ModuleOp> stablehlo_module = *std::move(deserialized);

  CopyStablehloModuleAttrs(*stablehlo_module, op);

  auto main_func = RenameStablehloFunctions(context, symbol_tables, module,
                                            stablehlo_module.get());
  if (failed(main_func)) {
    return failure();
  }

  // Translate `called_index` in TF function custom calls into symbol
  // references. `function_list` attribute is needed after that.
  llvm::SmallVector<SymbolRefAttr> function_list(
      op.getFunctionList().getAsRange<SymbolRefAttr>());
  if (failed(
          SymbolizeCustomCallCalledIndex(*stablehlo_module, function_list))) {
    return failure();
  }
  op.removeFunctionListAttr();

  MoveFunctions(symbol_tables, *stablehlo_module, module);

  // Module is deserialized, we set an empty string to it instead removing
  // it because it's a required attribute.
  op.setModule("");
  // Set the stablehlo main function as a symbol attribute. This is required
  // because we not only need this to look up the StableHLO function called by
  // XlaCallModule, but also need the symbol reference to prevent DCE from
  // removing the stablehlo functions from the top-level module.
  op->setAttr(kStablehloEntryFunctionAttrName, SymbolRefAttr::get(*main_func));

  return success();
}

class XlaCallModuleDeserializationPass
    : public impl::XlaCallModuleDeserializationPassBase<
          XlaCallModuleDeserializationPass> {
 public:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    XlaCallModuleDeserializationPassBase::getDependentDialects(registry);
    mlir::func::registerAllExtensions(registry);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTableCollection symbol_tables;
    WalkResult result = module.walk([&](XlaCallModuleOp op) {
      if (failed(DeserializeXlaCallModule(&getContext(), symbol_tables, module,
                                          op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateXlaCallModuleDeserializationPass() {
  return std::make_unique<XlaCallModuleDeserializationPass>();
}

}  // namespace TF
}  // namespace mlir
