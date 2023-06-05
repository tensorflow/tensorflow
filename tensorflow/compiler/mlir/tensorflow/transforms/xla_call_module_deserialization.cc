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

#include "absl/strings/str_format.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/stablehlo_custom_call.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_call_module_attrs.h"
#include "tensorflow/compiler/tf2xla/kernels/xla_call_module_loader.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep
#include "tensorflow/tsl/platform/statusor.h"

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

// The function name format for the deserialized stablehlo functions:
//   _stablehlo_{original function name}_{index}.
constexpr const char *kNewFuncNameFormat = "_stablehlo_%s_%d";

// Deserialize the StableHLO module embedded in XlaCallModuleOp's module
// attribute.
tsl::StatusOr<OwningOpRef<ModuleOp>> DeserializeStablehlo(MLIRContext *context,
                                                          XlaCallModuleOp op) {
  std::vector<std::string> dim_args_spec;
  for (auto attr : op.getDimArgsSpec().getAsRange<StringAttr>()) {
    dim_args_spec.push_back(attr.getValue().str());
  }
  // XlaCallModuleOp OpKernel will determine platform index when running
  // TF2XLA. We don't know the device/platform type in this MLIR pass, so
  // we set platform_index to -1.
  TF_ASSIGN_OR_RETURN(auto loader,
                      tensorflow::XlaCallModuleLoader::Create(
                          context, static_cast<int>(op.getVersion()),
                          op.getModule().str(), dim_args_spec,
                          /*platform_index=*/-1));
  return std::move(*loader).module();
}

// Returns a new function name in the kNewFuncNameFormat.
// The new name is unique in the symbol table.
std::string NewFuncName(const SymbolTable &symbol_table,
                        const llvm::StringRef func_name) {
  uint64_t index = 0;
  std::string new_func_name;
  do {
    new_func_name = absl::StrFormat(kNewFuncNameFormat, func_name, index++);
  } while (symbol_table.lookup(new_func_name));
  return new_func_name;
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
  SymbolTable &tf_sym_table = symbol_tables.getSymbolTable(tf_module);
  SymbolTable &stablehlo_sym_table =
      symbol_tables.getSymbolTable(stablehlo_module);
  Builder builder(context);
  StringAttr new_main_func_name;
  for (auto func : stablehlo_module.getOps<func::FuncOp>()) {
    auto new_func_name =
        builder.getStringAttr(NewFuncName(tf_sym_table, func.getSymName()));
    if (func.getSymName() == kStablehloMainFunctionName) {
      new_main_func_name = new_func_name;
    }
    if (failed(stablehlo_sym_table.replaceAllSymbolUses(func, new_func_name,
                                                        stablehlo_module))) {
      return failure();
    }
    func.setName(new_func_name);
    func->setAttr(kFromXlaCallModuleAttrName, builder.getUnitAttr());
  }
  return new_main_func_name;
}

// Copies functions from one module to another.
// The copied functions are set to private.
void CopyFunctions(SymbolTableCollection &symbol_tables, ModuleOp from,
                   ModuleOp to) {
  SymbolTable &to_sym_table = symbol_tables.getSymbolTable(to);
  for (auto func : from.getOps<func::FuncOp>()) {
    auto f = func.clone();
    f.setPrivate();
    to_sym_table.insert(f);
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

        auto called_index_attr = backend_config.get(kCalledIndexAttrName)
                                     .dyn_cast_or_null<IntegerAttr>();
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

  CopyFunctions(symbol_tables, *stablehlo_module, module);

  // Translate `called_index` in TF function custom calls into symbol
  // references. `function_list` attribute is needed after that.
  SmallVector<SymbolRefAttr> function_list(
      op.getFunctionList().getAsRange<SymbolRefAttr>());
  if (failed(SymbolizeCustomCallCalledIndex(module, function_list))) {
    return failure();
  }
  op.removeFunctionListAttr();

  // Module is deserialized, we set an empty string to it instead removing
  // it because it's a required attribute.
  op.setModule("");
  // Set the stablehlo main function as a symbol attribute.
  // This is required because we not only need this to look up the
  // stablehlo function called by XlaCallModule, but also need the symbol
  // reference to prevent DCE from removing the stablehlo functions from the
  // top-level module.
  op->setAttr(kStablehloEntryFunctionAttrName,
              SymbolRefAttr::get(main_func.value()));

  return success();
}

class XlaCallModuleDeserializationPass
    : public impl::XlaCallModuleDeserializationPassBase<
          XlaCallModuleDeserializationPass> {
 public:
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
