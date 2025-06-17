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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/Version.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/stablehlo_custom_call.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/visitor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_call_module_attrs.h"

namespace mlir {
namespace TF {
namespace {

#define GEN_PASS_DEF_XLACALLMODULESERIALIZATIONPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"  // IWYU pragma: keep

// `tf.backend_config` is a DictionaryAttr, JAX2TF sets the value of its
// i64 attribute `called_index` to the TF function's name.
constexpr llvm::StringRef kTfBackendConfigAttrName = "tf.backend_config";
constexpr llvm::StringRef kCalledIndexAttrName = "called_index";
constexpr llvm::StringRef kCalledFuncAttrName = "called_func";

// Converts `called_func` attributes in custom call ops back to `called_index`.
FailureOr<ArrayAttr> DesymbolizeCustomCallCalledIndex(ModuleOp module) {
  Builder builder(module.getContext());

  SmallVector<Attribute> function_list;
  llvm::DenseMap<SymbolRefAttr, int> called_indexes;

  WalkResult result = module.walk([&](stablehlo::CustomCallOp op) {
    if (!IsTfFuncCustomCall(op)) {
      return WalkResult::advance();
    }

    auto backend_config =
        op->getAttrOfType<DictionaryAttr>(kTfBackendConfigAttrName);
    if (!backend_config) {
      op->emitOpError() << "is missing attribute '" << kTfBackendConfigAttrName
                        << "'";
      return WalkResult::interrupt();
    }
    auto called_func = mlir::dyn_cast_or_null<SymbolRefAttr>(
        backend_config.get(kCalledFuncAttrName));
    if (!called_func) {
      op->emitOpError() << "is missing attribute '" << kCalledFuncAttrName
                        << "'";
      return WalkResult::interrupt();
    }

    llvm::SmallVector<NamedAttribute> new_config;
    // Copy the attributes in the current config except `called_func`.
    for (auto attr : backend_config) {
      if (attr.getName() != kCalledFuncAttrName) {
        new_config.push_back(attr);
      }
    }

    auto [it, inserted] =
        called_indexes.insert({called_func, called_indexes.size()});
    if (inserted) {
      function_list.push_back(called_func);
    }

    // Set the `called_index` attribute to the TF function's name.
    new_config.push_back(builder.getNamedAttr(
        kCalledIndexAttrName, builder.getI64IntegerAttr(it->second)));

    // Set the `tf.backend_config` attribute to the `new_config`.
    op->setAttr(kTfBackendConfigAttrName,
                builder.getDictionaryAttr(new_config));

    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }

  return builder.getArrayAttr(function_list);
}

// Creates a pruned module containing the XlaCallModule's entry function and
// other functions transitively called by the entry function.
FailureOr<OwningOpRef<ModuleOp>> PruneStablehloModule(
    SymbolTableCollection& symbol_table, ModuleOp module, XlaCallModuleOp op) {
  auto entry_func_symbol =
      op->getAttrOfType<FlatSymbolRefAttr>(kStablehloEntryFunctionAttrName);
  if (!entry_func_symbol) {
    return op.emitOpError() << "does not have "
                            << kStablehloEntryFunctionAttrName << " attribute";
  }
  auto entry_func =
      symbol_table.lookupSymbolIn<func::FuncOp>(module, entry_func_symbol);
  if (!entry_func) {
    return op.emitOpError()
           << "references an unknown entry function " << entry_func_symbol;
  }

  OpBuilder builder(module.getContext());

  OwningOpRef<ModuleOp> stablehlo_module =
      builder.create<ModuleOp>(op.getLoc());
  builder.setInsertionPointToEnd(stablehlo_module->getBody());

  // Copy all referenced StableHLO functions to the new module.
  WalkResult result = WalkReachableFunctions(
      entry_func,
      [&](func::FuncOp f) -> WalkResult {
        if (!f->hasAttr(kFromXlaCallModuleAttrName)) {
          return WalkResult::advance();
        }

        auto cloned = llvm::cast<func::FuncOp>(builder.clone(*f));
        cloned->removeAttr(kFromXlaCallModuleAttrName);

        if (f == entry_func) {
          // Entry function must be public and has symbol name "@main".
          cloned.setPublic();
          cloned.setName(kStablehloMainFunctionName);
        } else {
          cloned.setPrivate();
        }

        return WalkResult::advance();
      },
      &symbol_table);
  if (result.wasInterrupted()) {
    return failure();
  }

  // Rewrite `custom_call`'s `called_func` attribute to `called_index`.
  auto function_list = DesymbolizeCustomCallCalledIndex(*stablehlo_module);
  if (failed(function_list)) return failure();
  op.setFunctionListAttr(*function_list);

  // Restore the deserialized stablehlo module's attributes to the reconstructed
  // stablehlo module. The stablehlo module's attributes can contain important
  // information such as SPMD num_replicas and num_partitions.
  auto original_stablehlo_module_attrs =
      op->getAttrOfType<DictionaryAttr>(kStablehloModuleAttrsAttrName);
  if (original_stablehlo_module_attrs) {
    (*stablehlo_module)->setAttrs(original_stablehlo_module_attrs);
    // Now, remove the attribute because later passes may not know how to handle
    // it, we may encounter errors such as:
    // "Unhandled attribute kind for attribute '_stablehlo_module_attrs'".
    op->removeAttr(kStablehloModuleAttrsAttrName);
  }

  return stablehlo_module;
}

// Serializes the stablehlo module into bytecode.
FailureOr<std::string> SerializeStablehlo(ModuleOp stablehlo_module,
                                          StringRef target_version) {
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  // We need to pass `allowOtherDialects=true` if
  // `stablehlo_version >= 1.11.0`, since the lowered module from JAX can
  // have a mix of StableHLO and Shardy dialects.
  vhlo::Version mixed_serialization_ok = vhlo::Version(1, 11, 0);
  bool allow_other_dialects =
      mixed_serialization_ok <= vhlo::Version::fromString(target_version);
  if (mlir::failed(stablehlo::serializePortableArtifact(
          stablehlo_module, target_version, os, allow_other_dialects))) {
    return stablehlo_module.emitError()
           << "failed to serialize the pruned stablehlo module";
  }
  return bytecode;
}

// Serializes the stablehlo functions called by XlaCallModuleOp to bytecode
// and embeds the bytecode in XlaCallModuleOp's `module` attribute.
//
// The stablehlo functions include the function referred by XlaCallModuleOp's
// `_entry_function` attribute, and any stablehlo functions called transitively
// from the entry function.
LogicalResult SerializeXlaCallModule(SymbolTableCollection& symbol_table,
                                     ModuleOp module, XlaCallModuleOp op) {
  auto stablehlo_module = PruneStablehloModule(symbol_table, module, op);
  if (failed(stablehlo_module)) {
    return failure();
  }

  // Use the StableHLO version set during deserialization.
  auto stablehlo_version =
      op->getAttrOfType<StringAttr>(kStablehloVersionAttrName);
  if (!stablehlo_version) {
    return op->emitError() << "does not have " << kStablehloVersionAttrName
                           << " attribute";
  }

  StringRef target_version = stablehlo_version.getValue();
  auto bytecode = SerializeStablehlo(**stablehlo_module, target_version);
  if (failed(bytecode)) {
    return failure();
  }

  op.setModule(*bytecode);
  op->removeAttr(kStablehloEntryFunctionAttrName);
  op->removeAttr(kStablehloVersionAttrName);

  return success();
}

// Removes the serialized stablehlo functions, because `XlaCallModuleOp` no
// longer has `_entry_function` attribute referencing the stablehlo main
// function, so all stablehlo functions are of no use in the top-level module.
//
// Walk the module to find functions with `_from_xla_call_module` attribute,
// and remove them.
void RemoveSerializedStablehloFunctions(ModuleOp module) {
  module.walk([&](func::FuncOp f) {
    if (f->hasAttr(kFromXlaCallModuleAttrName)) {
      f->erase();
    }
  });
}

class XlaCallModuleSerializationPass
    : public impl::XlaCallModuleSerializationPassBase<
          XlaCallModuleSerializationPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::SymbolTableCollection symbol_table;

    mlir::WalkResult result =
        module.walk([&](mlir::TF::XlaCallModuleOp xla_call_module) {
          if (failed(SerializeXlaCallModule(symbol_table, module,
                                            xla_call_module))) {
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    if (result.wasInterrupted()) {
      return signalPassFailure();
    }

    RemoveSerializedStablehloFunctions(module);
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateXlaCallModuleSerializationPass() {
  return std::make_unique<XlaCallModuleSerializationPass>();
}

}  // namespace TF
}  // namespace mlir
