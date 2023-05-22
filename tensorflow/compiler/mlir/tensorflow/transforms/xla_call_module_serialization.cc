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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/api/PortableApi.h"  // from @stablehlo
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/visitor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_call_module_attrs.h"

namespace mlir {
namespace TF {
namespace {

#define GEN_PASS_DEF_XLACALLMODULESERIALIZATIONPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"  // IWYU pragma: keep

// Creates a pruned module containing the XlaCallModule's entry function and
// other functions transitively called by the entry function.
FailureOr<OwningOpRef<ModuleOp>> PruneStablehloModule(ModuleOp module,
                                                      XlaCallModuleOp op) {
  auto entry_func_symbol =
      op->getAttrOfType<FlatSymbolRefAttr>(kStablehloEntryFunctionAttrName);
  if (entry_func_symbol == nullptr) {
    return op.emitOpError() << "does not have "
                            << kStablehloEntryFunctionAttrName << " attribute";
  }
  auto entry_func_name = entry_func_symbol.getValue();

  OwningOpRef<ModuleOp> stablehlo_module;

  auto pruned = CreatePrunedModule(module, {entry_func_name});
  if (failed(pruned)) {
    return module.emitError()
           << "failed to create stablehlo module from entry function "
           << entry_func_name;
  }
  stablehlo_module = std::move(*pruned);

  // CreatePrunedModule copies the top-level module's attributes to the new
  // module. But we should restore the deserialized stablehlo module's
  // attributes to the reconstructed stablehlo module. The stablehlo module's
  // attributes can contain important information such as SPMD num_replicas and
  // num_partitions.
  auto original_stablehlo_module_attrs =
      op->getAttrOfType<DictionaryAttr>(kStablehloModuleAttrsAttrName);
  if (original_stablehlo_module_attrs) {
    (*stablehlo_module)->setAttrs(original_stablehlo_module_attrs);
  } else {
    (*stablehlo_module)->setAttrs(llvm::ArrayRef<NamedAttribute>());
  }

  // Remove _from_xla_call_module attr from all functions in the reconstructed
  // stablehlo module.
  stablehlo_module->walk(
      [&](func::FuncOp f) { f->removeAttr(kFromXlaCallModuleAttrName); });

  // Entry function must be public and has symbol name "@main".
  auto entry_func =
      stablehlo_module->lookupSymbol<mlir::func::FuncOp>(entry_func_name);
  if (entry_func == nullptr) {
    return stablehlo_module->emitOpError()
           << "does not have function " << entry_func_name;
  }
  entry_func.setPublic();
  entry_func.setName(kStablehloMainFunctionName);

  return stablehlo_module;
}

// Serializes the stablehlo module into bytecode.
FailureOr<std::string> SerializeStablehlo(ModuleOp stablehlo_module) {
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  if (mlir::failed(stablehlo::serializePortableArtifact(
          stablehlo_module, stablehlo::getCurrentVersion(), os))) {
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
LogicalResult SerializeXlaCallModule(ModuleOp module, XlaCallModuleOp op) {
  auto stablehlo_module = PruneStablehloModule(module, op);
  if (failed(stablehlo_module)) {
    return failure();
  }

  auto bytecode = SerializeStablehlo(**stablehlo_module);
  if (failed(bytecode)) {
    return failure();
  }

  op.setModule(*bytecode);
  op->removeAttr(kStablehloEntryFunctionAttrName);

  return success();
}

class XlaCallModuleSerializationPass
    : public impl::XlaCallModuleSerializationPassBase<
          XlaCallModuleSerializationPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    mlir::WalkResult result =
        module.walk([&](mlir::TF::XlaCallModuleOp xla_call_module) {
          if (failed(SerializeXlaCallModule(module, xla_call_module))) {
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    if (result.wasInterrupted()) {
      return signalPassFailure();
    }

    RemoveSerializedStablehloFunctions(module);
  }

 private:
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
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateXlaCallModuleSerializationPass() {
  return std::make_unique<XlaCallModuleSerializationPass>();
}

}  // namespace TF
}  // namespace mlir
