/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"

namespace mlir {
namespace TF {

namespace {

// A helper class that generates name strings that are both uniques among
// a pre-defined set of existing strings and among the new strings it generates.
class NameUniquifier : public tensorflow::OpOrArgNameMapper {
 public:
  explicit NameUniquifier(const llvm::StringSet<> &existing_names)
      : existing_names_(existing_names) {}

 private:
  bool IsUnique(llvm::StringRef name) override {
    return !existing_names_.contains(name);
  }

  std::string GetName(tensorflow::OpOrVal op_or_val) override {
    llvm_unreachable("This method shouldn't be used.");
    return "";
  }

  const llvm::StringSet<> &existing_names_;
};

// Returns an updated SymbolRefAttr according to `symbol_renaming_map`.
// If the symbol name is not in the map, then the function returns the `old`
// SymbolRefAttr.
SymbolRefAttr GetUpdatedSymbolRefAttr(
    SymbolRefAttr old, const llvm::StringMap<StringRef> &symbol_renaming_map) {
  auto it = symbol_renaming_map.find(old.getRootReference());
  if (it == symbol_renaming_map.end()) {
    return old;
  }

  StringRef new_symbol_name = it->second;
  return SymbolRefAttr::get(new_symbol_name, old.getNestedReferences(),
                            old.getContext());
}

// A pass that renames all the private functions to new names that don't exist
// in the original module.
struct RenamePrivateFunctionPass
    : public PassWrapper<RenamePrivateFunctionPass, OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

void RenamePrivateFunctionPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Get all old function names
  llvm::StringSet<> old_private_func_names;
  for (auto func : module.getOps<FuncOp>()) {
    old_private_func_names.insert(func.getName());
  }

  // Update private function names
  NameUniquifier name_uniquifier(old_private_func_names);
  llvm::StringMap<StringRef> func_name_map;
  for (auto func : module.getOps<FuncOp>()) {
    if (func.isPrivate()) {
      StringRef old_name = func.getName();
      StringRef new_name = name_uniquifier.GetUniqueName(old_name);
      func.setName(new_name);
      func_name_map.insert(std::make_pair(old_name, new_name));
    }
  }

  // Update any SymbolRefAttr
  module.walk([&func_name_map](Operation *op) {
    for (NamedAttribute p : op->getAttrs()) {
      Identifier id = p.first;
      Attribute attr = p.second;
      if (auto symbol_ref = attr.dyn_cast<SymbolRefAttr>()) {
        SymbolRefAttr new_symbol_ref =
            GetUpdatedSymbolRefAttr(symbol_ref, func_name_map);
        if (new_symbol_ref != symbol_ref) {
          op->setAttr(id, new_symbol_ref);
        }
      } else if (auto array_attr = attr.dyn_cast<ArrayAttr>()) {
        // Update any SymbolRefAttr in the ArrayAttr
        SmallVector<Attribute, 4> new_array;
        new_array.reserve(array_attr.size());
        for (Attribute attr : array_attr.getValue()) {
          if (auto symbol_ref = attr.dyn_cast<SymbolRefAttr>()) {
            SymbolRefAttr new_symbol_ref =
                GetUpdatedSymbolRefAttr(symbol_ref, func_name_map);
            new_array.push_back(new_symbol_ref);
          } else {
            new_array.push_back(attr);
          }
        }
        auto new_array_attr = ArrayAttr::get(new_array, op->getContext());
        if (new_array_attr != array_attr) {
          op->setAttr(id, new_array_attr);
        }
      }
    }
  });
}

PassRegistration<RenamePrivateFunctionPass> tpu_pass(
    "tf-rename-private-functions",
    "Renames all the private functions to new names");

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateRenamePrivateFunctionPass() {
  return std::make_unique<RenamePrivateFunctionPass>();
}

}  // namespace TF
}  // namespace mlir
