/* Copyright 2025 The JAX Authors.

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

#include "xla/mosaic/serde.h"

#include <optional>
#include <string>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"

namespace jaxlib::mosaic {

namespace {

llvm::StringRef mangle(llvm::StringRef name, llvm::StringRef prefix,
                       std::string* storage) {
  storage->clear();
  storage->reserve(prefix.size() + name.size());
  storage->insert(storage->end(), prefix.begin(), prefix.end());
  storage->insert(storage->end(), name.begin(), name.end());
  return *storage;
}

std::optional<llvm::StringRef> demangle(llvm::StringRef name,
                                        llvm::StringRef prefix) {
  if (!name.starts_with(prefix)) {
    return std::nullopt;
  }
  return name.drop_front(prefix.size());
}

}  // namespace

mlir::LogicalResult RunSerde(
    mlir::ModuleOp module, const llvm::StringMap<SerdeRuleType>& upgrade_rules,
    const llvm::StringMap<SerdeRuleType>& downgrade_rules, bool serialize,
    SerdeOptions options) {
  int version = options.highest_version;
  int serialize_version = options.serialize_version;
  if (!serialize && serialize_version != -1) {
    module.emitError("Cannot deserialize to a specific version");
    return mlir::failure();
  }
  if (serialize && serialize_version > version) {
    module.emitError("The highest supported version is ")
        << version << " but requested serialization at version "
        << serialize_version;
    return mlir::failure();
  }
  if (serialize && !module->getContext()->allowsUnregisteredDialects()) {
    module.emitError() << "Cannot serialize within a context that does not "
                          "allow unregistered dialects";
    return mlir::failure();
  }
  if (serialize) {
    module->setAttr(
        options.version_attr_name,
        mlir::IntegerAttr::get(mlir::IntegerType::get(module->getContext(), 64),
                               serialize_version));
  } else {
    mlir::IntegerAttr version_attr =
        module->getAttrOfType<mlir::IntegerAttr>(options.version_attr_name);
    if (!version_attr) {
      module->emitError("Missing or invalid version attribute");
      return mlir::failure();
    }
    if (version_attr.getInt() > version) {
      module->emitError("Unsupported version: expected <= ")
          << version << " but got " << version_attr.getInt();
      return mlir::failure();
    }
    version = version_attr.getInt();
    module->removeAttr(options.version_attr_name);
  }
  std::string storage;
  auto result = module.walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::ModuleOp>(op)) {  // Don't mangle the ModuleOp itself.
      return mlir::WalkResult::advance();
    }
    std::optional<mlir::OperationName> new_name;
    if (serialize) {
      auto new_name_str = mangle(op->getName().getStringRef(),
                                 options.dialect_prefix, &storage);
      new_name = mlir::OperationName(new_name_str, op->getContext());
    } else {
      if (auto demangled =
              demangle(op->getName().getStringRef(), options.dialect_prefix)) {
        auto new_name_str = *demangled;
        if (auto registered = mlir::RegisteredOperationName::lookup(
                new_name_str, op->getContext())) {
          new_name = *registered;
        } else {
          new_name = mlir::OperationName(new_name_str, op->getContext());
        }
      } else {
        op->emitError("Operation not in a serialized form");
        return mlir::WalkResult::interrupt();
      }
      // Upgrade the op to the current version, if needed.
      if (const auto rule = upgrade_rules.find(new_name->getStringRef());
          rule != upgrade_rules.end()) {
        if (rule->second(op, version).failed()) {
          return mlir::WalkResult::interrupt();
        }
      }
    }
    auto new_op = mlir::Operation::create(
        op->getLoc(), *new_name, op->getResultTypes(), op->getOperands(),
        op->getAttrs(), nullptr, op->getSuccessors(), op->getRegions());
    // Downgrade the op to the target version, if needed.
    bool downgrade_failed = false;
    if (serialize && version != serialize_version) {
      if (const auto rule = downgrade_rules.find(op->getName().getStringRef());
          rule != downgrade_rules.end()) {
        downgrade_failed = rule->second(new_op, serialize_version).failed();
      }
    }
    op->getBlock()->getOperations().insertAfter(mlir::Block::iterator(op),
                                                new_op);
    op->replaceAllUsesWith(new_op->getResults());
    op->erase();
    return downgrade_failed ? mlir::WalkResult::interrupt()
                            : mlir::WalkResult::advance();
  });
  return result.wasInterrupted() ? mlir::failure() : mlir::success();
}

}  // namespace jaxlib::mosaic
