//===- Module.cpp - MLIR Module Operation ---------------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/IR/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Module Operation.
//===----------------------------------------------------------------------===//

void ModuleOp::build(Builder *builder, OperationState &result,
                     Optional<StringRef> name) {
  ensureTerminator(*result.addRegion(), *builder, result.location);
  if (name)
    result.attributes.push_back(builder->getNamedAttr(
        mlir::SymbolTable::getSymbolAttrName(), builder->getStringAttr(*name)));
}

/// Construct a module from the given context.
ModuleOp ModuleOp::create(Location loc, Optional<StringRef> name) {
  OperationState state(loc, "module");
  Builder builder(loc->getContext());
  ModuleOp::build(&builder, state, name);
  return cast<ModuleOp>(Operation::create(state));
}

ParseResult ModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  // If the name is present, parse it.
  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(
      nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes);

  // If module attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the module body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, llvm::None, llvm::None))
    return failure();

  // Ensure that this module has a valid terminator.
  ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void ModuleOp::print(OpAsmPrinter &p) {
  p << "module";

  if (Optional<StringRef> name = getName()) {
    p << ' ';
    p.printSymbolName(*name);
  }

  // Print the module attributes.
  p.printOptionalAttrDictWithKeyword(getAttrs(),
                                     {mlir::SymbolTable::getSymbolAttrName()});

  // Print the region.
  p.printRegion(getOperation()->getRegion(0), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

LogicalResult ModuleOp::verify() {
  auto &bodyRegion = getOperation()->getRegion(0);

  // The body must contain a single basic block.
  if (!has_single_element(bodyRegion))
    return emitOpError("expected body region to have a single block");

  // Check that the body has no block arguments.
  auto *body = &bodyRegion.front();
  if (body->getNumArguments() != 0)
    return emitOpError("expected body to have no arguments");

  // Check that none of the attributes are non-dialect attributes, except for
  // the symbol name attribute.
  for (auto attr : getOperation()->getAttrList().getAttrs()) {
    if (!attr.first.strref().contains('.') &&
        attr.first.strref() != mlir::SymbolTable::getSymbolAttrName())
      return emitOpError(
                 "can only contain dialect-specific attributes, found: '")
             << attr.first << "'";
  }

  return success();
}

/// Return body of this module.
Region &ModuleOp::getBodyRegion() { return getOperation()->getRegion(0); }
Block *ModuleOp::getBody() { return &getBodyRegion().front(); }

Optional<StringRef> ModuleOp::getName() {
  if (auto nameAttr =
          getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName()))
    return nameAttr.getValue();
  return llvm::None;
}
