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

// Insert `module_terminator` at the end of the region's only block if it does
// not have a terminator already. If the region is empty, insert a new block
// first.
static void ensureModuleTerminator(Region &region, Builder &builder,
                                   Location loc) {
  impl::ensureRegionTerminator<ModuleTerminatorOp>(region, builder, loc);
}

void ModuleOp::build(Builder *builder, OperationState *result) {
  ensureModuleTerminator(*result->addRegion(), *builder, result->location);
}

/// Construct a module from the given context.
ModuleOp ModuleOp::create(MLIRContext *context) {
  OperationState state(UnknownLoc::get(context), "module");
  Builder builder(context);
  ModuleOp::build(&builder, &state);
  return llvm::cast<ModuleOp>(Operation::create(state));
}

ParseResult ModuleOp::parse(OpAsmParser *parser, OperationState *result) {
  // If module attributes are present, parse them.
  if (succeeded(parser->parseOptionalKeyword("attributes")))
    if (parser->parseOptionalAttributeDict(result->attributes))
      return failure();

  // Parse the module body.
  auto *body = result->addRegion();
  if (parser->parseRegion(*body, llvm::None, llvm::None))
    return failure();

  // Ensure that this module has a valid terminator.
  ensureModuleTerminator(*body, parser->getBuilder(), result->location);
  return success();
}

void ModuleOp::print(OpAsmPrinter *p) {
  *p << "module";

  // Print the module attributes.
  auto attrs = getAttrs();
  if (!attrs.empty()) {
    *p << " attributes";
    p->printOptionalAttrDict(attrs, {});
  }

  // Print the region.
  p->printRegion(getOperation()->getRegion(0), /*printEntryBlockArgs=*/false,
                 /*printBlockTerminators=*/false);
}

LogicalResult ModuleOp::verify() {
  auto &bodyRegion = getOperation()->getRegion(0);

  // The body must contain a single basic block.
  if (bodyRegion.empty() || std::next(bodyRegion.begin()) != bodyRegion.end())
    return emitOpError("expected body region to have a single block");

  // Check that the body has no block arguments.
  auto *body = &bodyRegion.front();
  if (body->getNumArguments() != 0)
    return emitOpError("expected body to have no arguments");

  if (body->empty() || !isa<ModuleTerminatorOp>(body->back())) {
    return emitOpError("expects region to end with '" +
                       ModuleTerminatorOp::getOperationName() + "'")
               .attachNote()
           << "in custom textual format, the absence of terminator implies '"
           << ModuleTerminatorOp::getOperationName() << "'";
  }

  for (auto &op : *body) {
    if (op.getNumResults() != 0) {
      return emitOpError()
          .append("may not contain operations that produce results")
          .attachNote(op.getLoc())
          .append("see offending operation defined here");
    }
  }

  // Check that all functions are uniquely named.
  llvm::StringMap<Location> nameToOrigLoc;
  for (auto fn : getOps<FuncOp>()) {
    auto it = nameToOrigLoc.try_emplace(fn.getName(), fn.getLoc());
    if (!it.second)
      return fn.emitError()
          .append("redefinition of symbol named '", fn.getName(), "'")
          .attachNote(it.first->second)
          .append("see existing symbol definition here");
  }

  return success();
}

/// Return body of this module.
Region &ModuleOp::getBodyRegion() { return getOperation()->getRegion(0); }
Block *ModuleOp::getBody() { return &getBodyRegion().front(); }

//===----------------------------------------------------------------------===//
// Module Terminator Operation.
//===----------------------------------------------------------------------===//

LogicalResult ModuleTerminatorOp::verify() {
  if (!isa_and_nonnull<ModuleOp>(getOperation()->getParentOp()))
    return emitOpError() << "is expected to terminate a '"
                         << ModuleOp::getOperationName() << "' operation";
  return success();
}
