//===- Dialect.cpp - Dialect implementation -------------------------------===//
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

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectHooks.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Regex.h"
using namespace mlir;

// Registry for all dialect allocation functions.
static llvm::ManagedStatic<SmallVector<DialectAllocatorFunction, 8>>
    dialectRegistry;

// Registry for functions that set dialect hooks.
static llvm::ManagedStatic<SmallVector<DialectHooksSetter, 8>>
    dialectHooksRegistry;

/// Registers a specific dialect creation function with the system, typically
/// used through the DialectRegistration template.
void mlir::registerDialectAllocator(const DialectAllocatorFunction &function) {
  assert(function &&
         "Attempting to register an empty dialect initialize function");
  dialectRegistry->push_back(function);
}

/// Registers a function to set specific hooks for a specific dialect, typically
/// used through the DialectHooksRegistreation template.
void mlir::registerDialectHooksSetter(const DialectHooksSetter &function) {
  assert(
      function &&
      "Attempting to register an empty dialect hooks initialization function");

  dialectHooksRegistry->push_back(function);
}

/// Registers all dialects and their const folding hooks with the specified
/// MLIRContext.
void mlir::registerAllDialects(MLIRContext *context) {
  for (const auto &fn : *dialectRegistry)
    fn(context);
  for (const auto &fn : *dialectHooksRegistry) {
    fn(context);
  }
}

Dialect::Dialect(StringRef name, MLIRContext *context)
    : name(name), context(context), allowUnknownOps(false) {
  assert(isValidNamespace(name) && "invalid dialect namespace");
  registerDialect(context);
}

Dialect::~Dialect() {}

/// Parse a type registered to this dialect.
Type Dialect::parseType(StringRef tyData, Location loc) const {
  getContext()->emitError(loc, "dialect '" + getNamespace() +
                                   "' provides no type parsing hook");
  return Type();
}

/// Utility function that returns if the given string is a valid dialect
/// namespace.
bool Dialect::isValidNamespace(StringRef str) {
  if (str.empty())
    return true;
  llvm::Regex dialectNameRegex("^[a-zA-Z_][a-zA-Z_0-9\\$]*$");
  return dialectNameRegex.match(str);
}
