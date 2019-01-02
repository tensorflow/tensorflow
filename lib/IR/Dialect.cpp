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
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/ManagedStatic.h"
using namespace mlir;

// Registry for all dialect allocation functions.
static llvm::ManagedStatic<SmallVector<DialectAllocatorFunction, 8>>
    dialectRegistry;

// Registry for dialect's constant fold hooks.
static llvm::ManagedStatic<SmallVector<ConstantFoldHookAllocator, 8>>
    constantFoldHookRegistry;

/// Registers a specific dialect creation function with the system, typically
/// used through the DialectRegistration template.
void mlir::registerDialectAllocator(const DialectAllocatorFunction &function) {
  assert(function &&
         "Attempting to register an empty dialect initialize function");
  dialectRegistry->push_back(function);
}

/// Registers a constant fold hook for a specific dialect with the system.
void mlir::registerConstantFoldHook(const ConstantFoldHookAllocator &function) {
  assert(
      function &&
      "Attempting to register an empty constant fold hook initialize function");
  constantFoldHookRegistry->push_back(function);
}

/// Registers all dialects and their const folding hooks with the specified
/// MLIRContext.
void mlir::registerAllDialects(MLIRContext *context) {
  for (const auto &fn : *dialectRegistry)
    fn(context);
  for (const auto &fn : *constantFoldHookRegistry)
    fn(context);
}

Dialect::Dialect(StringRef namePrefix, MLIRContext *context)
    : namePrefix(namePrefix), context(context) {
  assert(!namePrefix.contains('.') &&
         "Dialect names cannot contain '.' characters.");
  registerDialect(context);
}

Dialect::~Dialect() {}
