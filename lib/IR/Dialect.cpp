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

/// Register a specific dialect creation function with the system, typically
/// used through the DialectRegistration template.
void mlir::registerDialectAllocator(const DialectAllocatorFunction &function) {
  assert(function && "Attempting to register an empty op initialize function");
  dialectRegistry->push_back(function);
}

/// Registers all dialects with the specified MLIRContext.
void mlir::registerAllDialects(MLIRContext *context) {
  for (const auto &fn : *dialectRegistry)
    fn(context);
}

Dialect::Dialect(StringRef opPrefix, MLIRContext *context)
    : opPrefix(opPrefix), context(context) {
  registerDialect(context);
}

Dialect::~Dialect() {}
