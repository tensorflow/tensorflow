//===- Module.cpp - MLIR Module Class -------------------------------===//
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
using namespace mlir;

Module::Module(MLIRContext *context) : context(context) {}

/// Look up a function with the specified name, returning null if no such
/// name exists.  Function names never include the @ on them.
Function *Module::getNamedFunction(StringRef name) {
  return getNamedFunction(Identifier::get(name, context));
}

/// Look up a function with the specified name, returning null if no such
/// name exists.  Function names never include the @ on them.
Function *Module::getNamedFunction(Identifier name) {
  auto it = symbolTable.find(name);
  return it != symbolTable.end() ? it->second : nullptr;
}
