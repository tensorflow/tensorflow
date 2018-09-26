//===- OperationSet.cpp - OperationSet implementation ---------------------===//
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

#include "mlir/IR/OperationSet.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using llvm::StringMap;

// Registry for all Op initialization functions.
static llvm::ManagedStatic<std::vector<InitializeOpsFunction>> opRegistry;

OpInitializeRegistration::OpInitializeRegistration(
    const InitializeOpsFunction &function) {
  assert(function && "Attempting to register an empty op initialize function");
  opRegistry->push_back(function);
}

void mlir::initializeAllRegisteredOps(MLIRContext *context) {
  for (const auto &fn : *opRegistry)
    fn(context);
}

OpAsmParser::~OpAsmParser() {}

// The fallback for the printer is to reject the short form.
bool OpState::parse(OpAsmParser *parser, OperationState *result) {
  return parser->emitError(parser->getNameLoc(), "has no concise form");
}

// The fallback for the printer is to print it the longhand form.
void OpState::print(OpAsmPrinter *p) const {
  p->printDefaultOp(getOperation());
}

static StringMap<AbstractOperation> &getImpl(void *pImpl) {
  return *static_cast<StringMap<AbstractOperation> *>(pImpl);
}

OperationSet::OperationSet() { pImpl = new StringMap<AbstractOperation>(); }

OperationSet::~OperationSet() { delete &getImpl(pImpl); }

void OperationSet::addOperation(StringRef prefix, AbstractOperation opInfo) {
  assert(opInfo.name.startswith(prefix) && "op name doesn't start with prefix");

  if (!getImpl(pImpl).insert({opInfo.name, opInfo}).second) {
    llvm::errs() << "error: ops named '" << opInfo.name
                 << "' is already registered.\n";
    abort();
  }
}

/// Look up the specified operation in the operation set and return a pointer
/// to it if present.  Otherwise, return a null pointer.
const AbstractOperation *OperationSet::lookup(StringRef opName) const {
  auto &map = getImpl(pImpl);
  auto it = map.find(opName);
  if (it != map.end())
    return &it->second;
  return nullptr;
}
