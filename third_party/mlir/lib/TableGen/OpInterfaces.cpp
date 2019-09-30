//===- OpInterfaces.cpp - OpInterfaces class ------------------------------===//
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
//
// OpInterfaces wrapper to simplify using TableGen OpInterfaces.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/OpInterfaces.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

OpInterfaceMethod::OpInterfaceMethod(const llvm::Record *def) : def(def) {
  llvm::DagInit *args = def->getValueAsDag("arguments");
  for (unsigned i = 0, e = args->getNumArgs(); i != e; ++i) {
    arguments.push_back(
        {llvm::cast<llvm::StringInit>(args->getArg(i))->getValue(),
         args->getArgNameStr(i)});
  }
}

StringRef OpInterfaceMethod::getReturnType() const {
  return def->getValueAsString("returnType");
}

// Return the name of this method.
StringRef OpInterfaceMethod::getName() const {
  return def->getValueAsString("name");
}

// Return if this method is static.
bool OpInterfaceMethod::isStatic() const {
  return def->isSubClassOf("StaticInterfaceMethod");
}

// Return the body for this method if it has one.
llvm::Optional<StringRef> OpInterfaceMethod::getBody() const {
  auto value = def->getValueAsString("body");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}

// Return the description of this method if it has one.
llvm::Optional<StringRef> OpInterfaceMethod::getDescription() const {
  auto value = def->getValueAsString("description");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}

ArrayRef<OpInterfaceMethod::Argument> OpInterfaceMethod::getArguments() const {
  return arguments;
}

bool OpInterfaceMethod::arg_empty() const { return arguments.empty(); }

OpInterface::OpInterface(const llvm::Record *def) : def(def) {
  auto *listInit = dyn_cast<llvm::ListInit>(def->getValueInit("methods"));
  for (llvm::Init *init : listInit->getValues())
    methods.emplace_back(cast<llvm::DefInit>(init)->getDef());
}

// Return the name of this interface.
StringRef OpInterface::getName() const {
  return def->getValueAsString("cppClassName");
}

// Return the methods of this interface.
ArrayRef<OpInterfaceMethod> OpInterface::getMethods() const { return methods; }

// Return the description of this method if it has one.
llvm::Optional<StringRef> OpInterface::getDescription() const {
  auto value = def->getValueAsString("description");
  return value.empty() ? llvm::Optional<StringRef>() : value;
}
