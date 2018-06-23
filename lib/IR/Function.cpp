//===- Function.cpp - MLIR Function Classes -------------------------------===//
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

#include "mlir/IR/Function.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;

Function::Function(StringRef name, FunctionType *type)
  : name(name.str()), type(type) {
}

void Function::print(raw_ostream &os) {
  os << "extfunc @" << name << '(';
  interleave(type->getInputs(),
             [&](Type *eltType) { os << *eltType; },
             [&]() { os << ", "; });
  os << ')';

  switch (type->getResults().size()) {
  case 0: break;
  case 1:
    os << " -> " << *type->getResults()[0];
    break;
  default:
    os << " -> (";
    interleave(type->getResults(),
               [&](Type *eltType) { os << *eltType; },
               [&]() { os << ", "; });
    os << ')';
    break;
  }

  os << "\n";
}

void Function::dump() {
  print(llvm::errs());
}
