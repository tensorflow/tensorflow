//===- Types.cpp - MLIR Type Classes --------------------------------------===//
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

#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/STLExtras.h"
using namespace mlir;

void Type::print(raw_ostream &os) const {
  switch (getKind()) {
  case TypeKind::I1:   os << "i1"; return;
  case TypeKind::I8:   os << "i8"; return;
  case TypeKind::I16:  os << "i16"; return;
  case TypeKind::I32:  os << "i32"; return;
  case TypeKind::I64:  os << "i64"; return;
  case TypeKind::Int:  os << "int"; return;
  case TypeKind::BF16: os << "bf16"; return;
  case TypeKind::F16:  os << "f16"; return;
  case TypeKind::F32:  os << "f32"; return;
  case TypeKind::F64:  os << "f64"; return;
  case TypeKind::Function: {
    auto *func = cast<FunctionType>(this);
    os << '(';
    interleave(func->getInputs(),
               [&](Type *type) { os << *type; },
               [&]() { os << ", "; });
    os << ") -> ";
    auto results = func->getResults();
    if (results.size() == 1)
      os << *results[0];
    else {
      os << '(';
      interleave(results,
                 [&](Type *type) { os << *type; },
                 [&]() { os << ", "; });
      os << ")";
    }
    return;
  }
  case TypeKind::Vector: {
    auto *v = cast<VectorType>(this);
    os << "vector<";
    for (auto dim : v->getShape())
      os << dim << 'x';
    os << *v->getElementType() << '>';
    return;
  }
  }
}

void Type::dump() const {
  print(llvm::errs());
}
